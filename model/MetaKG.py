import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_sum
from torch_scatter.utils import broadcast
from collections import OrderedDict

class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()

    def forward(self, entity_agg, user_emb):
        attention_scores = F.pairwise_distance(entity_agg, user_emb, p=2)
        attention_weights = F.softmax(-attention_scores, dim=0)
        attention_output = torch.matmul(attention_weights, entity_agg)
        return attention_output

class CFModule(nn.Module):
    def __init__(self, n_users, n_items):
        super(CFModule, self).__init__()

    def forward(self, user_emb, interact_mat, weight, n_items):
        mat_row = interact_mat._indices()[0, :]
        mat_col = interact_mat._indices()[1, :]
        item_neigh_emb = user_emb[mat_row] * weight[0]
        i_u_agg = scatter_sum(src=item_neigh_emb, index=mat_col, dim_size=n_items, dim=0)
        return i_u_agg

class Aggregator(nn.Module):
    def __init__(self, n_users, n_items, triplet_attention, use_gate, use_cf_module, use_attention):
        super(Aggregator, self).__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.triplet_attention = triplet_attention
        self.use_gate = use_gate
        self.use_cf_module = use_cf_module
        self.use_attention = use_attention

        self.gate1 = nn.Linear(64, 64, bias=False)
        self.gate2 = nn.Linear(64, 64, bias=False)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)

        if self.use_cf_module:
            self.cf_module = CFModule(n_users, n_items)

        if self.use_attention:
            self.attention = Attention()

    def scatter_softmax(self, src, index, dim: int = -1, eps: float = 1e-12, max_per_element=None):
        if max_per_element is None:
            max_per_element = src.max()
        if not torch.is_floating_point(src):
            raise ValueError('`scatter_softmax` can only be computed over tensors '
                             'with floating point data types.')

        index = broadcast(index, src, dim)

        max_value_per_index = torch.scatter_add(src, index, dim=dim)[0]
        max_per_src_element = max_value_per_index.gather(dim, index)

        recentered_scores = src - max_per_element
        recentered_scores_exp = recentered_scores.exp()

        sum_per_index = scatter_sum(recentered_scores_exp, index, dim)
        normalizing_constants = sum_per_index.add_(eps).gather(dim, index)

        return recentered_scores_exp.div(normalizing_constants)

    def KG_forward(self, entity_emb, edge_index, edge_type, weight):
        n_entities = entity_emb.shape[0]

        head, tail = edge_index
        edge_relation_emb = weight[edge_type]
        neigh_relation_emb = entity_emb[tail] * edge_relation_emb
        entity_agg = scatter_sum(src=neigh_relation_emb, index=head, dim_size=n_entities, dim=0)

        return entity_agg

    def CF_forward(self, user_emb, interact_mat, weight):
        mat_row = interact_mat._indices()[0, :]
        mat_col = interact_mat._indices()[1, :]
        item_neigh_emb = user_emb[mat_row] * weight[0]
        i_u_agg = scatter_sum(src=item_neigh_emb, index=mat_col, dim_size=self.n_items, dim=0)
        return i_u_agg

    def forward(self, entity_emb, user_emb, edge_index, edge_type, interact_mat, weight, fast_weights=None, i=0, mat_row=None, mat_col=None, mat_val=None):
        """KG aggregate"""
        entity_agg = self.KG_forward(entity_emb, edge_index, edge_type, weight)

        """user aggregate"""
        if self.use_cf_module:
            i_u_agg = self.CF_forward(user_emb, interact_mat, weight)

            if fast_weights is None:
                gi = self.leaky_relu(self.gate1(entity_agg[:self.n_items]) + self.gate2(i_u_agg))
            else:
                gate1_name = 'convs.{}.gate1.weight'.format(str(i))
                gate2_name = 'convs.{}.gate2.weight'.format(str(i))
                conv_w1 = fast_weights[gate1_name]
                conv_w2 = fast_weights[gate2_name]
                gi = self.leaky_relu(F.linear(entity_agg[:self.n_items], conv_w1) + F.linear(i_u_agg, conv_w2))

            item_emb_fusion = (gi * entity_agg[:self.n_items]) + ((1 - gi) * i_u_agg)
            user_item_mat = torch.sparse.FloatTensor(torch.cat([mat_row, mat_col]).view(2, -1),
                                                     torch.ones_like(mat_val),
                                                     size=[self.n_users, self.n_items])
            user_agg = torch.sparse.mm(user_item_mat, item_emb_fusion)

            entity_agg = torch.cat([item_emb_fusion, entity_agg[self.n_items:]])
        else:
            user_agg = torch.sparse.mm(interact_mat, entity_emb)

        """Attention mechanism"""
        if self.use_attention:
            entity_agg = self.attention(entity_agg, user_emb)

        return entity_agg, user_agg

class GraphConv(nn.Module):
    def __init__(self, channel, n_hops, n_users, n_relations, n_items, use_gate,
                 node_dropout_rate=0.5, mess_dropout_rate=0.1):
        super(GraphConv, self).__init__()

        self.convs = nn.ModuleList()
        self.n_relations = n_relations
        self.n_users = n_users
        self.n_items = n_items
        self.node_dropout_rate = node_dropout_rate
        self.mess_dropout_rate = mess_dropout_rate

        weight = nn.init.xavier_uniform_(torch.empty(n_relations, channel))
        self.weight = nn.Parameter(weight)

        for i in range(n_hops):
            self.convs.append(Aggregator(n_users=n_users, n_items=n_items, triplet_attention=self.triplet_attention, use_gate=use_gate))

        self.dropout = nn.Dropout(p=mess_dropout_rate)

    def forward(self, user_emb, entity_emb, edge_index, edge_type,
                interact_mat, fast_weights=None, mess_dropout=True, node_dropout=True):
        if node_dropout:
            edge_index, edge_type = self._edge_sampling(edge_index, edge_type, self.node_dropout_rate)

        entity_res_emb = entity_emb
        user_res_emb = user_emb

        for i in range(len(self.convs)):
            entity_emb, user_emb = self.convs[i](entity_emb, user_emb,
                                                 edge_index, edge_type, interact_mat,
                                                 self.weight, fast_weights, i=i)

            if mess_dropout:
                entity_emb = self.dropout(entity_emb)
                user_emb = self.dropout(user_emb)
            entity_emb = F.normalize(entity_emb)
            user_emb = F.normalize(user_emb)

            entity_res_emb = torch.add(entity_res_emb, entity_emb)
            user_res_emb = torch.add(user_res_emb, user_emb)

        return entity_res_emb, user_res_emb

class Recommender(nn.Module):
    def __init__(self, data_config, args_config, graph, user_pre_embed, item_pre_embed):
        super(Recommender, self).__init__()

        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']
        self.n_relations = data_config['n_relations']
        self.n_entities = data_config['n_entities']
        self.n_nodes = data_config['n_nodes']
        self.user_pre_embed = user_pre_embed
        self.item_pre_embed = item_pre_embed

        self.num_inner_update = args_config.num_inner_update
        self.meta_update_lr = args_config.meta_update_lr

        self.decay = args_config.l2
        self.emb_size = args_config.dim
        self.context_hops = args_config.context_hops
        self.use_gate = args_config.use_gate
        self.node_dropout = args_config.node_dropout
        self.node_dropout_rate = args_config.node_dropout_rate
        self.mess_dropout = args_config.mess_dropout
        self.mess_dropout_rate = args_config.mess_dropout_rate
        self.device = torch.device("cuda:" + str(args_config.gpu_id)) if args_config.cuda \
                                                                      else torch.device("cpu")

        self.edge_index, self.edge_type = self._get_edges(graph)
        self._init_weight()
        self.gcn = self._init_model()
        self.interact_mat = None

    def _init_weight(self):
        self.all_embed = nn.init.xavier_uniform_(torch.empty(self.n_nodes, self.emb_size))

        if self.user_pre_embed is not None and self.item_pre_embed is not None:
            entity_emb = self.all_embed[(self.n_users + self.n_items):, :]
            self.all_embed = torch.cat([self.user_pre_embed, self.item_pre_embed, entity_emb])

        self.all_embed = nn.Parameter(self.all_embed)

    def _init_model(self):
        return GraphConv(channel=self.emb_size,
                         n_hops=self.context_hops,
                         n_users=self.n_users,
                         n_relations=self.n_relations,
                         n_items=self.n_items,
                         use_gate=self.use_gate,
                         node_dropout_rate=self.node_dropout_rate,
                         mess_dropout_rate=self.mess_dropout_rate)

    def _get_edges(self, graph):
        graph_tensor = torch.tensor(list(graph.edges))
        index = graph_tensor[:, :-1]
        type = graph_tensor[:, -1]
        return index.t().long().to(self.device), type.long().to(self.device)

    def forward_kg(self, h, r, pos_t, neg_t):
        entity_emb = self.all_embed[self.n_users:, :]
        h_emb = entity_emb[h]
        r_emb = entity_emb[r]
        pos_t_emb = entity_emb[pos_t]
        neg_t_emb = entity_emb[neg_t]

        r_t_pos = pos_t_emb * r_emb
        r_t_neg = neg_t_emb * r_emb

        pos_score = torch.sum(torch.pow(r_t_pos - h_emb, 2), dim=1)
        neg_score = torch.sum(torch.pow(r_t_neg - h_emb, 2), dim=1)

        kg_loss = (-1.0) * F.logsigmoid(neg_score - pos_score)
        kg_loss = torch.mean(kg_loss)

        return kg_loss

    def forward_meta(self, support, query, fast_weights=None):
        user_s = support[0]
        pos_item_s = support[1]
        neg_item_s = support[2]
        user_q = query[0]
        pos_item_q = query[1]
        neg_item_q = query[2]

        user_emb = self.all_embed[:self.n_users, :]
        entity_emb = self.all_embed[self.n_users:, :]

        if fast_weights is None:
            fast_weights = self.get_parameter()

        for i in range(self.num_inner_update):
            entity_gcn_emb, user_gcn_emb = self.gcn(user_emb,
                                                    entity_emb,
                                                    self.edge_index,
                                                    self.edge_type,
                                                    self.interact_mat,
                                                    fast_weights=fast_weights,
                                                    mess_dropout=self.mess_dropout,
                                                    node_dropout=self.node_dropout)
            u_e = user_gcn_emb[user_s]
            pos_e, neg_e = entity_gcn_emb[pos_item_s], entity_gcn_emb[neg_item_s]
            loss, _, _ = self.create_bpr_loss(u_e, pos_e, neg_e)
            gradients = torch.autograd.grad(loss, fast_weights.values(), create_graph=False)

            fast_weights = OrderedDict(
                (name, param - self.meta_update_lr * grad)
                for ((name, param), grad) in zip(fast_weights.items(), gradients)
            )

        entity_gcn_emb, user_gcn_emb = self.gcn(user_emb,
                                                entity_emb,
                                                self.edge_index,
                                                self.edge_type,
                                                self.interact_mat,
                                                fast_weights=fast_weights,
                                                mess_dropout=self.mess_dropout,
                                                node_dropout=self.node_dropout)
        u_e = user_gcn_emb[user_q]
        pos_e, neg_e = entity_gcn_emb[pos_item_q], entity_gcn_emb[neg_item_q]
        loss, _, _ = self.create_bpr_loss(u_e, pos_e, neg_e)

        return loss

    # 这里开始添加新的优化代码

    def forward(self, batch=None, is_adapt=False):
        if is_adapt:
            user = batch['users']
            pos_item = batch['pos_items']
            neg_item = batch['neg_items']
        else:
            user = batch[0]
            pos_item = batch[1]
            neg_item = batch[2]

        user_emb = self.all_embed[:self.n_users, :]
        entity_emb = self.all_embed[self.n_users:, :]

        entity_gcn_emb, user_gcn_emb = self.gcn(user_emb,
                                                 entity_emb,
                                                 self.edge_index,
                                                 self.edge_type,
                                                 self.interact_mat,
                                                 mess_dropout=self.mess_dropout,
                                                 node_dropout=self.node_dropout)

        u_e = user_gcn_emb[user]
        pos_e, neg_e = entity_gcn_emb[pos_item], entity_gcn_emb[neg_item]
        loss, _, _ = self.create_bpr_loss(u_e, pos_e, neg_e)

        return loss

    def generate(self, adapt_fast_weight=None):
        user_emb = self.all_embed[:self.n_users, :]
        entity_emb = self.all_embed[self.n_users:, :]
        entity_gcn_emb, user_gcn_emb = self.gcn(user_emb,
                                                entity_emb,
                                                self.edge_index,
                                                self.edge_type,
                                                self.interact_mat,
                                                fast_weights=adapt_fast_weight,
                                                mess_dropout=False, node_dropout=False)

        return entity_gcn_emb, user_gcn_emb

    def rating(self, u_g_embeddings, i_g_embeddings):
        return torch.matmul(u_g_embeddings, i_g_embeddings.t())

    def create_bpr_loss(self, users, pos_items, neg_items):
        batch_size = users.shape[0]
        pos_scores = torch.sum(torch.mul(users, pos_items), axis=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), axis=1)

        mf_loss = -1 * torch.mean(nn.LogSigmoid()(pos_scores - neg_scores))

        # cul regularizer
        regularizer = (torch.norm(users) ** 2
                       + torch.norm(pos_items) ** 2
                       + torch.norm(neg_items) ** 2) / 2
        emb_loss = self.decay * regularizer / batch_size

        return mf_loss + emb_loss, mf_loss, emb_loss
