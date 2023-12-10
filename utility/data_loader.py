import numpy as np
from tqdm import tqdm
import networkx as nx
import scipy.sparse as sp

import random
from time import time
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

n_users = 0
n_items = 0
n_entities = 0
n_relations = 0
n_nodes = 0

n_user_dataset = {'amazon-book': 70679, 'last-fm': 23566, 'yelp2018': 45919}
n_item_dataset = {'amazon-book': 24915, 'last-fm': 48123, 'yelp2018': 45538}

def read_cf(file_name):
    inter_mat = list()
    lines = open(file_name, "r").readlines()
    for l in lines:
        tmps = l.strip()
        inters = [int(i) for i in tmps.split(" ")]

        u_id, pos_ids = inters[0], inters[1:]
        pos_ids = list(set(pos_ids))
        for i_id in pos_ids:
            inter_mat.append([u_id, i_id])

    return np.array(inter_mat)


def remap_item(train_data, test_data, dataset):
    global n_users, n_items
    n_users = n_user_dataset[dataset]
    n_items = n_item_dataset[dataset]

    train_user_set = defaultdict(list)
    test_user_set = defaultdict(list)

    #相同用户的与item的交互
    for u_id, i_id in train_data:
        train_user_set[int(u_id)].append(int(i_id))
    for u_id, i_id in test_data:
        test_user_set[int(u_id)].append(int(i_id))
    return train_user_set, test_user_set

def read_triplets(file_name):
    global n_entities, n_relations, n_nodes

    can_triplets_np = np.loadtxt(file_name, dtype=np.int32)
    can_triplets_np = np.unique(can_triplets_np, axis=0)  #去除重复的行

    if args.inverse_r:
        # get triplets with inverse direction like <entity, is-aspect-of, item>
        inv_triplets_np = can_triplets_np.copy()
        inv_triplets_np[:, 0] = can_triplets_np[:, 2] #取所有数组的第二维的第2位数据赋值给第0位
        inv_triplets_np[:, 2] = can_triplets_np[:, 0] #取所有数组的第二维的第0位数据赋值给第2位
        inv_triplets_np[:, 1] = can_triplets_np[:, 1] + max(can_triplets_np[:, 1]) + 1
        # consider two additional relations --- 'interact' and 'be interacted'
        can_triplets_np[:, 1] = can_triplets_np[:, 1] + 1
        inv_triplets_np[:, 1] = inv_triplets_np[:, 1] + 1
        # get full version of knowledge graph
        triplets = np.concatenate((can_triplets_np, inv_triplets_np), axis=0) #按照行的方向合并两个数组
    else:
        # consider two additional relations --- 'interact'.
        can_triplets_np[:, 1] = can_triplets_np[:, 1] + 1
        triplets = can_triplets_np.copy()

    #找出第0位元素和第2位元素最大值加一
    n_entities = max(max(triplets[:, 0]), max(triplets[:, 2])) + 1
    n_nodes = n_entities + n_users
    n_relations = max(triplets[:, 1]) + 1

    return triplets


def build_graph(train_data, triplets):
    ckg_graph = nx.MultiDiGraph()
    rd = defaultdict(list)

    print("Begin to load interaction triples ...")
    for u_id, i_id in tqdm(train_data, ascii=True): #tqmd==进度条
        rd[0].append([u_id, i_id])

    print("\nBegin to load knowledge graph triples ...")
    for h_id, r_id, t_id in tqdm(triplets, ascii=True):
        ckg_graph.add_edge(h_id, t_id, key=r_id)
        rd[r_id].append([h_id, t_id]) #将三元组的实体信息以及关系权重转化为二维数组的列表：比如：(255,0,65)--->{0:[255,65]}

    return ckg_graph, rd


def build_sparse_relational_graph(relation_dict):
    def _bi_norm_lap(adj):
        # D^{-1/2}AD^{-1/2}
        rowsum = np.array(adj.sum(1))

        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

        # bi_lap = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
        bi_lap = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)
        return bi_lap.tocoo()

    def _si_norm_lap(adj):
        # D^{-1}A
        rowsum = np.array(adj.sum(1)) #行相加

        d_inv = np.power(rowsum, -1).flatten() #取倒数
        d_inv[np.isinf(d_inv)] = 0. #无穷大处置为0
        d_mat_inv = sp.diags(d_inv)  #将d_inv构造成对角矩阵

        norm_adj = d_mat_inv.dot(adj)  #将d_mat_inv矩阵和adj矩阵相乘
        return norm_adj.tocoo() #转换成稀疏矩阵coo形式

    adj_mat_list = []
    print("Begin to build sparse relation matrix ...")  ##构建稀疏关系矩阵
    for r_id in tqdm(relation_dict.keys()):
        np_mat = np.array(relation_dict[r_id])
        if r_id == 0:
            cf = np_mat.copy()
            cf[:, 1] = cf[:, 1] + n_users  # [0, n_items) -> [n_users, n_users+n_items)
            vals = [1.] * len(cf)
            adj = sp.coo_matrix((vals, (cf[:, 0], cf[:, 1])), shape=(n_nodes, n_nodes))
        else:
            vals = [1.] * len(np_mat)
            adj = sp.coo_matrix((vals, (np_mat[:, 0], np_mat[:, 1])), shape=(n_nodes, n_nodes))  #转化为coo稀疏矩阵
        adj_mat_list.append(adj)

    # norm_mat_list = [_bi_norm_lap(mat) for mat in adj_mat_list]
    # mean_mat_list = [_si_norm_lap(mat) for mat in adj_mat_list]
    mean_mat_list = _si_norm_lap(adj_mat_list[0])
    # interaction: user->item, [n_users, n_entities]
    # norm_mat_list[0] = norm_mat_list[0].tocsr()[:n_users, n_users:].tocoo()
    mean_mat_list = mean_mat_list.tocsr()[:n_users, n_users:].tocoo()

    return adj_mat_list, mean_mat_list


def load_data(model_args, state):
    global args
    args = model_args
    directory = args.data_path + args.dataset + '/' #读取数据集的路径

    print('reading train and test user-item set ...')
    train_cf = read_cf(directory + 'test_scenario/' + state + '_support.txt') #读取meta_training_support.txt文件
    test_cf = read_cf(directory + 'test_scenario/' + state + '_query.txt')    #读取meta_training_query.txt文件
    train_user_set, test_user_set = remap_item(train_cf, test_cf, args.dataset)

    print('combinating train_cf and kg data ...')
    triplets = read_triplets(directory + 'kg_final.txt')

    print('building the graph ...')
    graph, relation_dict = build_graph(train_cf, triplets)

    print('building the adj mat ...')
    adj_mat_list, mean_mat_list = build_sparse_relational_graph(relation_dict)

    n_params = {
        'n_users': int(n_users),
        'n_items': int(n_items),
        'n_entities': int(n_entities),
        'n_nodes': int(n_nodes),
        'n_relations': int(n_relations)
    }
    user_dict = {
        'train_user_set': train_user_set,
        'test_user_set': test_user_set
    }

    return train_cf, test_cf, user_dict, n_params, graph, \
           [adj_mat_list, mean_mat_list]
