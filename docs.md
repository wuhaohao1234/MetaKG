就是基于MetaKG这个模型改进，加入协同过滤的模块和注意力机制，将数据集换成CiteUlike 或者dblp数据集之后，能够让模型有所优化

1. **注意力模块（`Attention` 类）：**
   - 基于 `entity_agg` 和 `user_emb` 之间的成对距离计算注意力分数。
   - 应用 softmax 得到注意力权重。
   - 根据注意力权重计算基于 `entity_agg` 的加权和。

2. **协同过滤模块（`CFModule` 类）：**
   - 利用用户嵌入和交互矩阵执行协同过滤。
   - 基于用户交互聚合物品嵌入。

3. **聚合器模块（`Aggregator` 类）：**
   - 整合知识图谱信息和协同过滤。
   - 可选地使用门机制和注意力机制。
   - 定义了 KG 聚合（`KG_forward`）、CF 聚合（`CF_forward`）以及整体前向传播的方法。

4. **图卷积模块（`GraphConv` 类）：**
   - 通过迭代应用聚合器执行图卷积。
   - 支持可选的丢弃机制，如节点丢弃率和消息丢弃率。

5. **推荐器模块（`Recommender` 类）：**
   - 使用参数初始化模型，包括用户和物品嵌入。
   - 整合了知识图谱和用户-物品交互信息。
   - 定义了 KG 损失（`forward_kg`）、元学习损失（`forward_meta`）以及整体前向传播（`forward`）的方法。
   - 包含生成嵌入（`generate`）和计算用户和物品之间评分（`rating`）的方法。
   - 在训练期间计算 BPR 损失。

6. **BPR 损失（`create_bpr_loss` 方法）：**
   - 计算贝叶斯个性化排名（BPR）损失，这在协同过滤中常用。

7. **训练和适应：**
   - 该模型支持训练和适应两种情景。
   - 在适应过程中，模型通过元学习方法执行内部更新以获取快速权重。

8. **一般工作流程：**
   - 用户和物品嵌入最初来自知识图谱。
   - 该模型通过迭代应用图卷积结合 KG 和协同过滤信息。
   - 训练过程中根据 BPR 损失和正则化项优化参数。
   - 该模型能够生成用户和物品的嵌入，并为推荐生成评分。
