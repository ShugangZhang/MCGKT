import math
import os
import logging
import numpy as np
import torch
import torch.nn.functional as F

from torch.nn import Module, Embedding, Parameter, Sequential, Linear, ReLU, \
    Dropout, MultiheadAttention, GRU
from torch.nn.init import kaiming_normal_
from torch.nn.functional import one_hot, binary_cross_entropy
from sklearn import metrics
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, GPSConv, LightGCN, SuperGATConv, LGConv

logger = logging.getLogger('main.eval')

def mlp(in_size, out_size):
    return Sequential(
        Linear(in_size, out_size),
        ReLU(),
        Dropout(),
        Linear(out_size, out_size),
    )


class MYGKT(Module):
    '''
        Args:
            num_q: the total number of the questions(KCs) in the given dataset
            emb_size: the dimension of the embedding vectors in this model
            hidden_size: the dimension of the hidden vectors in this model
            num_attn_heads:
                the number of the attention heads in the multi-head attention
                module in this model.
                This argument would be used when the method is MHA.

        Note that this implementation is not exactly the same as the original
        paper. The erase-add gate was not implemented since the reason for
        the gate can't be found. And the batch normalization in MLP was not
        implemented because of the simplicity.
    '''
    def __init__(self, num_q, num_p, adj_trans, pq_rel, hidden_size, num_attn_heads, method):
        super().__init__()
        self.num_q = num_q
        self.hidden_size = hidden_size

        self.x_emb_layer = Embedding(self.num_q * 2, self.hidden_size)  # 每个concept，连同对错，都要编码，所以是两倍

        self.q_emb_layer = Embedding(self.num_q, self.hidden_size)
        # self.q_emb = Parameter(torch.Tensor(self.num_q, self.hidden_size))
        # kaiming_normal_(self.q_emb)

        self.init_h = Parameter(torch.Tensor(self.num_q, self.hidden_size))

        self.mlp_self = mlp(self.hidden_size * 2, self.hidden_size)

        self.gru = GRU(
            self.hidden_size * 2,
            self.hidden_size,
            batch_first=True
        )  # 感觉这个GRU用的很奇怪

        self.tgru = GRU(
            self.hidden_size,
            self.hidden_size,
            batch_first=True
        )

        self.mlp_pred = mlp(self.hidden_size * 2, self.hidden_size)  # 用来替代GRU

        self.bias = Parameter(torch.Tensor(1, self.num_q, 1))
        self.out_layer = Linear(self.hidden_size, 1, bias=False)

        # 图卷积相关
        self.gcn1 = GCNConv(self.hidden_size, self.hidden_size)
        self.gcn2 = GCNConv(self.hidden_size, self.hidden_size)
        self.adj = torch.from_numpy(adj_trans).to('cuda')  # 最好是能单写一个class，不想公用这个
        # self.adj = Parameter(torch.Tensor(self.num_q, self.num_q))  # 自学习邻接矩阵  .triu(diagonal=1)
        self.num_p = num_p
        self.bias_p = Parameter(torch.Tensor(1, self.num_p, 1))
        self.x_pemb = Embedding(self.num_p * 2, self.hidden_size)  # 每个problem，连同对错，都要编码，所以是两倍
        # self.p_emb = Parameter(torch.Tensor(self.num_p, self.hidden_size))  # 待优化成node2vec或GCN
        self.p_emb_layer = Embedding(self.num_p, self.hidden_size)   # 不理解有何区别
        self.out_layer_p = Linear(self.hidden_size * 2, 1, bias=False)
        # self.adj_p = torch.from_numpy(adj_p).to('cuda')
        # self.adj_p[self.adj_p != 0] = 1

        self.compress = Linear(self.num_q * self.hidden_size, self.hidden_size)
        self.uncompre = Linear(self.hidden_size, self.num_q * self.hidden_size)
        self.pq_rel = torch.from_numpy(pq_rel).to('cuda')

        self.pq_int_rel = torch.Tensor(self.num_p * 2, self.num_q * 2)  # [17751*2] * [123*2]
        self.pq_int_rel[:self.num_p,:self.num_q] = self.pq_rel
        self.pq_int_rel[-self.num_p:,-self.num_q:] = self.pq_rel

        # self.theta = Parameter(torch.randn(1)*0.1)
        self.theta = Parameter(torch.tensor(0.0))

        self.mlp_outgo = mlp(self.hidden_size * 4, self.hidden_size)
        self.mlp_income = mlp(self.hidden_size * 4, self.hidden_size)

    def forward1(self, q, r, p):  # 原方案备份，编码concept
        '''mpt
            Args:
                q: the question(KC) sequence with the size of [batch_size, n]
                r: the response sequence with the size of [batch_size, n]

            Returns:
                y: the knowledge level about the all questions(KCs)
                h: the hidden states of the all questions(KCs)
        '''
        batch_size = q.shape[0]

        # 暂时先不用任何encoder——后续准备用图卷积作为encoder编码问题（暂时不包括答案）
        # 送入图卷积
        edge_index = self.adj_p.nonzero(as_tuple=False).t()  # 注意adj_p不是对陈的
        # edge_weight = self.adj_p[edge_index[0], edge_index[1]]

        p_emb = self.gcn1(self.p_emb, edge_index)
        p_emb = F.relu(p_emb)
        p_emb = self.gcn2(p_emb, edge_index)

        p_emb = p_emb.unsqueeze(0).repeat(batch_size, 1, 1)

        p_idx = p.unsqueeze(-1).repeat(1, 1, p_emb.shape[-1])
        p_emb = torch.gather(p_emb, dim=1, index=p_idx)
        # p_onehot = one_hot(p, self.num_p)

        # 原方案，编码concept
        x = q + self.num_q * r  # 每个concept连同各自答案，一起编码，共有220种可能；乘以num_q是为了把对和错分开编码，且不会重叠  # 此处待修改


        x_emb = self.x_emb(x)  # 110个concept，每个有对错，共计会有220个编码
        q_emb = self.q_emb.unsqueeze(0).repeat(batch_size, 1, 1)

        q_onehot = one_hot(q, self.num_q)

        ht = self.init_h.unsqueeze(0).repeat(batch_size, 1, 1)
        h = [ht]
        y = []
        a = []

        for xt_emb, qt, qt_onehot, pt_emb in zip(
                x_emb.permute(1, 0, 2), q.permute(1, 0), q_onehot.permute(1, 0, 2), p_emb.permute(1, 0, 2)
        ):  # 转置后，可对batch内所有样本sequence的同一位置集中处理
            # xt_emb:concept+答案 编码；qt_onehot:concept索引，用于定位concept；q_emb:concept编码；ht:状态编码

            # ----------------------尝试改成普通GCN-----------------------
            """
            ht_ = self.aggregate(xt_emb, qt_onehot, q_emb, ht)

            # 处理邻接矩阵
            adj = self.adj + self.adj.t()
            adj = torch.sigmoid(adj)
            # 设置对角线
            mask = torch.eye(adj.size(0), dtype=bool, device=adj.device)
            adj = adj.masked_fill(mask, 1.0)  # 没有管对角线约束


            # 稀疏
            edge_index = self.adj.nonzero(as_tuple=False).t()
            # edge_weight = adj[edge_index[0], edge_index[1]]

            # sparse_adj = self.adj.to_sparse()
            # edge_index = sparse_adj.indices()
            # edge_weight = sparse_adj.values()

            ht_ = self.gcn1(ht_, edge_index)  # 邻接矩阵待修改   # 256,110,60    256,110,110
            ht_ = F.relu(ht_)
            ht_ = self.gcn2(ht_, edge_index)
            ht, _ = self.gru(torch.cat([ht, ht_],dim=-1))  # ht需要更新
            yt = self.predict(ht)
            """

            # ----------------------原操作-----------------------
            ht_ = self.aggregate(xt_emb, qt_onehot, q_emb, pt_emb, ht)  # q_emb和ht是两个独立状态，和batch无关，也没有经历过转置等操作
            ht = self.update(ht, ht_, qt, qt_onehot)  # ht随着时间变化，不断更新迭代（在本循环内保有范围）
            yt = self.predict(ht)
            at = self.predict_adj_trans(ht)  # predicted adj_trans

            h.append(ht)
            y.append(yt)
            a.append(at)

        h = torch.stack(h, dim=1)  # 按照时间序列堆叠起来，注意共有101个时序状态，比y多一个
        y = torch.stack(y, dim=1)  # 按照时间序列堆叠起来
        a = torch.stack(a, dim=1)

        return y, h, a  # 掩码都是后来的事情，这里不用管

    # one—by-one 存在泄漏问题
    def forward(self, p, r):
        '''mpt
            Args:
                q: the question(KC) sequence with the size of [batch_size, n]
                r: the response sequence with the size of [batch_size, n]

            Returns:
                y: the knowledge level about the all questions(KCs)
                h: the hidden states of the all questions(KCs)
        '''
        batch_size = p.shape[0]

        # 原方案，编码concept
        p_onehot = one_hot(p, self.num_p)  # [64,100] => [64,100,17751]
        q_onehot = torch.matmul(p_onehot.float(), self.pq_rel.float())  # [64,100,17751] * [17751,123] => [64,100,123]  将每时刻习题转换成概念点
        # xq_onehot = q_onehot + r

        # xp = p + self.num_p * r  # [64,100] 每个batch的问题连同各自答案，一起编码；乘以num_p是为了把对和错分开编码，且不会重叠
        # xp_onehot = one_hot(xp, self.num_p * 2)  # [64,100,17751*2]
        # xq_onehot = torch.matmul(xp_onehot.float(), self.pq_int_rel.float())  # [64,100,123*2]  每时刻每个interaction对应的所有concept interaction


        xq_no = torch.arange(self.num_q * 2, device=p.device).long()  # 0~245  所有概念连同对错的编码
        xq_emb = self.x_emb_layer(xq_no)  # 所有246个inter的embedding  # [246,30]

        # x_emb = torch.matmul(xq_onehot, xq_emb)  #     [64,100,246] * [246,30]  => [64,100,30]

        q_no = torch.arange(self.num_q, device=p.device).long()  # 0~122
        q_emb = self.q_emb_layer(q_no)  # [123, 30]  所有123个concept的embedding

        p_emb = torch.mm(self.pq_rel.float(), q_emb)    # [17751, 123] * [123,30] => [17751,30]

        q_emb = q_emb.unsqueeze(0).repeat(batch_size, 1, 1)  # [64,123,30]


        # 新方案，编码problem
        """
        p_no = torch.arange(self.num_p, device=p.device)
        p_emb = self.p_emb_layer(p_no)  # 所有问题的embedding
        
        
        # xp = p + self.num_p * r
        # xp_emb = self.xp_emb_layer(xp)  # N个问题，每个有对错，共计会有2*N个编码
        # q_emb应该来源于p_emb，而不是初始化了
        # qp_rel = self.pq_rel.t()   # [110,16891]
        # row_sum = avg_qp_rel.sum(dim=1, keepdim=True)
        # avg_qp_rel = avg_qp_rel / (row_sum + 1e-6)
        # concept的embedding来源于所有关联问题的embedding的平均聚合
        q_emb = torch.mm(self.pq_rel.t().float(), p_emb) # [123,17751] * [17751,30] = [123,30]
        q_emb = q_emb.unsqueeze(0).repeat(batch_size, 1, 1)

        p_onehot = one_hot(p, self.num_p)  # [64,100,17751]
        q_onehot = torch.matmul(p_onehot, self.pq_rel)  # [64,100,17751] * [17751, 123] => [64,100,123]  并非严格的onehot，仅表示当前问题对应哪些concept
        """

        ht = self.init_h.unsqueeze(0).repeat(batch_size, 1, 1)
        h = [ht]
        y = []

        for pt, rt, qt_onehot in zip(
            p.permute(1, 0), r.permute(1, 0), q_onehot.permute(1, 0, 2)
        ):  # 转置后，可对batch内所有样本sequence的同一位置集中处理
            # xt_emb:concept+答案 编码；qt_onehot:concept索引于定位concept；q_emb:concept编码；ht:状态编码

            # ----------------------尝试改成普通GCN-----------------------
            """
            ht_ = self.aggregate(xt_emb, qt_onehot, q_emb, ht)
            
            # 处理邻接矩阵
            adj = self.adj + self.adj.t()
            adj = torch.sigmoid(adj)
            # 设置对角线
            mask = torch.eye(adj.size(0), dtype=bool, device=adj.device)
            adj = adj.masked_fill(mask, 1.0)  # 没有管对角线约束
        

            # 稀疏
            edge_index = self.adj.nonzero(as_tuple=False).t()
            # edge_weight = adj[edge_index[0], edge_index[1]]

            # sparse_adj = self.adj.to_sparse()
            # edge_index = sparse_adj.indices()
            # edge_weight = sparse_adj.values()

            ht_ = self.gcn1(ht_, edge_index)  # 邻接矩阵待修改   # 256,110,60    256,110,110
            ht_ = F.relu(ht_)
            ht_ = self.gcn2(ht_, edge_index)
            ht, _ = self.gru(torch.cat([ht, ht_],dim=-1))  # ht需要更新
            yt = self.predict(ht)
            """

            # ----------------------原操作-----------------------
            ht_ = self.aggregate(rt, qt_onehot, q_emb, xq_emb, ht)  # q_emb和ht是两个独立状态，和batch无关，也没有经历过转置等操作
            ht = self.update(ht, ht_, qt_onehot)  # ht随着时间变化，不断更新迭代（在本循环内保有范围）
            # ht = self.update_with_seq(h, ht_, qt, qt_onehot)  # ht由所有历史记录而来，而不只是上一时刻特征，区别只有ht替换成了h
            # yt = self.predict(ht)  # 回答概念
            yt = self.predict_p(ht)  # 回答具体问题
            # yt = self.predict_p_with_qemb(ht, pt, p_emb)  # 结合问题embedding回答具体问题
            # yt = self.predict_p_with_hist(ht, h, p_emb, p) # 结合历史状态回答当前问题

            # 规避batch内唯一元素问题  Glory
            if(len(yt.shape) == 1):  # [16891] => [1,16891]
                yt = yt.unsqueeze(0)

            h.append(ht)
            y.append(yt)

        h = torch.stack(h, dim=1)  # 按照时间序列堆叠起来，注意共有101个时序状态，比y多一个
        y = torch.stack(y, dim=1)  # 按照时间序列堆叠起来

        return y, h  # 掩码都是后来的事情，这里不用管


    def aggregate(self, rt, qt_onehot, q_emb, xq_emb, ht):
        batch = ht.shape[0]

        # 未答对
        xt_emb_ncorrect = xq_emb[:self.num_q].unsqueeze(0).repeat(batch, 1,1)  # [64,123,30]
        # 答对
        xt_emb_correct = xq_emb[-self.num_q:].unsqueeze(0).repeat(batch,1,1)  # [64,123,30]

        # 聚合xt_emb
        rt = rt.unsqueeze(-1).unsqueeze(-1)
        xt_emb = (1-rt) * xt_emb_ncorrect + rt * xt_emb_correct  # [64,123,30]  inter编码根据回答正确与否，只保留对或者错其中一种

        qt_onehot = qt_onehot.unsqueeze(-1)  # [64,123,1]

        ht_ = qt_onehot * torch.cat([ht, xt_emb], dim=-1) + \
            (1 - qt_onehot) * torch.cat([ht, q_emb], dim=-1)  # 对回答节点整合[concept+回答]表征；其他所有节点，整合concept的表征

        # 加入问题编码
        return ht_

    def aggregate1(self, xt_emb, qt_onehot, q_emb, ht):  # 备份：未加入问题编码
        xt_emb = xt_emb.unsqueeze(1).repeat(1, self.num_q, 1)  # concept+回答 的embedding

        qt_onehot = qt_onehot.unsqueeze(-1)

        ht_ = qt_onehot * torch.cat([ht, xt_emb], dim=-1) + \
            (1 - qt_onehot) * torch.cat([ht, q_emb], dim=-1)  # 对回答节点整合[concept+回答]表征；其他所有节点，整合concept的表征

        # 加入问题编码
        return ht_

    def f_self(self, ht_):
        return self.mlp_self(ht_)

    def f_neighbor(self, ht_, qt_onehot):
        """
        Args:
            ht_: [64,123,30]
            qt_onehot: 当前时刻对应的所有concept [64,123]
        """
        batch_size = qt_onehot.shape[0]

        Aij = self.adj  # [123,123]

        # 执行mask操作，从[64,123,60]中执行[64,123,1]，将非主角concept全置为0
        ht_masked = ht_ * qt_onehot  # [64,123,60]

        outgo_mask = torch.matmul(Aij, ht_masked).sum(-1)  # [64,123,60]
        outgo_mask[outgo_mask != 0] = 1
        outgo_mask = outgo_mask.unsqueeze(-1)

        outgo_part = torch.cat([ht_, torch.matmul(Aij, ht_masked)], dim=-1)  # [64,123,30*4]  聚合了和每个concept相关的“主角concept”
        # outgo_part = torch.matmul(Aij, self.mlp_outgo(outgo_part))
        outgo_part = outgo_mask * self.mlp_outgo(outgo_part)

        income_mask = torch.matmul(Aij.T, ht_masked).sum(-1)  # [64,123,60]
        income_mask[income_mask != 0] = 1
        income_mask = income_mask.unsqueeze(-1)

        income_part = torch.cat([ht_, torch.matmul(Aij.T, ht_masked)], dim=-1)  # [64,123,30*4]
        # income_part = torch.matmul(Aij.T, self.mlp_income(income_part))
        income_part = income_mask * self.mlp_income(income_part)

        return outgo_part + income_part

    def update(self, ht, ht_, qt_onehot):
        # ht_是用来更新特征的（生成m），ht是原特征
        qt_onehot = qt_onehot.unsqueeze(-1)

        m = qt_onehot * self.f_self(ht_) + \
            (1 - qt_onehot) * self.f_neighbor(ht_, qt_onehot)

        # 非典型GRU用法
        ht, _ = self.gru(torch.cat([m, ht], dim=-1))  # [64,110,30]
        # 改为普通MLP
        # ht = self.mlp_pred(torch.cat([m, ht], dim=-1))

        return ht

    def update_with_seq(self, h, ht_, qt, qt_onehot):
        # ht_是用来更新特征的（生成m），ht是原特征
        qt_onehot = qt_onehot.unsqueeze(-1)

        m = qt_onehot * self.f_self(ht_) + \
            (1 - qt_onehot) * self.f_neighbor(ht_, qt)

        # 典型GRU用法，按照时序来组织更新
        # h = torch.stack(h, dim=1)  # 测试一下传递问题：不会有影响，甚至重名都没问题，因为这里的h已经被绑定到了新的内存空间上（对应新创建的对象）
        # B, T, C, F = h.shape
        # h_reshaped = h.permute(0, 2, 1, 3)
        # h_reshaped = h_reshaped.contiguous().view(B*C, T, F)
        # h_reshaped, _ = self.tgru(h_reshaped)  # 经历一轮gru，注意是时序维度，有别于下面的概念维度
        # h = h_reshaped.view(B, C, T, F).permute(0, 2, 1, 3)
        # ht = h[:, -1, :, :]

        h = torch.stack(h, dim=1)
        B, T, C, F = h.shape
        h = h.view(B, T, -1)
        h = self.compress(h)
        h, _ = self.tgru(h)
        h = self.uncompre(h)
        h = h.view(B, T, C, F)
        ht = h[:,-1,:,:]

        # 非典型GRU用法
        ht, _ = self.gru(torch.cat([m, ht], dim=-1))  # [64,110,30]
        # 改为普通MLP
        # ht = self.mlp_pred(torch.cat([m, ht], dim=-1))

        return ht

    def predict(self, ht):
        return torch.sigmoid(self.out_layer(ht) + self.bias).squeeze()

    def predict_p_with_qemb(self, ht, pt, p_emb):
        # 将state转化为具体问题
        pq_rel = self.pq_rel.float()  # [16891,110]

        p_emb = p_emb.unsqueeze(0).repeat(ht.shape[0],1,1)  # [64,17751,30]

        # q_emb = q_emb[0]  # 去掉batch，防止混淆  # [110,30]

        # p_emb = torch.mm(pq_rel, q_emb).unsqueeze(0).repeat(ht.shape[0], 1, 1) # [64, 16891,30]

        # ----------------并入问题本身的embedding-------------------
        # 方案一，直接初始化tensor
        # p_emb = self.p_emb.unsqueeze(0).repeat(ht.shape[0], 1, 1)

        # 方案二，通过embedding编码problem
        # p_idx = torch.arange(self.num_p, device=pq_rel.device)
        # p_emb = self.p_emb(p_idx)   # [16891,30]
        # p_emb = p_emb.unsqueeze(0).repeat(ht.shape[0], 1, 1)

        # 方案三，通过embedding+GCN编码problem
        # p_idx = torch.arange(self.num_p, device=pq_rel.device)
        # p_emb = self.p_emb(p_idx)
        # edge_index = self.adj_p.nonzero(as_tuple=False).t()  # 注意adj_p不是对陈的

        # p_emb = self.gcn1(p_emb, edge_index)
        # p_emb = F.relu(p_emb)
        # p_emb = self.gcn2(p_emb, edge_index)  # [16891,30]
        # p_emb = p_emb.unsqueeze(0).repeat(ht.shape[0], 1, 1) # [64,16891,30]

        # ----------------将concept转化为具体问题----------------
        # [64,110,30]
        ht = torch.matmul(pq_rel,ht)  # [17751,110] * [64,110,30] => [64,17751,30]

        out = torch.cat([ht, p_emb], dim=-1)
        return torch.sigmoid(self.out_layer(out) + self.bias_p).squeeze()

    def predict_p_with_hist(self, ht, h_his, p_emb, p):  # h 历史状态，待聚合；ht 当前状态


        # edge_index = self.adj.nonzero(as_tuple=False).t()  # 注意adj_p不是对陈的
        # q_emb = self.gcn1(q_emb, edge_index)
        # import torch.nn.functional as F
        # q_emb = F.relu(q_emb)
        # q_emb = self.gcn2(q_emb, edge_index)

        h_his = torch.stack(h_his, dim=1)  # [64,2,110,30] 所有历史状态，按时间堆叠

        rate = torch.exp(self.theta)


        # test = torch.arange(h_his.shape[1]).flip(0)
        # test = torch.exp(-test)
        # test = test.view(1,-1,1)

        decay = torch.exp(-rate*torch.arange(h_his.shape[1]).flip(0)).view(1,-1, 1)  # [1,2,1] 利用sum可在dim=1相加

        B, T, C, F = h_his.shape
        P = p_emb.shape[0]

        p_idx = p[:,:T] # [64,2], 只取前T维
        p = p_emb[p_idx] # [64,2,30] 得到所有问题的表征


        query = p.unsqueeze(2).repeat(1,1,P,1) # [64,2,124,30]
        key = p_emb.unsqueeze(0).unsqueeze(0).repeat(B,T,1,1) # [124,30]=>[64,2,124,30]   110个问题的embedding，扩展前两维
        sim = (query * key / torch.tensor(math.sqrt(F))).sum(dim=-1) # [64,2,17751]  某个历史时刻的题目，和所有可能的110个题目的相似性

        # if(T<100):
        #     Tkey = q_emb[qT_idx]  # [64,30]
        #     Tque = q[:,T-1,:].squeeze()  # [64,30]
        #     Tsim = (Tque * Tkey / torch.tensor(math.sqrt(F))).sum(dim=-1)  # [64]
        #     test = sim[:,T-1][torch.arange(qT_idx.size(0)), qT_idx]
        #     test = test.squeeze()
        #     assert(torch.allclose(Tsim, test))  # [64,110]





        alpha = torch.softmax(decay * sim, dim=1) # [64,2,17751] # [64,2,110]   注意这里应该对时间维度作归一化

        # if(T==10):
        #     print("current:", qT_idx[0])
        #     print("history:", q_idx[0])
        #     print(sim[0,:,qT_idx[0]])
        #     print(alpha[0,:,qT_idx[0]])

        alpha = alpha.unsqueeze(-1) # =>[64,2,17751,30]  decay去掉？

        # hq = ht.unsqueeze(1).repeat(1,T,1,1).view(B,T, C*F)  #  [64,2,110*30]  # query,沿着时间维度扩展
        # hk = h.view(B,T,C*F)  # key，历史记录，[64,2,110*30]
        # sim = (hq*hk).sum(dim=-1)  # [64,2]
        # sim = torch.softmax(sim, dim=-1)  # [64,2]
        # alpha = (decay*sim).view(B,T,1,1)

        # 将state转化为具体问题
        pq_rel = self.pq_rel.float()  # [17751,123]

        h_his = torch.matmul(pq_rel, h_his) # [17751,123] * [64,2,123,30] => [64,2,17751,30]
        ht = torch.matmul(pq_rel, ht)

        h_his = (h_his * alpha).sum(dim=1, keepdim=False)         #    h: [64,2,17751,30]  沿着dim=1 相加

        # 将过去与当前相加
        ht = ht + h_his



        # [64,110,30]
        # ht = torch.matmul(pq_rel,ht)  # [16891,110] * [64,110,30] => [64,16891,30]
        return torch.sigmoid(self.out_layer(ht) + self.bias_p).squeeze()

    def predict_p(self, ht):
        # 将state转化为具体问题
        pq_rel = self.pq_rel.float()  # [16891,123]
        # 是否取平均？
        # row_sum = torch.sum(pq_rel, dim=1, keepdim=True)
        # avg_pq_rel = pq_rel / (row_sum + 1e-6)
        ht = torch.matmul(pq_rel,ht)  # [16891,110] * [64,110,30] => [64,16891,30]
        return torch.sigmoid(self.out_layer(ht) + self.bias_p).squeeze()

    def predict_adj_trans(self, ht):
        pred_adj_trans = torch.matmul(ht, ht.transpose(-1, -2))
        pred_adj_trans = torch.sigmoid(pred_adj_trans)
        return pred_adj_trans


    def train_model(
        self, train_loader, valid_loader, test_loader, num_epochs, opt, ckpt_path, device
    ):
        '''
            Args:
                train_loader: the PyTorch DataLoader instance for training
                test_loader: the PyTorch DataLoader instance for test
                num_epochs: the number of epochs
                opt: the optimization to train this model
                ckpt_path: the path to save this model's parameters
        '''
        loss_means = []
        valid_aucs = []
        test_aucs = []

        max_auc = 0

        for i in range(1, num_epochs + 1):
            loss_mean = []

            for data in train_loader:
                p, r, pshft, rshft, m = data  # q:题号 r:答案号  m:mask  # -------TRAIN-------

                # m[:,0:5] = False # 屏蔽前两个--------------------

                self.train()


                y, _ = self(p.long(), r.long())  # 对于每一时刻的下一步的所有问题的回答。long：长整型  # a:[64,100,110,110]  # h:[64,101,110,30]
                # y = (y * one_hot(qshft.long(), self.num_q)).sum(-1)  # 回答concept # 只取ground truth中的问题，用来检验
                y = (y * one_hot(pshft.long(), self.num_p)).sum(-1)  # 回答具体问题

                y = torch.masked_select(y, m)  # 摒弃padding的部分
                t = torch.masked_select(rshft, m)  # 摒弃padding的部分



                # 全归1
                # ------------ 辅助预测 ---------------
                """
                g_a = self.adj.clone()
                g_a[g_a !=0] = 1
                g_a = self.adj.unsqueeze(0).unsqueeze(0).repeat(a.shape[0], a.shape[1], 1, 1)

                m_a = m.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, self.num_q, self.num_q)

                g_a = torch.masked_select(g_a, m_a)
                a = torch.masked_select(a, m_a)
                """


                opt.zero_grad()
                loss = binary_cross_entropy(y, t)
                loss = loss
                loss.backward()
                opt.step()

                loss_mean.append(loss.detach().cpu().numpy())

            loss_mean = np.mean(loss_mean)  # 一个epoch训练结束，统计train loss

            # 一个epoch训练结束，计算valid_set的指标并监督  -------VALID----------------
            prediction = torch.tensor([])
            ground_truth = torch.tensor([])
            with torch.no_grad():
                for data in valid_loader:  # valid
                    p, r, pshft, rshft, m = data

                    # m[:, 0:5] = False

                    self.eval()

                    y, _ = self(p.long(), r.long())
                    # y = (y * one_hot(qshft.long(), self.num_q)).sum(-1)  # 回答concept
                    y = (y * one_hot(pshft.long(), self.num_p)).sum(-1)  # 回答具体问题

                    y = torch.masked_select(y, m)
                    t = torch.masked_select(rshft, m)

                    prediction = torch.cat([prediction, y])
                    ground_truth = torch.cat([ground_truth, t])

                valid_auc = metrics.roc_auc_score(
                    # y_true=t.numpy(), y_score=y.numpy()
                    y_true=ground_truth.detach().cpu(), y_score=prediction.detach().cpu()
                )

                if valid_auc > max_auc:
                    torch.save(
                        self.state_dict(),
                        os.path.join(
                            ckpt_path, f"model.ckpt"
                        )
                    )
                    max_auc = valid_auc
                    best_epoch = i

                print(
                    "Epoch: {}, Loss Mean: {} valid-AUC: {}， best-valid-AUC: {} at epoch {}"
                    .format(i, loss_mean, valid_auc, max_auc, best_epoch)
                )

                logger.info("Epoch: {}, Loss Mean: {}, valid-AUC: {}， best-valid-AUC: {} at epoch {}"
                    .format(i, loss_mean, valid_auc, max_auc, best_epoch))

                # valid_aucs.append(valid_auc)  # 没用，想删掉
                # loss_means.append(loss_mean)  # 没用，想删掉

        # 全部训练结束后，在test上预测结果
        best_state_dict = torch.load(os.path.join(ckpt_path, "model.ckpt"))
        self.test_model(test_loader, best_state_dict)

        return loss_means, valid_aucs, test_aucs

    def test_model(self, test_loader, state_dict):
        with torch.no_grad():

            # 先根据路径加载最优模型
            self.load_state_dict(state_dict)
            # print(self.state_dict())
            test_prediction = torch.tensor([])
            test_ground_truth = torch.tensor([])
            test_w = torch.tensor([])
            num = 0
            for data in test_loader:
                p, r, pshft, rshft, m = data  # ---------TEST--------



                self.eval()

                y, _ = self(p.long(), r.long())
                # y = (y * one_hot(qshft.long(), self.num_q)).sum(-1)  # 回答concept
                y = (y * one_hot(pshft.long(), self.num_p)).sum(-1)  # 回答具体问题

                # m[:,0:5] = False

                w = torch.masked_select(pshft, m)  # ???
                # c = torch.masked_select(qshft, m)
                y = torch.masked_select(y, m)
                t = torch.masked_select(rshft, m)


                # per_loss = binary_cross_entropy(y, t)


                # if(per_loss.item() > 0.90):  # 排查原因
                #     formatted = [f"{e.item():8.0f}" for e in c]
                #     print("".join(formatted))  # true
                #
                #     formatted = [f"{e:8.0f}" for e in t]
                #     print("".join(formatted))  # true
                #
                #     formatted = [f"{e:8.0f}" for e in y]
                #     print("".join(formatted))  # pred
                #
                #     print("per_loss: {}".format(per_loss))

                test_w = torch.cat([test_w, w])
                test_prediction = torch.cat([test_prediction, y])
                test_ground_truth = torch.cat([test_ground_truth, t])

            test_auc = metrics.roc_auc_score(
                y_true=test_ground_truth.detach().cpu(),
                y_score=test_prediction.detach().cpu()
            )

            preds = torch.round(test_prediction)
            trues = torch.round(test_ground_truth)
            wrong_mask = (preds!=trues)
            quess = test_w[wrong_mask]

            unique_ids, counts = torch.unique(quess, return_counts=True)

            error_count = {qid.item(): count.item() for qid, count in zip(unique_ids, counts) if count > 7}

            print("error_count: {}".format(error_count))


            print("test-AUC: {}".format(test_auc))

            logger.info("test-AUC: {}".format(test_auc))


class MYPAM(MYGKT):
    def __init__(self, num_q, num_p, adj_trans, pq_rel, hidden_size, num_attn_heads, method):
        super().__init__(num_q, num_p, adj_trans, pq_rel, hidden_size, num_attn_heads, method)

        # self.A = Parameter(torch.Tensor(self.num_q, self.num_q))
        # kaiming_normal_(self.A)  # 探索有多大用？

        self.A = torch.from_numpy(adj_trans).to('cuda')  # 最好是能单写一个class，不想公用这个

        self.mlp_outgo = mlp(self.hidden_size * 4, self.hidden_size)
        self.mlp_income = mlp(self.hidden_size * 4, self.hidden_size)

    def f_neighbor(self, ht_, qt_onehot):
        """
        Args:
            ht_: [64,123,30]
            qt_onehot: 当前时刻对应的所有concept [64,123]
        """
        batch_size = qt_onehot.shape[0]

        Aij = self.adj  # [123,123]

        # 执行mask操作，从[64,123,60]中执行[64,123,1]，将非主角concept全置为0
        ht_masked = ht_ * qt_onehot  # [64,123,60]

        outgo_mask = torch.matmul(Aij, ht_masked).sum(-1)  # [64,123,60]
        outgo_mask[outgo_mask != 0] = 1
        outgo_mask = outgo_mask.unsqueeze(-1)

        outgo_part = torch.cat([ht_, torch.matmul(Aij, ht_masked)], dim=-1)  # [64,123,30*4]  聚合了和每个concept相关的“主角concept”
        # outgo_part = torch.matmul(Aij, self.mlp_outgo(outgo_part))
        outgo_part = outgo_mask * self.mlp_outgo(outgo_part)

        income_mask = torch.matmul(Aij.T, ht_masked).sum(-1)  # [64,123,60]
        income_mask[income_mask != 0] = 1
        income_mask = income_mask.unsqueeze(-1)

        income_part = torch.cat([ht_, torch.matmul(Aij.T, ht_masked)], dim=-1)  # [64,123,30*4]
        # income_part = torch.matmul(Aij.T, self.mlp_income(income_part))
        income_part = income_mask * self.mlp_income(income_part)

        return outgo_part + income_part

    def f_neighbor11(self, ht_, qt):
        batch_size = qt.shape[0]
        src = ht_
        tgt = torch.gather(
            ht_,
            dim=1,
            index=qt.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, ht_.shape[-1])
        )

        # 加约束
        """
        A_sym = (self.A + self.A.T) / 2  # 对陈
        A_nonneg = torch.relu(A_sym)  # 非负
        """

        # outgo
        Aij = torch.gather(
            self.A.unsqueeze(0).repeat(batch_size, 1, 1),
            dim=1,
            index=qt.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, self.A.shape[-1])
        ).squeeze()

        outgo_part = Aij.unsqueeze(-1) * \
            self.mlp_outgo(
                torch.cat([tgt.repeat(1, self.num_q, 1), src], dim=-1)
            )

        # income
        Aji = torch.gather(
            self.A.unsqueeze(0).repeat(batch_size, 1, 1),
            dim=2,
            index=qt.unsqueeze(-1).unsqueeze(-1).repeat(1, self.A.shape[-1], 1)
        ).squeeze()  # 通过index，取所有汇聚于当前问题的问题。Aji可以理解为存储着权重

        income_part = Aji.unsqueeze(-1) * \
            self.mlp_income(
                torch.cat([tgt.repeat(1, self.num_q, 1), src], dim=-1)
            )  # 将当前问题与每个问题都融合，准备好，但未必每个都用的上，取决于权重；Aji* 相当于将权重乘在对应的问题上

        return outgo_part + income_part


class MYMHA(MYGKT):
    def __init__(self, num_q, hidden_size, num_attn_heads, method):
        super().__init__(num_q, hidden_size, num_attn_heads, method)
        self.num_attn_heads = num_attn_heads

        #############################################################
        # These definitions are due to a bug of PyTorch.
        # Please check the following link if you want to check details:
        # https://github.com/pytorch/pytorch/issues/27623
        # https://github.com/pytorch/pytorch/pull/39402
        self.Q = Linear(
            self.hidden_size * 2,
            self.hidden_size,
            bias=False
        )
        self.K = Linear(
            self.hidden_size * 2,
            self.hidden_size,
            bias=False
        )
        self.V = Linear(
            self.hidden_size * 4,
            self.hidden_size,
            bias=False
        )
        #############################################################

        self.mha = MultiheadAttention(
            self.hidden_size,
            self.num_attn_heads,
        )

    def f_neighbor(self, ht_, qt):
        src = ht_
        tgt = torch.gather(
            ht_,
            dim=1,
            index=qt.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, ht_.shape[-1])
        )

        q = self.Q(tgt)
        k = self.K(src)
        v = self.V(
            torch.cat([tgt.repeat(1, self.num_q, 1), src], dim=-1)
        )

        _, weights = self.mha(
            q.permute(1, 0, 2), k.permute(1, 0, 2), v.permute(1, 0, 2)
        )
        # Average attention weights over heads
        # https://github.com/pytorch/pytorch/blob/5fdcc20d8d96a6b42387f57c2ce331516ad94228/torch/nn/functional.py#L5257
        weights = weights.permute(0, 2, 1)

        return weights * v
