import os

import numpy as np
import torch
import logging

from torch.nn import Module, Embedding, Parameter, Sequential, Linear, ReLU, \
    Dropout, MultiheadAttention, GRU
from torch.nn.init import kaiming_normal_
from torch.nn.functional import one_hot, binary_cross_entropy
from sklearn import metrics

logger = logging.getLogger('main.eval')

def mlp(in_size, out_size):
    return Sequential(
        Linear(in_size, out_size),
        ReLU(),
        Dropout(),
        Linear(out_size, out_size),
    )


class MYGKT2015(Module):
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
    def __init__(self, num_q, hidden_size, num_attn_heads, method):
        super().__init__()
        self.num_q = num_q
        self.hidden_size = hidden_size

        self.x_emb = Embedding(self.num_q * 2, self.hidden_size)
        self.q_emb = Parameter(torch.Tensor(self.num_q, self.hidden_size))

        kaiming_normal_(self.q_emb)

        self.init_h = Parameter(torch.Tensor(self.num_q, self.hidden_size))

        self.mlp_self = mlp(self.hidden_size * 2, self.hidden_size)

        self.gru = GRU(
            self.hidden_size * 2,
            self.hidden_size,
            batch_first=True
        )

        self.bias = Parameter(torch.Tensor(1, self.num_q, 1))
        self.out_layer = Linear(self.hidden_size, 1, bias=False)

        self.theta = Parameter(torch.tensor(0.0))

    def forward(self, q, r):
        '''
            Args:
                q: the question(KC) sequence with the size of [batch_size, n]
                r: the response sequence with the size of [batch_size, n]

            Returns:
                y: the knowledge level about the all questions(KCs)
                h: the hidden states of the all questions(KCs)
        '''
        batch_size = q.shape[0]

        x = q + self.num_q * r  # 每个问题，连同各自编码；乘以num_q是为了把对和错分开编码，且不会重叠

        x_emb = self.x_emb(x)  # 110个concept，每个有对错，共计会有220个编码
        q_emb = self.q_emb.unsqueeze(0).repeat(batch_size, 1, 1)

        q_onehot = one_hot(q, self.num_q)

        ht = self.init_h.unsqueeze(0).repeat(batch_size, 1, 1)
        h = [ht]
        y = []

        for xt_emb, qt, qt_onehot in zip(
            x_emb.permute(1, 0, 2), q.permute(1, 0), q_onehot.permute(1, 0, 2)
        ):  # 转置后，可对batch内所有样本sequence的同一位置集中处理
            # xt_emb:问题+答案 编码；qt_onehot:问题索引，用于定位问题；q_emb:问题编码；ht:状态编码
            ht_ = self.aggregate(xt_emb, qt_onehot, q_emb, ht)  # q_emb和ht是两个独立状态，和batch无关，也没有经历过转置等操作

            ht = self.update(ht, ht_, qt, qt_onehot)  # ht随着时间变化，不断更新迭代（在本循环内保有范围）
            # yt = self.predict(ht)
            yt = self.predict_p_with_hist(ht, h, q_emb, q)

            h.append(ht)
            y.append(yt)

        h = torch.stack(h, dim=1)  # 按照时间序列堆叠起来，注意共有101个时序状态，比y少一个
        y = torch.stack(y, dim=1)  # 按照时间序列堆叠起来

        return y, h

    def aggregate(self, xt_emb, qt_onehot, q_emb, ht):
        xt_emb = xt_emb.unsqueeze(1).repeat(1, self.num_q, 1)
        qt_onehot = qt_onehot.unsqueeze(-1)

        qt_one = qt_onehot.detach().cpu()

        ht_ = qt_onehot * torch.cat([ht, xt_emb], dim=-1) + \
            (1 - qt_onehot) * torch.cat([ht, q_emb], dim=-1)  # 自身节点，整合[问题+回答]表征；其他所有节点，整合q的表征

        return ht_

    def f_self(self, ht_):
        return self.mlp_self(ht_)

    def f_neighbor(self, ht_, qt):
        pass

    def update(self, ht, ht_, qt, qt_onehot):
        qt_onehot = qt_onehot.unsqueeze(-1)

        m = qt_onehot * self.f_self(ht_) + \
            (1 - qt_onehot) * self.f_neighbor(ht_, qt)

        ht, _ = self.gru(
            torch.cat([m, ht], dim=-1)
        )

        return ht

    def predict(self, ht):
        return torch.sigmoid(self.out_layer(ht) + self.bias).squeeze()

    def predict_p_with_hist(self, ht, h_his, q_emb, q_idx):  # h 历史状态，待聚合；ht 当前状态

        q_emb = q_emb[0]  # 去掉batch，防止混淆

        # edge_index = self.adj.nonzero(as_tuple=False).t()  # 注意adj_p不是对陈的
        # q_emb = self.gcn1(q_emb, edge_index)
        # import torch.nn.functional as F
        # q_emb = F.relu(q_emb)
        # q_emb = self.gcn2(q_emb, edge_index)

        h_his = torch.stack(h_his, dim=1)  # [64,2,110,30] 所有历史状态，按时间堆叠

        rate = torch.exp(self.theta)


        decay = torch.exp(-rate*torch.arange(h_his.shape[1]).flip(0)).view(1,-1, 1)  # [1,2,1] 利用sum可在dim=1相加

        B, T, C, F = h_his.shape
        q_idx = q_idx[:,:T] # [64,2], 只取前T维
        q = q_emb[q_idx] # [64,2,30] 得到所有问题的表征

        query = q.unsqueeze(2).repeat(1,1,C,1) # [64,2,110,30]
        key = q_emb.unsqueeze(0).unsqueeze(0).repeat(B,T,1,1) # [110,30]=>[64,2,110,30]   110个问题的embedding，扩展前两维
        sim = (query * key).sum(dim=-1) # [64,2,110]  某个历史时刻的题目，和所有可能的110个题目的相似性
        alpha = torch.softmax(decay * sim, dim=1) # [64,2,110] # [64,2,110]   注意这里应该对时间维度作归一化
        alpha = alpha.unsqueeze(-1).repeat(1,1,1,F) # =>[64,2,110,30]  decay去掉？

        # hq = ht.unsqueeze(1).repeat(1,T,1,1).view(B,T, C*F)  #  [64,2,110*30]  # query,沿着时间维度扩展
        # hk = h.view(B,T,C*F)  # key，历史记录，[64,2,110*30]
        # sim = (hq*hk).sum(dim=-1)  # [64,2]
        # sim = torch.softmax(sim, dim=-1)  # [64,2]
        # alpha = (decay*sim).view(B,T,1,1)

        h_his = (h_his * alpha).sum(dim=1, keepdim=False)         #    h: [64,2,110,30]  沿着dim=1 相加

        # 将过去与当前相加
        ht = ht + h_his

        # 将state转化为具体问题
        return torch.sigmoid(self.out_layer(ht) + self.bias).squeeze()

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
                q, r, qshft, rshft, m = data  # q:题号 r:答案号  m:mask

                self.train()

                y, _ = self(q.long(), r.long())  # 对于每一时刻的下一步的所有问题的回答。long：长整型
                y = (y * one_hot(qshft.long(), self.num_q)).sum(-1)  # 只取ground truth中的问题，用来检验

                y = torch.masked_select(y, m)  # 摒弃padding的部分
                t = torch.masked_select(rshft, m)  # 摒弃padding的部分

                opt.zero_grad()
                loss = binary_cross_entropy(y, t)
                loss.backward()
                opt.step()

                loss_mean.append(loss.detach().cpu().numpy())

            loss_mean = np.mean(loss_mean)  # 一个epoch训练结束，统计train loss

            # 一个epoch训练结束，计算valid_set的指标并监督
            prediction = torch.tensor([])
            ground_truth = torch.tensor([])
            with torch.no_grad():
                for data in valid_loader:  # valid
                    q, r, qshft, rshft, m = data

                    self.eval()

                    y, _ = self(q.long(), r.long())
                    y = (y * one_hot(qshft.long(), self.num_q)).sum(-1)

                    # y = torch.masked_select(y, m).detach().cpu()
                    # t = torch.masked_select(rshft, m).detach().cpu()
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
                            ckpt_path, "model.ckpt"
                        )
                    )
                    max_auc = valid_auc
                    best_epoch = i

                print(
                    "Epoch: {}, Loss Mean: {}, valid-AUC: {}， best-valid-AUC: {} at epoch {}"
                    .format(i, loss_mean, valid_auc, max_auc, best_epoch)
                )

                logger.info("Epoch: {}, Loss Mean: {}, valid-AUC: {}， best-valid-AUC: {} at epoch {}"
                    .format(i, loss_mean, valid_auc, max_auc, best_epoch))

                # valid_aucs.append(valid_auc)  # 没用，想删掉
                # loss_means.append(loss_mean)  # 没用，想删掉

        # 全部训练结束后，在test上预测结果
        self.test_model(test_loader)

        return loss_means, valid_aucs, test_aucs

    def test_model(self, test_loader):
        with torch.no_grad():

            # 先根据路径加载最优模型
            self.load_state_dict(self.state_dict())
            test_prediction = torch.tensor([])
            test_ground_truth = torch.tensor([])
            for data in test_loader:
                q, r, qshft, rshft, m = data

                self.eval()

                y, _ = self(q.long(), r.long())

                y = (y * one_hot(qshft.long(), self.num_q)).sum(-1)

                # y = torch.masked_select(y, m).detach().cpu()
                # t = torch.masked_select(rshft, m).detach().cpu()
                y = torch.masked_select(y, m)
                t = torch.masked_select(rshft, m)

                test_prediction = torch.cat([test_prediction, y])
                test_ground_truth = torch.cat([test_ground_truth, t])

            test_auc = metrics.roc_auc_score(
                y_true=test_ground_truth.detach().cpu(),
                y_score=test_prediction.detach().cpu()
            )

            print("test-AUC: {}".format(test_auc))

            logger.info("test-AUC: {}".format(test_auc))


class MYPAM2015(MYGKT2015):
    def __init__(self, num_q, hidden_size, num_attn_heads, method):
        super().__init__(num_q, hidden_size, num_attn_heads, method)

        self.A = Parameter(torch.Tensor(self.num_q, self.num_q))
        kaiming_normal_(self.A)

        self.mlp_outgo = mlp(self.hidden_size * 4, self.hidden_size)
        self.mlp_income = mlp(self.hidden_size * 4, self.hidden_size)

    def f_neighbor(self, ht_, qt):
        batch_size = qt.shape[0]
        src = ht_
        tgt = torch.gather(
            ht_,
            dim=1,
            index=qt.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, ht_.shape[-1])
        )

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
        ).squeeze()

        income_part = Aji.unsqueeze(-1) * \
            self.mlp_income(
                torch.cat([tgt.repeat(1, self.num_q, 1), src], dim=-1)
            )

        return outgo_part + income_part


class MYMHA2015(MYGKT2015):
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
