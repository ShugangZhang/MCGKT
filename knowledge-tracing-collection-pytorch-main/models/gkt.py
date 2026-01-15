import os

import numpy as np
import torch
import logging

from torch.nn import Module, Embedding, Parameter, Sequential, Linear, ReLU, \
    Dropout, MultiheadAttention, GRU
from torch.nn.init import kaiming_normal_
from torch.nn.functional import one_hot, binary_cross_entropy
from sklearn import metrics

def mlp(in_size, out_size):
    return Sequential(
        Linear(in_size, out_size),
        ReLU(),
        Dropout(),
        Linear(out_size, out_size),
    )


class GKT(Module):
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
            yt = self.predict(ht)

            # 规避batch内唯一元素问题
            if(len(yt.shape) == 1):  # [16891] => [1,16891]
                yt = yt.unsqueeze(0)

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

    def train_model(
        self, train_loader, valid_loader, test_loader, num_epochs, opt, ckpt_path, device
    ):
        '''
            Args:
                train_loader: the PyTorch DataLoader instance for training
                valid_loader: the PyTorch DataLoader instance for validation
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

                # valid_aucs.append(valid_auc)  # 没用，想删掉
                # loss_means.append(loss_mean)  # 没用，想删掉

        # 全部训练结束后，在test上预测结果
        best_state_dict = torch.load(os.path.join(ckpt_path, "model.ckpt"))
        test_auc, test_acc = self.test_model(test_loader, best_state_dict)

        return test_auc, test_acc

    def test_model(self, test_loader, state_dict):
        with torch.no_grad():

            # 先根据路径加载最优模型
            self.load_state_dict(state_dict)
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

            test_acc = metrics.accuracy_score(
                y_true=test_ground_truth.detach().cpu(),
                y_pred=(test_prediction.detach().cpu() > 0.5).float()
            )

            print("test-AUC: {}".format(test_auc))
            print("test-ACC: {}".format(test_acc))

            return test_auc, test_acc


class PAM(GKT):
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
        ).squeeze()  # Aij是[64,123]的向量，代表64个学生当前时刻的概念节点和其他所有节点的连接关系。
                     # 64个学生每个人在该时刻的问题不一样，通过用qt索引（qt是[64,]存储的当前时刻所有学生的题号）、从self.A中gather，只取当前考题的关联邻居

        outgo_part = Aij.unsqueeze(-1) * \
            self.mlp_outgo(
                torch.cat([tgt.repeat(1, self.num_q, 1), src], dim=-1)
            ) # 将当前节点特征重复N次，与所有节点拼接，但最终只聚合与当前节点相关联的邻居。这相当于简化计算，不必再费心挑选关联行进行拼接，而是直接由Aij后手处理、一步到位
        # 再具体一些，Aij由[64,123]变成了[64,123,1]，并进一步通过广播机制扩展到[64,123,feat_dim]，这样对于123中的关联节点中的特征进行mask。
        # 但是对于MCGKT而言，由于每个问题牵涉到多个概念，对于不同的节点、要拼接的特征不同，不能简单通过重复方式进行，所以采取的策略如下：
        # 1. 先把当前问题牵涉到的节点（以下简称当前节点集）特征保留，其他特征全部抹除（一轮mask）
        # 2. 预计算，把当前节点集将要影响的邻居节点记录下来（通过判断该行特征是否为0），并生成mask
        # 3. 正式计算，将当前节点集聚合到邻居节点，并与邻居节点的原特征concat、MLP，形成消息，再聚合给邻居节点（该过程可形成个性化消息，而不再是GKT中的单一节点特征重复）
        # 4. 在上述过程中，由于要concat原特征的缘故，一些非邻居节点也会参与更新（concat了空特征）。对于这些无效消息，通过刚才生成的mask全部抹除

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


class MHA(GKT):
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
