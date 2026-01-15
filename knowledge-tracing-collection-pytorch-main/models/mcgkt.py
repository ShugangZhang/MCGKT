import math
import os
import logging
import numpy as np
import torch
import torch.nn.functional as F

from torch.nn import Module, Embedding, Parameter, Sequential, Linear, ReLU, Dropout, GRU
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


class MCGKT(Module):
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
    def __init__(self, num_q, num_p, adj_trans, pq_rel, hidden_size):
        super().__init__()
        self.num_q = num_q
        self.hidden_size = hidden_size

        self.x_emb_layer = Embedding(self.num_q * 2, self.hidden_size)  # 每个concept，连同对错，都要编码，所以是两倍

        self.q_emb_layer = Embedding(self.num_q, self.hidden_size)

        self.init_h = Parameter(torch.Tensor(self.num_q, self.hidden_size))

        self.mlp_self = mlp(self.hidden_size * 2, self.hidden_size)

        self.gru = GRU(
            self.hidden_size * 2,
            self.hidden_size,
            batch_first=True
        )  # 感觉这个GRU用的很奇怪

        self.bias = Parameter(torch.Tensor(1, self.num_q, 1))
        self.out_layer = Linear(self.hidden_size, 1, bias=False)

        # concept之间的连接关系，注意这里是一题多概念，该矩阵刻画了所有关联概念的连接关系，且是单向的
        self.adj = torch.from_numpy(adj_trans).to('cuda')
        self.num_p = num_p
        self.bias_p = Parameter(torch.Tensor(1, self.num_p, 1))

        self.pq_rel = torch.from_numpy(pq_rel).to('cuda')

        self.mlp_outgo = mlp(self.hidden_size * 4, self.hidden_size)
        self.mlp_income = mlp(self.hidden_size * 4, self.hidden_size)

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


        xq_no = torch.arange(self.num_q * 2, device=p.device).long()  # 0~245  所有概念连同对错的编码
        xq_emb = self.x_emb_layer(xq_no)  # 所有246个inter的embedding  # [246,30]

        q_no = torch.arange(self.num_q, device=p.device).long()  # 0~122
        q_emb = self.q_emb_layer(q_no)  # [123, 30]  所有123个concept的embedding

        q_emb = q_emb.unsqueeze(0).repeat(batch_size, 1, 1)  # [64,123,30]

        ht = self.init_h.unsqueeze(0).repeat(batch_size, 1, 1)
        h = [ht]
        y = []

        for pt, rt, qt_onehot in zip(
            p.permute(1, 0), r.permute(1, 0), q_onehot.permute(1, 0, 2)
        ):  # 转置后，可对batch内所有样本sequence的同一位置集中处理
            # xt_emb:concept+答案 编码；qt_onehot:concept索引于定位concept；q_emb:concept编码；ht:状态编码

            ht_ = self.aggregate(rt, qt_onehot, q_emb, xq_emb, ht)  # q_emb和ht是两个独立状态，和batch无关，也没有经历过转置等操作
            ht = self.update(ht, ht_, qt_onehot)  # ht随着时间变化，不断更新迭代（在本循环内保有范围）
            yt = self.predict_p(ht)  # 回答具体问题

            # 规避batch内唯一元素问题
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

    def f_self(self, ht_):
        return self.mlp_self(ht_)

    def f_neighbor(self, ht_, qt_onehot):
        """
        Args:
            ht_: [64,123,30]
            qt_onehot: 当前时刻对应的所有concept [64,123]
        """

        Aij = self.adj  # [123,123]

        test = torch.sum(Aij,1)

        # 执行mask操作，从[64,123,60]中执行[64,123,1]，将非主角concept的特征全置为0。从计算规则上，onehot的最后一维可以复制扩展到ht的特征维上（即60）
        ht_masked = ht_ * qt_onehot  # [64,123,60]

        outgo_mask = torch.matmul(Aij, ht_masked).sum(-1)  # [64,123,60] sum=> [64,123]  注意sum后最后一个维度直接没有了
        outgo_mask[outgo_mask != 0] = 1  # [64,123]
        outgo_mask = outgo_mask.unsqueeze(-1)  # => [64,123,1] 将最后一个维度恢复出来

        outgo_part = torch.cat([ht_, torch.matmul(Aij, ht_masked)], dim=-1)  # [64,123,30*4]  matmul聚合了和每个concept相关的“主角concept”，然后与自身concat
        # outgo_part = torch.matmul(Aij, self.mlp_outgo(outgo_part))
        outgo_part = outgo_mask * self.mlp_outgo(outgo_part)  # 再次mask：原因是上一步concat了ht_并作了MLP,而对于那些与当前概念无关联的ht_,是不应该参与到消息传递中的,需要抹掉

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

    def predict(self, ht):
        return torch.sigmoid(self.out_layer(ht) + self.bias).squeeze()

    def predict_p(self, ht):
        # 将state转化为具体问题
        pq_rel = self.pq_rel.float()  # [16891,123]
        # 是否取平均？
        # row_sum = torch.sum(pq_rel, dim=1, keepdim=True)
        # avg_pq_rel = pq_rel / (row_sum + 1e-6)
        ht = torch.matmul(pq_rel,ht)  # [16891,110] * [64,110,30] => [64,16891,30]
        return torch.sigmoid(self.out_layer(ht) + self.bias_p).squeeze()

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
                p, r, pshft, rshft, m = data  # q:题号 r:答案号  m:mask  # -------TRAIN-------

                self.train()

                y, _ = self(p.long(), r.long())  # 对于每一时刻的下一步的所有问题的回答。long：长整型  # a:[64,100,110,110]  # h:[64,101,110,30]
                y = (y * one_hot(pshft.long(), self.num_p)).sum(-1)  # 回答具体问题

                y = torch.masked_select(y, m)  # 摒弃padding的部分
                t = torch.masked_select(rshft, m)  # 摒弃padding的部分


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

                    self.eval()

                    y, _ = self(p.long(), r.long())
                    y = (y * one_hot(pshft.long(), self.num_p)).sum(-1)  # 回答具体问题

                    y = torch.masked_select(y, m)
                    t = torch.masked_select(rshft, m)

                    prediction = torch.cat([prediction, y])
                    ground_truth = torch.cat([ground_truth, t])

                valid_auc = metrics.roc_auc_score(
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
                    "Epoch: {}, Loss Mean: {} valid-AUC: {}， best-valid-AUC: {} at epoch {}"
                    .format(i, loss_mean, valid_auc, max_auc, best_epoch)
                )

                logger.info("Epoch: {}, Loss Mean: {}, valid-AUC: {}， best-valid-AUC: {} at epoch {}"
                    .format(i, loss_mean, valid_auc, max_auc, best_epoch))


        # 全部训练结束后，在test上预测结果
        best_state_dict = torch.load(os.path.join(ckpt_path, "model.ckpt"))
        test_auc, test_acc = self.test_model(test_loader, best_state_dict)

        return test_auc, test_acc

    def test_model(self, test_loader, state_dict):
        with torch.no_grad():
            # 根据路径加载待测模型
            self.load_state_dict(state_dict)
            test_prediction = torch.tensor([])
            test_ground_truth = torch.tensor([])

            for data in test_loader:
                p, r, pshft, rshft, m = data  # ---------TEST--------

                self.eval()

                y, _ = self(p.long(), r.long())
                y = (y * one_hot(pshft.long(), self.num_p)).sum(-1)  # 回答具体问题

                # 掩码去掉padding问题
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

            logger.info("test-AUC: {}".format(test_auc))
            logger.info("test-AUC: {}".format(test_acc))

            return test_auc, test_acc
