import os

import numpy as np
import torch

from torch.nn import Module, Parameter, Embedding, Linear, Transformer
from torch.nn.init import normal_
from torch.nn.functional import binary_cross_entropy
from sklearn import metrics


class SAINT(Module):
    def __init__(
        self, num_q, n, d, num_attn_heads, dropout, num_tr_layers=1
    ):
        super().__init__()
        self.num_q = num_q
        self.n = n
        self.d = d
        self.num_attn_heads = num_attn_heads
        self.dropout = dropout
        self.num_tr_layers = num_tr_layers

        self.E = Embedding(self.num_q, self.d)
        self.R = Embedding(2, self.d)
        self.P = Parameter(torch.Tensor(self.n, self.d))
        self.S = Parameter(torch.Tensor(1, self.d))

        normal_(self.P)
        normal_(self.S)

        self.transformer = Transformer(
            self.d,
            self.num_attn_heads,
            num_encoder_layers=self.num_tr_layers,
            num_decoder_layers=self.num_tr_layers,
            dropout=self.dropout,
        )

        self.pred = Linear(self.d, 1)

    def forward(self, q, r):
        batch_size = r.shape[0]

        E = self.E(q).permute(1, 0, 2)  # dim_q:[64,100]    dim_E:[123,30]   => [64,100,30]
        R = self.R(r[:, :-1]).permute(1, 0, 2)   # dim_R:[99,64,30]
        S = self.S.repeat(batch_size, 1).unsqueeze(0)  # dim_S:[1,64,30]

        R = torch.cat([S, R], dim=0)

        P = self.P.unsqueeze(1)

        mask = self.transformer.generate_square_subsequent_mask(self.n).to(device=E.device)
        R = self.transformer(
            E + P, R + P, mask, mask, mask
        )
        R = R.permute(1, 0, 2)

        p = torch.sigmoid(self.pred(R)).squeeze()

        return p

    def discover_concepts(self, q, r):
        queries = torch.LongTensor([list(range(self.num_q))] * self.n)\
            .permute(1, 0)

        x = q + self.num_q * r
        x = x.repeat(self.num_q, 1)

        M = self.M(x).permute(1, 0, 2)
        E = self.E(queries).permute(1, 0, 2)
        P = self.P.unsqueeze(1)

        causal_mask = torch.triu(
            torch.ones([E.shape[0], M.shape[0]]), diagonal=1
        ).bool()

        M += P

        S, attn_weights = self.attn(E, M, M, attn_mask=causal_mask)
        S = self.attn_dropout(S)
        S = S.permute(1, 0, 2)
        M = M.permute(1, 0, 2)
        E = E.permute(1, 0, 2)

        S = self.attn_layer_norm(S + M + E)

        F = self.FFN(S)
        F = self.FFN_layer_norm(F + S)

        p = torch.sigmoid(self.pred(F)).squeeze()

        return p, attn_weights

    def train_model(self, train_loader, valid_loader, test_loader, num_epochs, opt, ckpt_path, device):
        loss_means = []
        valid_aucs = []
        test_aucs = []

        max_auc = 0

        for i in range(1, num_epochs + 1):
            loss_mean = []

            for data in train_loader:
                q, r, _, _, m = data

                self.train()

                p = self(q.long(), r.long())
                p = torch.masked_select(p, m)
                t = torch.masked_select(r, m).float()

                opt.zero_grad()
                loss = binary_cross_entropy(p, t)
                loss.backward()
                opt.step()

                loss_mean.append(loss.detach().cpu().numpy())

            loss_mean = np.mean(loss_mean)  # 一个epoch训练结束，统计train loss

            # 一个epoch训练结束，计算valid_set的指标并监督
            prediction = torch.tensor([])
            ground_truth = torch.tensor([])


            with torch.no_grad():
                for data in valid_loader:
                    q, r, _, _, m = data

                    self.eval()

                    p = self(q.long(), r.long())
                    p = torch.masked_select(p, m)
                    t = torch.masked_select(r, m).float()

                    prediction = torch.cat([prediction, p])
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
                    "Epoch: {}, Loss Mean: {}, valid-AUC: {}， best-valid-AUC: {} at epoch {}"
                    .format(i, loss_mean, valid_auc, max_auc, best_epoch)
                )

                # 全部训练结束后，在test上预测结果
        best_state_dict = torch.load(os.path.join(ckpt_path, "model.ckpt"))
        test_auc, test_acc = self.test_model(test_loader, best_state_dict)

        # return loss_means, valid_aucs, test_aucs
        return test_auc, test_acc

    def test_model(self, test_loader, state_dict):
        with torch.no_grad():
            # 先根据路径加载最优模型
            self.load_state_dict(state_dict)
            test_prediction = torch.tensor([])
            test_ground_truth = torch.tensor([])
            for data in test_loader:
                q, r, _, _, m = data

                self.train()

                p = self(q.long(), r.long())
                p = torch.masked_select(p, m)
                t = torch.masked_select(r, m).float()


                test_prediction = torch.cat([test_prediction, p])
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