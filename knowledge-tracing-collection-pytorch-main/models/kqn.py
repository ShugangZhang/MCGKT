import os

import numpy as np
import torch

from torch.nn import Module, Embedding, LSTM, Linear, Dropout, Sequential, ReLU
from torch.nn.functional import binary_cross_entropy
from sklearn import metrics


class KQN(Module):
    def __init__(self, num_q, dim_v, dim_s, hidden_size):
        super().__init__()
        self.num_q = num_q
        self.dim_v = dim_v
        self.dim_s = dim_s
        self.hidden_size = hidden_size

        self.x_emb = Embedding(self.num_q * 2, self.dim_v)
        self.knowledge_encoder = LSTM(self.dim_v, self.dim_v, batch_first=True)
        self.out_layer = Linear(self.dim_v, self.dim_s)
        self.dropout_layer = Dropout()

        self.q_emb = Embedding(self.num_q, self.dim_v)
        self.skill_encoder = Sequential(
            Linear(self.dim_v, self.hidden_size),
            ReLU(),
            Linear(self.hidden_size, self.dim_v),
            ReLU()
        )

    def forward(self, q, r, qry):
        # Knowledge State Encoding
        x = q + self.num_q * r
        x = self.x_emb(x)
        h, _ = self.knowledge_encoder(x)
        ks = self.out_layer(h)
        ks = self.dropout_layer(ks)

        # Skill Encoding
        e = self.q_emb(qry)
        o = self.skill_encoder(e)
        s = o / torch.norm(o, p=2)

        p = torch.sigmoid((ks * s).sum(-1))

        return p

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
                q, r, qshft, rshft, m = data

                self.train()

                p = self(q.long(), r.long(), qshft.long())
                p = torch.masked_select(p, m)
                if(p.numel() == 0):
                    print(p)
                t = torch.masked_select(rshft, m)

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
                    q, r, qshft, rshft, m = data

                    self.eval()

                    p = self(q.long(), r.long(), qshft.long())
                    p = torch.masked_select(p, m)
                    t = torch.masked_select(rshft, m)

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

                p = self(q.long(), r.long(), qshft.long())
                p = torch.masked_select(p, m)
                t = torch.masked_select(rshft, m)

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