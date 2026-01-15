import os

import numpy as np
import torch

from torch.nn import Module, Embedding, LSTM, Linear, Dropout
from torch.nn.functional import one_hot, binary_cross_entropy
from sklearn import metrics


class DKTPlus(Module):
    '''
        Args:
            num_q: the total number of the questions(KCs) in the given dataset
            emb_size: the dimension of the embedding vectors in this model
            hidden_size: the dimension of the hidden vectors in this model
            lambda_r: the hyperparameter of this model
            lambda_w1: the hyperparameter of this model
            lambda_w2: the hyperparameter of this model
    '''
    def __init__(
        self, num_q, emb_size, hidden_size, lambda_r, lambda_w1, lambda_w2
    ):
        super().__init__()
        self.num_q = num_q
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.lambda_r = lambda_r
        self.lambda_w1 = lambda_w1
        self.lambda_w2 = lambda_w2

        self.interaction_emb = Embedding(self.num_q * 2, self.emb_size)
        self.lstm_layer = LSTM(
            self.emb_size, self.hidden_size, batch_first=True
        )
        self.out_layer = Linear(self.hidden_size, self.num_q)
        self.dropout_layer = Dropout()

    def forward(self, q, r):
        '''
            Args:
                q: the question(KC) sequence with the size of [batch_size, n]
                r: the response sequence with the size of [batch_size, n]

            Returns:
                y: the knowledge level about the all questions(KCs)
        '''
        x = q + self.num_q * r

        h, _ = self.lstm_layer(self.interaction_emb(x))
        y = self.out_layer(h)
        y = self.dropout_layer(y)
        y = torch.sigmoid(y)

        return y

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
                q, r, qshft, rshft, m = data

                self.train()

                y = self(q.long(), r.long())
                y_curr = (y * one_hot(q.long(), self.num_q)).sum(-1)
                y_next = (y * one_hot(qshft.long(), self.num_q)).sum(-1)

                y_curr = torch.masked_select(y_curr, m)
                y_next = torch.masked_select(y_next, m)
                r = torch.masked_select(r, m)
                rshft = torch.masked_select(rshft, m)

                loss_w1 = torch.masked_select(
                    torch.norm(y[:, 1:] - y[:, :-1], p=1, dim=-1),
                    m[:, 1:]
                )
                loss_w2 = torch.masked_select(
                    (torch.norm(y[:, 1:] - y[:, :-1], p=2, dim=-1) ** 2),
                    m[:, 1:]
                )

                opt.zero_grad()
                loss = \
                    binary_cross_entropy(y_next, rshft) + \
                    self.lambda_r * binary_cross_entropy(y_curr, r) + \
                    self.lambda_w1 * loss_w1.mean() / self.num_q + \
                    self.lambda_w2 * loss_w2.mean() / self.num_q
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

                    y = self(q.long(), r.long())
                    y = (y * one_hot(qshft.long(), self.num_q)).sum(-1)

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

                y = self(q.long(), r.long())

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