
"""
原始备份：构造学生的hidden state图, q graph
"""

import os

import pickle

import numpy as np
import pandas as pd

from torch.utils.data import Dataset

from models.utils import match_seq_len


DATASET_DIR = "datasets/ASSIST2009/"


class ASSIST2009(Dataset):
    def __init__(self, seq_len, dataset_dir=DATASET_DIR) -> None:
        super().__init__()

        self.dataset_dir = dataset_dir
        self.dataset_path = os.path.join(
            self.dataset_dir, "skill_builder_data.csv"
        )

        if os.path.exists(os.path.join(self.dataset_dir, "q_seqs.pkl")):
            with open(os.path.join(self.dataset_dir, "q_seqs.pkl"), "rb") as f:
                self.q_seqs = pickle.load(f)
            with open(os.path.join(self.dataset_dir, "r_seqs.pkl"), "rb") as f:
                self.r_seqs = pickle.load(f)
            with open(os.path.join(self.dataset_dir, "q_list.pkl"), "rb") as f:
                self.q_list = pickle.load(f)
            with open(os.path.join(self.dataset_dir, "u_list.pkl"), "rb") as f:
                self.u_list = pickle.load(f)
            with open(os.path.join(self.dataset_dir, "q2idx.pkl"), "rb") as f:
                self.q2idx = pickle.load(f)
            with open(os.path.join(self.dataset_dir, "u2idx.pkl"), "rb") as f:
                self.u2idx = pickle.load(f)
            with open(os.path.join(self.dataset_dir, "adj_trans.pkl"), "rb") as f:  # 构筑邻接矩阵
                self.adj_trans = pickle.load(f)
        else:
            self.q_seqs, self.r_seqs, self.q_list, self.u_list, self.q2idx, \
                self.u2idx, self.adj_trans = self.preprocess()

        self.num_u = self.u_list.shape[0]
        self.num_q = self.q_list.shape[0]

        if seq_len:
            self.q_seqs, self.r_seqs = \
                match_seq_len(self.q_seqs, self.r_seqs, seq_len)

        self.len = len(self.q_seqs)

    def __getitem__(self, index):
        return self.q_seqs[index], self.r_seqs[index]

    def __len__(self):
        return self.len

    def preprocess(self):
        df = pd.read_csv(self.dataset_path).dropna(subset=["skill_name"])\
            .drop_duplicates(subset=["order_id", "skill_name"])\
            .sort_values(by=["order_id"])

        u_list = np.unique(df["user_id"].values)
        q_list = np.unique(df["skill_name"].values)  # 以concept作为节点

        u2idx = {u: idx for idx, u in enumerate(u_list)}
        q2idx = {q: idx for idx, q in enumerate(q_list)}

        q_seqs = []
        r_seqs = []

        # 构筑邻接关系
        num_q = q_list.shape[0]
        adj_trans = np.zeros((num_q, num_q), dtype=np.float32)

        for u in u_list:  # 遍历所有用户，按照用户来组织条目
            df_u = df[df["user_id"] == u]

            q_seq = np.array([q2idx[q] for q in df_u["skill_name"]])   # q_seq代表数量
            r_seq = df_u["correct"].values

            q_seqs.append(q_seq)  # 这里尚未pad，是到外面的collate_fn完成padding的
            r_seqs.append(r_seq)

            # 从这里就要构筑邻接关系，借助这里的用户遍历逻辑，进一步遍历问题，并梳理问题的先后关系
            # 注意在当前场景下，这种邻接关系是双向的
            pairs = list(zip(q_seq, q_seq[1:]))  # 注意zip按照较短数组来中止
            for (i,j) in pairs:
                adj_trans[i][j] += 1

        # 取平均
        row_sum = np.sum(adj_trans, axis=1, keepdims=True)
        adj_trans = adj_trans/row_sum

        # 取非零
        # adj_trans[adj_trans != 0] = 1


        with open(os.path.join(self.dataset_dir, "q_seqs.pkl"), "wb") as f:
            pickle.dump(q_seqs, f)
        with open(os.path.join(self.dataset_dir, "r_seqs.pkl"), "wb") as f:
            pickle.dump(r_seqs, f)
        with open(os.path.join(self.dataset_dir, "q_list.pkl"), "wb") as f:
            pickle.dump(q_list, f)
        with open(os.path.join(self.dataset_dir, "u_list.pkl"), "wb") as f:
            pickle.dump(u_list, f)
        with open(os.path.join(self.dataset_dir, "q2idx.pkl"), "wb") as f:
            pickle.dump(q2idx, f)
        with open(os.path.join(self.dataset_dir, "u2idx.pkl"), "wb") as f:
            pickle.dump(u2idx, f)
        with open(os.path.join(self.dataset_dir, "adj_trans.pkl"), "wb") as f:
            pickle.dump(adj_trans, f)

        return q_seqs, r_seqs, q_list, u_list, q2idx, u2idx, adj_trans
