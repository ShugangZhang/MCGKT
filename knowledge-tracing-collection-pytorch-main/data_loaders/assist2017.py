
"""
尝试面向问题构图
"""

import os

import pickle

import numpy as np
import pandas as pd
import itertools

from torch.utils.data import Dataset

from models.utils import match_seq_len, match_seq_len_2009, match_seq_len_2009_corrected  # 尚不知是否适配17数据集

DATASET_DIR = "datasets/ASSIST2017/"


class ASSIST2017(Dataset):
    def __init__(self, seq_len, dataset_dir=DATASET_DIR) -> None:
        super().__init__()

        self.dataset_dir = dataset_dir
        self.dataset_path = os.path.join(
            # self.dataset_dir, "skill_builder_data.csv"
            self.dataset_dir, "anonymized_full_release_competition_dataset_with_skill_id.csv"
        )

        if os.path.exists(os.path.join(self.dataset_dir, "p_seqs.pkl")):  # q_seq被取消了，因为现在是一题多概念
            with open(os.path.join(self.dataset_dir, "p_seqs.pkl"), "rb") as f:
                self.p_seqs = pickle.load(f)
            with open(os.path.join(self.dataset_dir, "r_seqs.pkl"), "rb") as f:
                self.r_seqs = pickle.load(f)
            with open(os.path.join(self.dataset_dir, "p_list.pkl"), "rb") as f:
                self.p_list = pickle.load(f)
            with open(os.path.join(self.dataset_dir, "q_list.pkl"), "rb") as f:
                self.q_list = pickle.load(f)
            with open(os.path.join(self.dataset_dir, "u_list.pkl"), "rb") as f:
                self.u_list = pickle.load(f)
            with open(os.path.join(self.dataset_dir, "p2idx.pkl"), "rb") as f:
                self.p2idx = pickle.load(f)
            with open(os.path.join(self.dataset_dir, "q2idx.pkl"), "rb") as f:
                self.q2idx = pickle.load(f)
            with open(os.path.join(self.dataset_dir, "u2idx.pkl"), "rb") as f:
                self.u2idx = pickle.load(f)
            with open(os.path.join(self.dataset_dir, "adj_trans.pkl"), "rb") as f:  # 构筑概念邻接矩阵
                self.adj_trans = pickle.load(f)
            with open(os.path.join(self.dataset_dir, "pq_rel.pkl"), "rb") as f:  # 构筑问题-概念关系矩阵
                self.pq_rel = pickle.load(f)

        else:
            self.p_seqs, self.r_seqs, self.p_list, self.q_list, self.u_list, self.p2idx, self.q2idx, \
                self.u2idx, self.adj_trans, self.pq_rel = self.preprocess()  # 问题图尚未构建
            # self.p_seqs, self.p_list, self.p2idx, self.adj_p = self.generate_prob_graph()
            # self.pq_rel = self.generate_pq_matrix()

        self.num_u = self.u_list.shape[0]
        self.num_q = self.q_list.shape[0]
        self.num_p = self.p_list.shape[0]

        if seq_len:
            self.p_seqs, self.r_seqs = \
                match_seq_len_2009_corrected(self.p_seqs, self.r_seqs, seq_len)

        self.len = len(self.p_seqs)

    def __getitem__(self, index):
        return self.p_seqs[index], self.r_seqs[index]

    def __len__(self):
        return self.len

    def generate_prob_graph(self):
        df = pd.read_csv(self.dataset_path).dropna(subset=["skill"])

        u_list = np.unique(df["studentId"].values)
        p_list = np.unique(df["problemId"].values)  # problem list

        p2idx = {p: idx for idx, p in enumerate(p_list)}

        p_seqs = []

        # 构筑邻接关系
        num_p = p_list.shape[0]
        adj_p = np.zeros((num_p, num_p), dtype=np.float32)

        for u in u_list:  # 遍历所有用户，按照用户来组织条目
            df_u = df[df["studentId"] == u]

            p_seq = np.array([p2idx[p] for p in df_u["problemId"]])

            p_seqs.append(p_seq)  # 这里尚未pad，是到外面的collate_fn完成padding的

            # 从这里就要构筑邻接关系，借助这里的用户遍历逻辑，进一步遍历问题，并梳理问题的先后关系
            # 注意在当前场景下，这种邻接关系是双向的
            pairs = list(zip(p_seq, p_seq[1:]))  # 注意zip按照较短数组来中止
            for (i, j) in pairs:
                adj_p[i][j] += 1

        # 取平均
        row_sum = np.sum(adj_p, axis=1, keepdims=True)
        row_sum = row_sum.clip(min=1)  # 防止除零错误
        adj_p = adj_p / row_sum

        # 取非零
        # adj_p[adj_p != 0] = 1

        with open(os.path.join(self.dataset_dir, "p_seqs.pkl"), "wb") as f:  # 需要返回plist吗？
            pickle.dump(p_seqs, f)
        with open(os.path.join(self.dataset_dir, "p_list.pkl"), "wb") as f:  # 需要返回plist吗？
            pickle.dump(p_list, f)
        with open(os.path.join(self.dataset_dir, "p2idx.pkl"), "wb") as f:
            pickle.dump(p2idx, f)
        with open(os.path.join(self.dataset_dir, "adj_p.pkl"), "wb") as f:
            pickle.dump(adj_p, f)

        return p_seqs, p_list, p2idx, adj_p

    def generate_pq_matrix111(self):
        df = pd.read_csv(self.dataset_path).dropna(subset=["skill_name"]) \
            .drop_duplicates(subset=["order_id", "skill_name"]) \
            .sort_values(by=["order_id"])

        u_list = np.unique(df["studentId"].values)
        q_list = np.unique(df["skill_name"].values)  # concept list
        p_list = np.unique(df["problem_id"].values)  # problem list

        u2idx = {u: idx for idx, u in enumerate(u_list)}
        q2idx = {q: idx for idx, q in enumerate(q_list)}
        p2idx = {p: idx for idx, p in enumerate(p_list)}

        # 构筑邻接关系
        num_q = q_list.shape[0]
        num_p = p_list.shape[0]
        pq_rel = np.zeros((num_p, num_q), dtype=np.integer)

        p_idx = np.array([p2idx[p] for p in df["problemId"]])
        q_idx = np.array([q2idx[q] for q in df["skill_name"]])

        pq_rel[p_idx, q_idx] = 1

        with open(os.path.join(self.dataset_dir, "pq_rel.pkl"), "wb") as f:  # 需要返回plist吗？
            pickle.dump(pq_rel, f)

        return pq_rel  # 之后再想办法优化

    def preprocess(self):

        df = pd.read_csv(self.dataset_path)
        df = df.dropna(subset=["studentId", "problemId", "correct", "skill_id"])

        # 处理skills，将skill_id列切分为具体的skill_id
        skills = df["skill_id"].astype(str).str.split("_").apply(convert_to_int)
        # 切分处理好后，还给df
        df["skill_id"] = skills

        u_list = np.unique(df["studentId"].values)  # 统计所有用户并编号
        # q_list = np.unique(df["skill_name"].values)  # 以concept作为节点
        q_list = np.unique(df["skill_id"].explode().values)  # 统计所有concept并编号；一题多概念，因此要explode展开
        p_list = np.unique(df["problemId"].values)  # 统计所有问题并编号

        u2idx = {u: idx for idx, u in enumerate(u_list)}
        q2idx = {q: idx for idx, q in enumerate(q_list)}
        p2idx = {p: idx for idx, p in enumerate(p_list)}

        q_seqs = []
        r_seqs = []
        p_seqs = []

        # 构筑concept间的邻接关系（对应到knowledge state上；不构筑问题间的图，一是因为问题sparsity，而是因为显存不够）
        num_q = q_list.shape[0]
        adj_trans = np.zeros((num_q, num_q), dtype=np.float32)

        # 构筑problem和concept连接关系
        num_p = p_list.shape[0]
        pq_rel = np.zeros((num_p, num_q), dtype=np.integer)
        df_group = df[["problemId", "skill_id"]].groupby("problemId", as_index=False)
        df_group = df_group.agg({'skill_id': 'first'})
        for p, Q in zip(df_group["problemId"], df_group["skill_id"]):
            for q in Q:
                p_idx = p2idx[p]
                q_idx = q2idx[q]
                pq_rel[p_idx][q_idx] = 1

        for u in u_list:  # 遍历所有用户，按照用户来组织条目
            df_u = df[df["studentId"] == u]

            # 这里返回p_seq，因为q_seq不是一一对应的关系（一题多概念），可通过pq_matrix到前台再处理
            p_seq = np.array([p2idx[p] for p in df_u["problemId"]])
            r_seq = df_u["correct"].values

            p_seqs.append(p_seq)  # 这里尚未pad，是到外面的collate_fn完成padding的
            r_seqs.append(r_seq)

            # 开始处理一题多概念的问题
            for Q1, Q2 in zip(df_u["skill_id"], df_u["skill_id"][1:]):
                for q1, q2 in itertools.product(Q1, Q2):
                    q1_idx = q2idx[q1]
                    q2_idx = q2idx[q2]
                    adj_trans[q1_idx][q2_idx] += 1

        # 取平均
        row_sum = np.sum(adj_trans, axis=1, keepdims=True)
        row_sum = row_sum.clip(min=1)  # 防止除零错误
        adj_trans = adj_trans / row_sum

        # 取非零
        # adj_trans[adj_trans != 0] = 1

        # with open(os.path.join(self.dataset_dir, "q_seqs.pkl"), "wb") as f:
        #     pickle.dump(q_seqs, f)
        with open(os.path.join(self.dataset_dir, "p_seqs.pkl"), "wb") as f:
            pickle.dump(p_seqs, f)
        with open(os.path.join(self.dataset_dir, "r_seqs.pkl"), "wb") as f:
            pickle.dump(r_seqs, f)
        with open(os.path.join(self.dataset_dir, "p_list.pkl"), "wb") as f:
            pickle.dump(p_list, f)
        with open(os.path.join(self.dataset_dir, "q_list.pkl"), "wb") as f:
            pickle.dump(q_list, f)
        with open(os.path.join(self.dataset_dir, "u_list.pkl"), "wb") as f:
            pickle.dump(u_list, f)
        with open(os.path.join(self.dataset_dir, "p2idx.pkl"), "wb") as f:
            pickle.dump(p2idx, f)
        with open(os.path.join(self.dataset_dir, "q2idx.pkl"), "wb") as f:
            pickle.dump(q2idx, f)
        with open(os.path.join(self.dataset_dir, "u2idx.pkl"), "wb") as f:
            pickle.dump(u2idx, f)
        with open(os.path.join(self.dataset_dir, "adj_trans.pkl"), "wb") as f:
            pickle.dump(adj_trans, f)
        with open(os.path.join(self.dataset_dir, "pq_rel.pkl"), "wb") as f:
            pickle.dump(pq_rel, f)

        return p_seqs, r_seqs, p_list, q_list, u_list, p2idx, q2idx, u2idx, adj_trans, pq_rel


def convert_to_int(x):
    return [int(float(item)) for item in x]


"""
以下这段代码是为了生成每个问题对应的多个skills
"""


def preprocess():
    # d.preprocess()
    df = pd.read_csv("../datasets/ASSIST2017_for_mcgkt/anonymized_full_release_competition_dataset.csv")
    df['sid'] = pd.factorize(df['skill'])[0] + 1

    sid_counts = df.groupby('problemId')['sid'].nunique()
    multi_skill_problems = sid_counts[sid_counts > 1]
    print(multi_skill_problems)

    skill_list = df.groupby('problemId')['sid'].unique().apply(lambda arr: '_'.join(map(str, sorted(arr))))
    df['skill_id'] = df['problemId'].map(skill_list)

    df.to_csv("../datasets/ASSIST2017_for_mcgkt/anonymized_full_release_competition_dataset_with_skill_id.csv", index=False)


if __name__ == "__main__":
    preprocess()