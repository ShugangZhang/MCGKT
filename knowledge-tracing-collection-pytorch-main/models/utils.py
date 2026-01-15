import numpy as np
import torch

from torch.nn.utils.rnn import pad_sequence

if torch.cuda.is_available():
    from torch.cuda import FloatTensor
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
else:
    from torch import FloatTensor

# FOR 2015   面向concept
def match_seq_len(q_seqs, r_seqs, seq_len, pad_val=-1):
    '''
        Args:
            q_seqs: the question(KC) sequences with the size of \
                [batch_size, some_sequence_length]
            r_seqs: the response sequences with the size of \
                [batch_size, some_sequence_length]

            Note that the "some_sequence_length" is not uniform over \
                the whole batch of q_seqs and r_seqs

            seq_len: the sequence length to match the q_seqs, r_seqs \
                to same length
            pad_val: the padding value for the sequence with the length \
                longer than seq_len

        Returns:
            proc_q_seqs: the processed q_seqs with the size of \
                [batch_size, seq_len + 1]
            proc_r_seqs: the processed r_seqs with the size of \
                [batch_size, seq_len + 1]
    '''
    proc_q_seqs = []
    proc_r_seqs = []

    for q_seq, r_seq in zip(q_seqs, r_seqs):
        i = 0
        while i + seq_len + 1 < len(q_seq):
            proc_q_seqs.append(q_seq[i:i + seq_len + 1])
            proc_r_seqs.append(r_seq[i:i + seq_len + 1])

            i += seq_len + 1


        # 对于长度过短的，要进行质量控制！原始代码未做到这一点
        # proc_q_seqs.append(
        #     np.concatenate(
        #         [
        #             q_seq[i:],  # 从原始序列中不足以摘取seq_len的位置开始读取
        #             np.array([pad_val] * (i + seq_len + 1 - len(q_seq)))  # 准备一个全-1向量，将原向量补齐（全部padding为-1）
        #         ]
        #     )
        # )
        # proc_r_seqs.append(
        #     np.concatenate(
        #         [
        #             r_seq[i:],
        #             np.array([pad_val] * (i + seq_len + 1 - len(q_seq)))
        #         ]
        #     )
        # )



        # 以下为改进：保证最短读取序列不小于10
        q_pad = np.concatenate(
                [
                    q_seq[i:],  # 从原始序列中不足以摘取seq_len的位置开始读取
                    np.array([pad_val] * (i + seq_len + 1 - len(q_seq)))  # 准备一个全-1向量，将原向量补齐（全部padding为-1）
                ]
            )

        r_pad = np.concatenate(
                [
                    r_seq[i:],  # 从原始序列中不足以摘取seq_len的位置开始读取
                    np.array([pad_val] * (i + seq_len + 1 - len(q_seq)))  # 准备一个全-1向量，将原向量补齐（全部padding为-1）
                ]
            )
        count = np.count_nonzero(q_pad!=pad_val)

        if (count >= 10):
            proc_q_seqs.append(q_pad)
            proc_r_seqs.append(r_pad)

    return proc_q_seqs, proc_r_seqs

# FOR 2009
def match_seq_len_2009(q_seqs, r_seqs, p_seqs, seq_len, pad_val=-1):
    '''
        Args:
            q_seqs: the question(KC) sequences with the size of \
                [batch_size, some_sequence_length]
            r_seqs: the response sequences with the size of \
                [batch_size, some_sequence_length]

            Note that the "some_sequence_length" is not uniform over \
                the whole batch of q_seqs and r_seqs

            seq_len: the sequence length to match the q_seqs, r_seqs \
                to same length
            pad_val: the padding value for the sequence with the length \
                longer than seq_len

        Returns:
            proc_q_seqs: the processed q_seqs with the size of \
                [batch_size, seq_len + 1]
            proc_r_seqs: the processed r_seqs with the size of \
                [batch_size, seq_len + 1]
    '''
    proc_q_seqs = []
    proc_r_seqs = []
    proc_p_seqs = []

    for q_seq, r_seq, p_seq in zip(q_seqs, r_seqs, p_seqs):
        i = 0
        while i + seq_len + 1 < len(q_seq):
            proc_q_seqs.append(q_seq[i:i + seq_len + 1])
            proc_r_seqs.append(r_seq[i:i + seq_len + 1])
            proc_p_seqs.append(p_seq[i:i + seq_len + 1])

            i += seq_len + 1

        proc_q_seqs.append(
            np.concatenate(
                [
                    q_seq[i:],
                    np.array([pad_val] * (i + seq_len + 1 - len(q_seq)))
                ]
            )
        )
        proc_r_seqs.append(
            np.concatenate(
                [
                    r_seq[i:],
                    np.array([pad_val] * (i + seq_len + 1 - len(q_seq)))
                ]
            )
        )
        proc_p_seqs.append(
            np.concatenate(
                [
                    p_seq[i:],
                    np.array([pad_val] * (i + seq_len + 1 - len(p_seq)))
                ]
            )
        )

    return proc_q_seqs, proc_r_seqs, proc_p_seqs

# FOR 2009  面向problem/question
def match_seq_len_2009_corrected(p_seqs, r_seqs, seq_len, pad_val=-1):
    '''
        Args:
            q_seqs: the question(KC) sequences with the size of \
                [batch_size, some_sequence_length]
            r_seqs: the response sequences with the size of \
                [batch_size, some_sequence_length]

            Note that the "some_sequence_length" is not uniform over \
                the whole batch of q_seqs and r_seqs

            seq_len: the sequence length to match the q_seqs, r_seqs \
                to same length
            pad_val: the padding value for the sequence with the length \
                longer than seq_len

        Returns:
            proc_q_seqs: the processed q_seqs with the size of \
                [batch_size, seq_len + 1]
            proc_r_seqs: the processed r_seqs with the size of \
                [batch_size, seq_len + 1]
    '''
    proc_r_seqs = []
    proc_p_seqs = []

    for r_seq, p_seq in zip(r_seqs, p_seqs):
        i = 0
        while i + seq_len + 1 < len(p_seq):
            proc_r_seqs.append(r_seq[i:i + seq_len + 1])
            proc_p_seqs.append(p_seq[i:i + seq_len + 1])

            i += seq_len + 1


        # 对于长度过短的，要进行质量控制！原始代码未做到这一点
        # proc_r_seqs.append(
        #     np.concatenate(
        #         [
        #             r_seq[i:],
        #             np.array([pad_val] * (i + seq_len + 1 - len(p_seq)))
        #         ]
        #     )
        # )
        # proc_p_seqs.append(
        #     np.concatenate(
        #         [
        #             p_seq[i:],
        #             np.array([pad_val] * (i + seq_len + 1 - len(p_seq)))
        #         ]
        #     )
        # )


        # 以下为改进：保证最短读取序列不小于10
        p_pad = np.concatenate(
                [
                    p_seq[i:],  # 从原始序列中不足以摘取seq_len的位置开始读取
                    np.array([pad_val] * (i + seq_len + 1 - len(p_seq)))  # 准备一个全-1向量，将原向量补齐（全部padding为-1）
                ]
            )

        r_pad = np.concatenate(
                [
                    r_seq[i:],  # 从原始序列中不足以摘取seq_len的位置开始读取
                    np.array([pad_val] * (i + seq_len + 1 - len(p_seq)))  # 准备一个全-1向量，将原向量补齐（全部padding为-1）
                ]
            )
        count = np.count_nonzero(p_pad!=pad_val)

        if (count >= 10):
            proc_p_seqs.append(p_pad)
            proc_r_seqs.append(r_pad)

    return proc_p_seqs, proc_r_seqs


def collate_fn_2015(batch, pad_val=-1):
    '''
        The collate function for torch.utils.data.DataLoader

        Returns:
            q_seqs: the question(KC) sequences with the size of \
                [batch_size, maximum_sequence_length_in_the_batch]
            r_seqs: the response sequences with the size of \
                [batch_size, maximum_sequence_length_in_the_batch]
            qshft_seqs: the question(KC) sequences which were shifted \
                one step to the right with ths size of \
                [batch_size, maximum_sequence_length_in_the_batch]
            rshft_seqs: the response sequences which were shifted \
                one step to the right with ths size of \
                [batch_size, maximum_sequence_length_in_the_batch]
            mask_seqs: the mask sequences indicating where \
                the padded entry is with the size of \
                [batch_size, maximum_sequence_length_in_the_batch]
    '''
    q_seqs = []
    r_seqs = []

    qshft_seqs = []
    rshft_seqs = []



    for q_seq, r_seq in batch:
        q_seqs.append(FloatTensor(q_seq[:-1]))
        r_seqs.append(FloatTensor(r_seq[:-1]))
        qshft_seqs.append(FloatTensor(q_seq[1:]))
        rshft_seqs.append(FloatTensor(r_seq[1:]))

    q_seqs = pad_sequence(
        q_seqs, batch_first=True, padding_value=pad_val
    )
    r_seqs = pad_sequence(
        r_seqs, batch_first=True, padding_value=pad_val
    )
    qshft_seqs = pad_sequence(
        qshft_seqs, batch_first=True, padding_value=pad_val
    )
    rshft_seqs = pad_sequence(
        rshft_seqs, batch_first=True, padding_value=pad_val
    )

    mask_seqs = (q_seqs != pad_val) * (qshft_seqs != pad_val)

    q_seqs, r_seqs, qshft_seqs, rshft_seqs = \
        (q_seqs * mask_seqs, r_seqs * mask_seqs,
         qshft_seqs * mask_seqs, rshft_seqs * mask_seqs)

    return q_seqs, r_seqs, qshft_seqs, rshft_seqs, mask_seqs

def collate_fn(batch, pad_val=-1):
    '''
        The collate function for torch.utils.data.DataLoader

        Returns:
            q_seqs: the question(KC) sequences with the size of \
                [batch_size, maximum_sequence_length_in_the_batch]
            r_seqs: the response sequences with the size of \
                [batch_size, maximum_sequence_length_in_the_batch]
            qshft_seqs: the question(KC) sequences which were shifted \
                one step to the right with ths size of \
                [batch_size, maximum_sequence_length_in_the_batch]
            rshft_seqs: the response sequences which were shifted \
                one step to the right with ths size of \
                [batch_size, maximum_sequence_length_in_the_batch]
            mask_seqs: the mask sequences indicating where \
                the padded entry is with the size of \
                [batch_size, maximum_sequence_length_in_the_batch]
    '''
    q_seqs = []
    r_seqs = []
    p_seqs = []  # problem sequence
    qshft_seqs = []
    rshft_seqs = []
    pshft_seqs = []


    for p_seq, r_seq in batch:
        # q_seqs.append(FloatTensor(q_seq[:-1]))
        r_seqs.append(FloatTensor(r_seq[:-1]))
        p_seqs.append(FloatTensor(p_seq[:-1]))
        # qshft_seqs.append(FloatTensor(q_seq[1:]))
        rshft_seqs.append(FloatTensor(r_seq[1:]))
        pshft_seqs.append(FloatTensor(p_seq[1:]))


    # q_seqs = pad_sequence(
    #     q_seqs, batch_first=True, padding_value=pad_val
    # )
    r_seqs = pad_sequence(  # 形式上的作用：把64长度的list转为[64,seq_len]的tensor，实际上并未padding，debug发现送入之前已经padding好了
        r_seqs, batch_first=True, padding_value=pad_val
    )
    p_seqs = pad_sequence(
        p_seqs, batch_first=True, padding_value=pad_val
    )
    # qshft_seqs = pad_sequence(
    #     qshft_seqs, batch_first=True, padding_value=pad_val
    # )
    rshft_seqs = pad_sequence(
        rshft_seqs, batch_first=True, padding_value=pad_val
    )
    pshft_seqs = pad_sequence(
        pshft_seqs, batch_first=True, padding_value=pad_val
    )

    # mask_seqs = (q_seqs != pad_val) * (qshft_seqs != pad_val)
    mask_seqs = (p_seqs != pad_val) * (pshft_seqs != pad_val)

    # q_seqs, r_seqs, p_seqs, qshft_seqs, rshft_seqs, pshft_seqs = \
    #     (q_seqs * mask_seqs, r_seqs * mask_seqs, p_seqs * mask_seqs,
    #      qshft_seqs * mask_seqs, rshft_seqs * mask_seqs, pshft_seqs * mask_seqs)

    if (not mask_seqs.any()):
        print(mask_seqs)  # 为什么会有全为False的情况？
        print(p_seqs)
        print(pshft_seqs)

    r_seqs, p_seqs, rshft_seqs, pshft_seqs = \
        (r_seqs * mask_seqs, p_seqs * mask_seqs,
         rshft_seqs * mask_seqs, pshft_seqs * mask_seqs)


    return p_seqs, r_seqs, pshft_seqs, rshft_seqs, mask_seqs
