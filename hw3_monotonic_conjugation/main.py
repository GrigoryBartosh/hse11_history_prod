import numpy as np
from scipy.stats import rankdata


def read_input():
    with open("in.txt", "r") as file:
        data = file.readlines()

    data = [line.split() for line in data]
    data = [list(map(int, line)) for line in data]
    data = np.array(data)

    # check data shape
    assert data.dtype == np.int64, "in.txt must contain only integers"
    assert len(data.shape) == 2, "each line of in.txt must contain two integers"
    assert data.shape[1] == 2, "each line of in.txt must contain two integers"

    # check that we have enough data
    assert data.shape[0] >= 9, "algorithm needs at least 9 elements"

    return data


def write_ans(r1_m_r2, std, conj):
    with open("out.txt", "w") as file:
        print(r1_m_r2, std, conj, file=file)


def sort_by_x(data):
    x = data[:, 0]

    ids = x.argsort()

    data = data[ids]

    return data


def get_y_ranks(data):
    y = data[:, 1]

    # function rankdata returns increasing ranks,
    # so we reverse order
    n = y.shape[0]
    ranks = rankdata(y)
    ranks = n + 1 - ranks

    return ranks, n


def get_r1_r2(ranks):
    n = ranks.shape[0]
    p = int(np.round(n / 3))
    
    r1 = ranks[:p].sum().astype(np.int64)
    r2 = ranks[-p:].sum().astype(np.int64)

    return (r1, r2), p


def check_conj(n, p, r1, r2):
    r1_m_r2 = r1 - r2

    std = (n + 0.5) * (p / 6) ** 0.5
    std = np.round(std).astype(np.int64)

    conj = r1_m_r2 / (p * (n - p))
    conj = np.round(conj, 2)

    return r1_m_r2, std, conj


if __name__ == "__main__":
    data = read_input()

    data = sort_by_x(data)
    ranks, n = get_y_ranks(data)

    (r1, r2), p = get_r1_r2(ranks)

    r1_m_r2, std, conj = check_conj(n, p, r1, r2)

    write_ans(r1_m_r2, std, conj)