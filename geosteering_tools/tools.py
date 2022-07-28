import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import *
from skfda import FDataGrid
from skfda.misc import cosine_similarity
from skfda.misc.metrics import LpDistance


def plot_curves(curve1, curve2):
    x = range(len(curve1))
    plt.plot(x, curve1, '-r')
    plt.plot(x, curve2, '-g')
    plt.show()


def get_alpha(log):
    '''
    The function transforms the original SP or GR log
    and returns list of normalized arrays
    '''
    max_well = np.max(log)
    min_well = np.min(log)
    norm_log = np.array([(x - min_well) / (max_well - min_well) for x in log], dtype=np.double)
    return norm_log


def lp_distance(curve1, curve2):
    lp_dist = LpDistance(p=2)
    fd1 = FDataGrid(curve1)
    fd2 = FDataGrid(curve2)
    distance = lp_dist(fd1, fd2)
    return distance[0]


def cos_sim(curve1, curve2):
    fd1 = FDataGrid(curve1)
    fd2 = FDataGrid(curve2)
    score = cosine_similarity(fd1, fd2)
    return score[0]


def filter_signal(signal, threshold=1e8):
    fourier = rfft(signal)
    frequencies = rfftfreq(signal.size, d=20e-3 / signal.size)
    fourier[frequencies > threshold] = 0
    return irfft(fourier)


def get_val(point, df, column, target_column='MD'):
    df_delta = df.copy()
    df_delta['delta'] = df_delta[target_column] - point
    prev_d = df_delta[df_delta.delta < 0]['delta'].max()
    next_d = df_delta[df_delta.delta >= 0]['delta'].min()
    prev_v = df_delta[df_delta['delta'] == prev_d][column].values[0]
    next_v = df_delta[df_delta['delta'] == next_d][column].values[0]
    prev_d, next_d = abs(prev_d), abs(next_d)
    val = prev_v + ((next_v - prev_v) * prev_d) / (prev_d + next_d)
    return val


def magn(a, dim):
    dim = dim - 1
    b = np.sum(np.conj(a) * a, axis=dim)
    return np.sqrt(b)


def parallel_curves(x, y, d=1, flag1=True):
    if x.ndim != 1 or y.ndim != 1:
        raise Exception("X and Y must be vectors")

    dx = np.gradient(x)
    dy = np.gradient(y)

    dx2 = np.gradient(dx)
    dy2 = np.gradient(dy)

    nv = np.ndarray(shape=[len(dy), 2])
    nv[:, 0] = dy
    nv[:, 1] = -dx

    unv = np.zeros(shape=nv.shape)
    norm_nv = magn(nv, 2)

    unv[:, 0] = nv[:, 0] / norm_nv
    unv[:, 1] = nv[:, 1] / norm_nv

    R = ((dx ** 2 + dy ** 2) ** 1.5) / (np.abs(dx * dy2 - dy * dx2))

    overlap = R < d

    dy3 = np.zeros(shape=dy2.shape)
    dy3[dy2 > 0] = 1
    concavity = 2 * dy3 - 1

    if flag1 is True:
        x_inner = x - unv[:, 0] * concavity * d
        y_inner = y - unv[:, 1] * concavity * d

        x_outer = x + unv[:, 0] * concavity * d
        y_outer = y + unv[:, 1] * concavity * d
    else:
        x_inner = x - unv[:, 0] * d - 2
        y_inner = y - unv[:, 1] * d

        x_outer = x + unv[:, 0] * d + 2
        y_outer = y + unv[:, 1] * d

    res = {'x_inner': x_inner,
           'y_inner': y_inner,
           'x_outer': x_outer,
           'y_outer': y_outer,
           'R': R,
           'unv': unv,
           'concavity': concavity,
           'overlap': overlap}
    return res