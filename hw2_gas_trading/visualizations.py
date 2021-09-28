import matplotlib.pyplot as plt

import pandas as pd
from pandas.plotting import register_matplotlib_converters

import numpy as np

from sklearn.manifold import TSNE


register_matplotlib_converters()



def show_entire_period_price(serieses):
    date = [s.date for s in serieses]
    price = [s.get_mean_price() for s in serieses]

    date_time = pd.to_datetime(date)

    df = pd.DataFrame()
    df['value'] = price
    df = df.set_index(date_time)

    plt.plot(df)
    plt.gcf().autofmt_xdate()
    plt.title("Изменение средней цены по дням")
    plt.xlabel("Дата")
    plt.ylabel("Средняя цена")
    plt.show()


def show_entire_period_lot_size(serieses):
    date = [s.date for s in serieses]
    lot_size = [s.get_sum_lot_size() for s in serieses]

    date_time = pd.to_datetime(date)

    df = pd.DataFrame()
    df['value'] = lot_size
    df = df.set_index(date_time)

    plt.plot(df)
    plt.gcf().autofmt_xdate()
    plt.title("Изменение объема проданного газа по дням")
    plt.xlabel("Дата")
    plt.ylabel("Объем газа")
    plt.show()


def show_entire_period_money(serieses):
    date = [s.date for s in serieses]
    money = [s.get_sum_money() for s in serieses]

    date_time = pd.to_datetime(date)

    df = pd.DataFrame()
    df['value'] = money
    df = df.set_index(date_time)

    plt.plot(df)
    plt.gcf().autofmt_xdate()
    plt.title("Изменение объема торгов по дням")
    plt.xlabel("Дата")
    plt.ylabel("Объем торгов (суммарная цена проданного газа)")
    plt.show()


def show_series_price(chart):
    minutes = np.arange(60)

    plt.plot(minutes, chart)
    plt.title("Изменение цены в один день")
    plt.xlabel("Минута")
    plt.ylabel("Цена")
    plt.show()

def show_tsne(charts, metric_name):
    if metric_name == "cos":
        metric = "cosine"
    elif metric_name == "mae":
        metric = "euclidean"
    else:
        metric = lambda u, v: ((u[:, None] - v[None, :]) ** 2).sum() ** 0.5

    x_embedded = TSNE(n_components=2, metric=metric, square_distances=True).fit_transform(charts)

    dist = x_embedded[:, None, :] - x_embedded[None, :, :]
    dist = (dist ** 2).sum(axis=2)
    id_a, id_b = np.unravel_index(dist.argmax(), dist.shape)

    n = x_embedded.shape[0]
    ids = np.arange(n)
    dist[ids, ids] = 1e9
    id_c, id_d = np.unravel_index(dist.argmin(), dist.shape)

    f, axarr = plt.subplots(1, 3)
    f.set_figwidth(20)

    axarr[0].scatter(x_embedded[:, 0], x_embedded[:, 1], label="все точки")
    axarr[0].scatter(x_embedded[[id_a, id_b], 0], x_embedded[[id_a, id_b], 1], label="наиболее отдаленные")
    axarr[0].scatter(x_embedded[[id_c, id_d], 0], x_embedded[[id_c, id_d], 1], label="ближайшие")
    axarr[0].set_title(f"TSNE для графиков изменения цены, метрика {metric_name}")
    axarr[0].legend()

    minutes = np.arange(60)

    axarr[1].plot(minutes, charts[id_a])
    axarr[1].plot(minutes, charts[id_b])
    axarr[1].set_title(f"Изменение цены (наиболее отдаленные), метрика {metric_name}")
    axarr[1].set_xlabel("Минута")
    axarr[1].set_ylabel("Цена")

    axarr[2].plot(minutes, charts[id_c])
    axarr[2].plot(minutes, charts[id_d])
    axarr[2].set_title(f"Изменение цены (ближайшие), метрика {metric_name}")
    axarr[2].set_xlabel("Минута")
    axarr[2].set_ylabel("Цена")

    plt.show()

