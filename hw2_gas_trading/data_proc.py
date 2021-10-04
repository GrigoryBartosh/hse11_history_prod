import sqlite3
import pandas as pd

import numpy as np

from datetime import datetime


class Series:
    def __init__(self, date, start_price, time, lot_size, price):
        self.date = date
        self.start_price = start_price
        self.time = time
        self.lot_size = lot_size
        self.price = price
        
    def get_mean_price(self):
        return (self.lot_size * self.price).sum() / self.lot_size.sum()
        
    def get_sum_lot_size(self):
        return self.lot_size.sum()
        
    def get_sum_money(self):
        return (self.lot_size * self.price).sum()

    def get_price_chart(self):
        minutes = [int(t[3:5]) for t in self.time]
        minutes = np.array(minutes)

        cur_price = self.start_price
        chart = []
        for m in range(60):
            ids = np.where(minutes == m)[0]
            if len(ids) == 0:
                chart += [cur_price]
            else:
                cur_price = (self.lot_size[ids] * self.price[ids]).sum() / self.lot_size[ids].sum()
                chart += [cur_price]

        return np.array(chart)


def read_data(file_name):
    connection = sqlite3.connect(file_name)
    trading_session_df = pd.read_sql('SELECT * FROM Trading_session', connection)
    chart_data_df = pd.read_sql('SELECT * FROM Chart_data', connection)
    return trading_session_df, chart_data_df


def merge_data(trading_session_df, chart_data_df):
    trading_session_df.rename(columns={"id":"session_id"}, inplace = True)
    data_df = pd.merge(trading_session_df, chart_data_df, how="inner", on="session_id")
    return data_df


def filter_monthly(data_df):
    mask = data_df["trading_type"] == "monthly"
    return data_df[mask]


def sort_data(data_df):
    return data_df.sort_values(by=['date', 'time'])


def get_first_deal_id(data_df):
    deal_id = data_df["deal_id"].to_numpy()
    _, ids = np.unique(deal_id, return_index=True)
    ids.sort()
    data_df = data_df.iloc[ids]
    return data_df


def build_serieses(data_df, key="session_id"):
    start_price = 0
    serieses = []

    for k in data_df[key].unique():
        data_i = data_df[data_df[key] == k]
        
        series = Series(
            date=data_i["date"].to_numpy()[0],
            start_price=start_price,
            time=data_i["time"].to_numpy(),
            lot_size=data_i["lot_size"].to_numpy(),
            price=data_i["price"].to_numpy()
        )
        
        serieses += [series]
        start_price = series.get_mean_price()

    return serieses


def prepare_data(file_name, key="session_id"):
    trading_session_df, chart_data_df = read_data(file_name)
    data_df = merge_data(trading_session_df, chart_data_df)
    data_df = filter_monthly(data_df)
    data_df = sort_data(data_df)
    data_df = get_first_deal_id(data_df)
    serieses = build_serieses(data_df, key)
    return serieses