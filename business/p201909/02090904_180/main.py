import tushare as ts
import pandas as pd
from collections import OrderedDict
import os


# 获取数据
ts.set_token('9c59c147730560f538f15d104c41e4f32ecc21e1a7e3544d2f280f7f')
pro = ts.pro_api()

exchanges = ['CFFEX', 'DCE', 'CZCE', 'SHFE', 'INE']
# 交易所代码 CFFEX-中金所 DCE-大商所 CZCE-郑商所 SHFE-上期所 INE-上海国际能源交易中心
symbols = ['NI1911', 'C']
headlines = "trade_date,symbol,vol,vol_chg,long_hld,long_chg,short_hld,short_chg\n"


for ex in exchanges:
    save = []
    for sy in symbols:
        df1 = pro.fut_holding(symbol=sy, exchange=ex)
        del df1['broker']
        for x, y in df1.iterrows():
            rows = {"trade_date": y['trade_date'],
                    "symbol": y['symbol'],
                    "vol": y['vol'],
                    "vol_chg": y['vol_chg'],
                    "long_hld": y['long_hld'],
                    "long_chg": y['long_chg'],
                    "short_hld": y['short_hld'],
                    "short_chg": y['short_chg'],
                    "long_short": y['long_hld'] - y['short_hld'],
                    }
            save.append(rows)
    if save:
        df = pd.DataFrame(save)
        resuluts = []
        for x2, y2 in df.groupby(by=['trade_date', 'symbol']):
            # trade_date, symbol, vol, vol_chg, long_hld, long_chg, short_hld, short_chg

            rows2 = OrderedDict({"trade_date": y2['trade_date'].values[0],
                                 "symbol": y2['symbol'].values[0],
                                 "vol": y2['vol'].sum(),
                                 "vol_chg": y2['vol_chg'].sum(),
                                 "long_hld": y2['long_hld'].sum(),
                                 "long_chg": y2['long_chg'].sum(),
                                 "short_hld": y2['short_hld'].sum(),
                                 "short_chg": y2['short_chg'].sum(),
                                 "long_short": y2['long_short'].sum(),
                                 })

            resuluts.append(rows2)

        df_res = pd.DataFrame(resuluts)
        for save_x, save_y in df_res.groupby(by='symbol'):

            for item_x, item_y in save_y.iterrows():

                saved_ = None
                fw = None
                save_file = "{}.csv".format(save_x)
                if os.path.isfile(save_file):
                    fr = open(save_file, 'r', encoding='utf-8')
                    saved_ = [i.replace("\n", "") for i in fr]
                    fw = open(save_file, 'a+', encoding='utf-8')
                    res = "{},{},{},{},{},{},{},{},{}".format(item_y['trade_date'],
                                                              save_x,
                                                              item_y['vol'],
                                                              item_y['vol_chg'],
                                                              item_y['long_hld'],
                                                              item_y['long_chg'],
                                                              item_y['short_hld'],
                                                              item_y['short_chg'],
                                                              item_y['long_short'])
                    if res not in saved_:
                        fw.writelines(res + "\n")
                else:
                    fw = open(save_file, 'a+', encoding='utf-8')
                    fw.writelines(headlines)
                    res = "{},{},{},{},{},{},{},{},{}".format(item_y['trade_date'],
                                                              save_x,
                                                              item_y['vol'],
                                                              item_y['vol_chg'],
                                                              item_y['long_hld'],
                                                              item_y['long_chg'],
                                                              item_y['short_hld'],
                                                              item_y['short_chg'],
                                                              item_y['long_short'])
                    fw.writelines(res + "\n")

