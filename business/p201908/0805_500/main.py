import pandas as pd
import matplotlib.pyplot as plt
import os

"""
python销售数据分析，具体要求是从省份层面 城市层面和医院层面 分析2014年-2018年的销售趋势和特点
着重分析前150家医院和60个城市
"""

# 画图 显示中文
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


if __name__ == "__main__":

    for method in ['splitbyyear', 'top150', 'top60', 'province', 'top150pie', 'top60pie']:
        if method == 'splitbyyear':
            # 按照年份分割数据
            data = pd.read_excel("赫赛汀2018数据.xlsx")
            data_na = data[~data['销量（RMB）'].isin([None])]
            data_na['年份'] = data_na['年月'].apply(lambda x: str(x)[:4])
            for date, y_data in data_na.groupby(by='年份'):
                if not os.path.exists("./datedata"):
                    os.makedirs("./datedata")
                y_data.to_excel("./datedata/{}.xlsx".format(date), index=None)

        if method == 'top150':
            if not os.path.exists("./top150"):
                os.makedirs('./top150')

            for file in os.listdir("./datedata"):
                years = file.replace(".xlsx", "")
                file_name = os.path.join("./datedata", file)
                data = pd.read_excel(file_name)
                save = []
                for x, y in data.groupby(by='医院'):
                    rows = {}
                    # 年月	医院	城市	省份	销量（RMB）	年份
                    rows['医院'] = x
                    rows['销量（RMB）'] = y['销量（RMB）'].sum()
                    rows['年份'] = y['年份'].values[0]
                    rows['城市'] = y['城市'].values[0]
                    rows['省份'] = y['省份'].values[0]
                    save.append(rows)
                df = pd.DataFrame(save)
                df.to_excel("./top150/{}".format(file), index=None)
                print(df.shape)

        if method == 'top60':

            if not os.path.exists("./top60"):
                os.makedirs('./top60')

            for file in os.listdir("./datedata"):
                years = file.replace(".xlsx", "")
                file_name = os.path.join("./datedata", file)
                data = pd.read_excel(file_name)
                save = []
                for x, y in data.groupby(by='城市'):
                    rows = {}
                    # 年月	医院	城市	省份	销量（RMB）	年份
                    rows['城市'] = x
                    rows['销量（RMB）'] = y['销量（RMB）'].sum()
                    rows['年份'] = y['年份'].values[0]
                    rows['医院'] = y['医院'].values[0]
                    rows['省份'] = y['省份'].values[0]

                    save.append(rows)
                df = pd.DataFrame(save)
                df.to_excel("./top60/{}".format(file), index=None)
                print(df.shape)

        if method == 'province':
            fwp = open('省份.txt', 'w', encoding='utf-8')
            if not os.path.exists("./results"):
                os.makedirs("./results")

            for file in os.listdir("./top150"):
                years = file.replace(".xlsx", "")
                file_name = os.path.join("./datedata", file)
                data = pd.read_excel(file_name)
                total_price = data['销量（RMB）'].sum()
                top_150h = data.sort_values(by='销量（RMB）', ascending=False).head(150)
                rows = {}
                for province, p_data in top_150h.groupby(by='省份'):
                    sale_price = p_data['销量（RMB）'].sum() / total_price
                    rows[province] = sale_price
                for k, v in rows.items():
                    res = "{}，{},{}".format(years, k, v)
                    fwp.writelines(res + "\n")

                value = list(rows.values())
                names = list(rows.keys())

                plt.figure(figsize=(30, 10), dpi=200)
                plt.bar(names, value, width=0.3, color='y')
                plt.title("各省销量对比")
                plt.xticks(rotation=15)
                for a, b in zip(names, value):
                    plt.text(a, b + 0.001, '{:0.2f}'.format(b*100), ha='center', va='bottom', fontsize=9)
                plt.savefig("./results/Top150_各省销量对比-{}.png".format(years))
                plt.close()
                print("保存成功")
            fwp = open('省份.txt', 'w', encoding='utf-8')

            for file in os.listdir("./top60"):
                years = file.replace(".xlsx", "")
                file_name = os.path.join("./datedata", file)
                data = pd.read_excel(file_name)
                total_price = data['销量（RMB）'].sum()
                top_sixtwy = data.sort_values(by='销量（RMB）', ascending=False).head(60)
                rows = {}
                for province, p_data in top_sixtwy.groupby(by='省份'):
                    sale_price = p_data['销量（RMB）'].sum() / total_price
                    rows[province] = sale_price

                for k, v in rows.items():
                    res = "{}，{},{}".format(years, k, v)
                    fwp.writelines(res + "\n")

                value = list(rows.values())
                names = list(rows.keys())
                plt.figure(figsize=(30, 10), dpi=200)
                plt.bar(names, value, width=0.3, color='r')
                plt.title("各省销量对比")
                plt.xticks(rotation=15)
                for a, b in zip(names, value):
                    plt.text(a, b + 0.001,  '{:0.2f}'.format(b*100), ha='center', va='bottom', fontsize=9)
                plt.savefig("./results/Top60_各省销量对比-{}.png".format(years))
                plt.close()
                print("保存成功")

        if method == 'top150pie':
            fwnames = open('top 150 的医院名单变化.txt', 'w', encoding='utf-8')

            for file in os.listdir("./top150"):
                years = file.replace(".xlsx", "")
                file_name = os.path.join("./top150", file)
                data = pd.read_excel(file_name)
                top_150h = data.sort_values(by='销量（RMB）', ascending=False).head(150)

                hos_names = top_150h['医院'].values
                fwnames.writelines("{},{}\n".format(years, " ".join(hos_names)))

                # top 150医院， 各省企业数量占比
                rows0 = {}
                for province, p_data in top_150h.groupby(by='省份'):
                    rows0[province] = p_data.shape[0]
                value = list(rows0.values())
                names = list(rows0.keys())

                plt.figure(figsize=(30, 10), dpi=200)
                plt.pie(value, labels=names, autopct='%1.1f%%', shadow=False, startangle=150)
                plt.title("Top150_各省医院数量占比饼图")
                plt.savefig("./results/Top150_各省医院数量占比饼图-{}.png".format(years))
                plt.close()
                print("保存成功")

        if method == 'top60pie':
            fwnames = open('top 60 的城市名单变化.txt', 'w', encoding='utf-8')
            for file in os.listdir("./top60"):
                years = file.replace(".xlsx", "")
                file_name = os.path.join("./top60", file)
                data = pd.read_excel(file_name)
                top_sixtwy = data.sort_values(by='销量（RMB）', ascending=False).head(60)
                hos_names = top_sixtwy['城市'].values
                fwnames.writelines("{},{}\n".format(years, " ".join(hos_names)))

                rows0 = {}
                for province, p_data in top_sixtwy.groupby(by='省份'):
                    rows0[province] = p_data.shape[0]

                value = list(rows0.values())
                names = list(rows0.keys())
                plt.figure(figsize=(30, 10), dpi=200)
                plt.pie(value, labels=names, autopct='%1.1f%%', shadow=False, startangle=150)
                plt.title("Top60_省份分布占比饼图-{}".format(years))
                plt.savefig("./results/Top60_省份分布占比饼图-{}.png".format(years))
                plt.close()
                print("保存成功")

