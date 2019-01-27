from bs4 import BeautifulSoup as bs
import requests
import pandas as pd
from collections import OrderedDict
import matplotlib.pyplot as plt

data = pd.read_csv("mysubset.csv")


name1 = 'admissions.csv'
name4 = 'statepopulation.csv'
name3 = 'G20.csv'

name2 = 'countrydata.csv'
name5 = 'summer.csv'

# Country,Code,Population,GDP per Capita  df1
# Year,City,Sport,Discipline,Athlete,Country,Gender,Event,Medal, df2
df1 = pd.read_csv(name2)
df2 = pd.read_csv(name5)


# def mySubset(df1, df2):
# Year, Country, Medal, Population, GDPpc
def mySubset(df1, df2):
    code = df1['Code'].values.tolist()
    save = []
    for x in code:
        Population = df1[df1['Code'].isin([x])]['Population'].values.tolist()[0]
        GDPpc = df1[df1['Code'].isin([x])]['GDP per Capita'].values.tolist()[0]
        data = df2[df2['Country'].isin([x])]
        for x1, y1 in data.iterrows():
            year = y1['Year']
            country = x
            Medal = df2[df2['Year'].isin([year]) & df2['Country'].isin([x])]['Medal']
            rows = OrderedDict({'Year': year, 'Country': country, 'Medal': Medal, 'Population': Population, 'GDPpc': GDPpc})
            print(rows)
            save.append(rows)

    df = pd.DataFrame(save)
    df.to_csv("mysubset.csv", index=None)


mySubset(df1, df2)
exit(1)
g20 = pd.read_csv(name3).sort_values(by='HDI', ascending=False).head(5)['Member'].values.tolist()

data = pd.read_csv("summer.csv")
res = []
for x, y in data.groupby(by='Year'):
    res.append(y.shape[0])

print(len(res), res)

