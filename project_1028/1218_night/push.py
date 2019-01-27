from bs4 import BeautifulSoup as bs
import requests
import pandas as pd
from collections import OrderedDict
import matplotlib.pyplot as plt

url = "https://en.wikipedia.org/wiki/List_of_data_breaches"

r = requests.get(url)
data = r.text
soup = bs(data, "html.parser")
rows = soup.find("table").find("tbody").findAll("tr")
myRows = []
for row in rows:
    try:
        teamRank = row.findAll('td')
        entity = teamRank[0].text.replace("\n", "")
        year = teamRank[1].text.replace("\n", "")
        Records = teamRank[2].text.replace("\n", "").replace(",", "")
        Organization = teamRank[3].text.replace("\n", "")
        Method = teamRank[4].text.replace("\n", "")
        Sources = teamRank[5].text.replace("\n", "")
        rows = OrderedDict({"Entity": entity, "Year": year, "Records": Records,
                            "Organization type": Organization, "Method": Method,
                            'Sources': Sources})
        myRows.append(rows)
    except:
        continue

df = pd.DataFrame(myRows)
df.to_csv("Q4_results.csv", index=None)

boxplot_data = df[df['Year'].isin(['2013', '2014', '2015', '2016', '2017'])
                  & df['Method'].isin(['hacked'])]['Records'].values

res = []
for x in boxplot_data:
    try:
        res.append(int(x))
    except:
        continue

plt.figure()
plt.boxplot(res)
plt.show()
# =================================
# 5.1
import pandas as pd
name2 = 'countrydata.csv'
name3 = 'G20.csv'
name5 = 'summer.csv'  # Year,City,Sport,Discipline,Athlete,Country,Gender,Event,Medal
g20 = pd.read_csv(name3).sort_values(by='HDI', ascending=False).head(5)['Member'].values.tolist()
g20 = [i.replace(" ", "") for i in g20]
summer = pd.read_csv(name5)
countrydata = pd.read_csv(name2)
code = countrydata[countrydata['Country'].isin(g20)]['Code'].values.tolist()
medals = summer[summer['Country'].isin(code) & summer['Medal'].isin(['Gold'])]
for x, y in medals.groupby(by='Country'):
    print(x, y.shape[0])

# 5.2

name2 = 'countrydata.csv'
name3 = 'G20.csv'
countrydata = pd.read_csv(name2)
g20 = pd.read_csv(name3)
Member = g20['Member'].values.tolist()
Member = [i.replace(" ", "") for i in Member]
hdi = g20['HDI'].values.tolist()
gdp_pc = countrydata[countrydata['Country'].isin(Member)]['GDP per Capita'].values.tolist()
plt.figure()
plt.scatter(hdi[:14], gdp_pc)
plt.show()


# 5.5
import pandas as pd
from collections import OrderedDict

df1 = pd.read_csv(name2)
df2 = pd.read_csv(name5)


# def mySubset(df1, df2):
# Year, Country, Medal, Population, GDPpc
def mySubset(df1, df2):
    code = df1['Code'].values.tolist()
    data = df2[df2['Country'].isin(code)]
    save = []
    for x, y in data.groupby(by='Country'):
        years = list(set(y['Year'].values.tolist()))
        country = x
        Medal = y.shape[0]
        Population = df1[df1['Code'].isin([x])]['Population'].values.tolist()[0]
        GDPpc = df1[df1['Code'].isin([x])]['GDP per Capita'].values.tolist()[0]
        for year in years:
            rows = OrderedDict({'Year': year, 'Country': country, 'Medal': Medal, 'Population': Population, 'GDPpc': GDPpc})
            print(rows)
            save.append(rows)

    df = pd.DataFrame(save)
    df.to_csv("mysubset.csv", index=None)

mySubset(df1, df2)
