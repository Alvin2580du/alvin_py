import numpy as np
import urllib.request
from bs4 import BeautifulSoup
import csv

url = "http://www.imdb.com/list/ls055592025/"
page = urllib.request.urlopen(url)
soup = BeautifulSoup(page, 'html.parser')
# Fetch movie titles
div_info = soup.select('div.info b a')
titles = [i.text.strip() for i in div_info]
# Fetch titles ends
# Fetch movie descriptions
item_description = soup.find_all('div', attrs={'class': 'item_description'})

description = [i.text.strip() for i in item_description]
print(description[0])
rate = soup.find_all('span', attrs={'class': 'value'})
ratings = [i.text.strip() for i in rate]
# fetch ratings ends
# Text scrapping ends
# zipping and arraying
data = zip(titles, description, ratings)
data = np.array(data)
# zipping and arraying ends
with open('./datasets/records.txt', 'wb') as myfile:
    writer = csv.writer(myfile, delimiter='\t')
    writer.writerows(data)
