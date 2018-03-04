from bs4 import BeautifulSoup
from tqdm import trange
import re
from pyduyp.utils.spyder_helper import urlhelper
from dateutil import parser
from selenium import webdriver
from selenium.webdriver import ActionChains
import time

driver = webdriver.Chrome()


# browser = webdriver.Firefox()


def str2datetime(inputs):
    datetime_struct = parser.parse(inputs)
    res = datetime_struct.strftime('%Y-%m-%d %H:%M:%S')
    return res


def bs4_spyder():
    fw = open("./datasets/cailianshe/cailianshe.txt", 'a+', encoding="utf-8")
    rooturl = 'https://www.cailianpress.com/'
    html = urlhelper(rooturl)
    soup = BeautifulSoup(html, "lxml")

    date = soup.find_all('div', attrs={'class': 'time'})[0].text

    newsRight = soup.find_all('div', attrs={'class': 'newsRight'})
    newsLeft = soup.find_all('div', attrs={'class': 'newsLeft'})

    for i in trange(len(newsRight)):
        # content = newsRight[i].text
        # print(type(content))
        ctime = newsLeft[i].findAll('div', attrs={'class': 'cTime'})[0].text

        position_link = newsRight[i].findAll('a', attrs={'target': '_blank'})
        link = position_link[0]['href']
        pattern = r'roll/\d+'
        link = re.compile(pattern).findall(link)
        newsLink = rooturl + link[0]
        newshtml = urlhelper(newsLink)
        newssoup = BeautifulSoup(newshtml, "lxml")
        content = newssoup.findAll('div', attrs={'class': 'thisContent'})[0].text

        save_time = str2datetime("{} {}".format(date, ctime))

        save_dict = {"news": content, "createtime": save_time, "comment": 0}
        writes = "{},{},{}".format(content, save_time, 0)
        fw.writelines(writes + "\n")


def execute_times(times):
    for i in range(times + 1):
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(5)


def dontai():
    rooturl = 'https://www.cailianpress.com/'
    driver.get(rooturl)
    loadmore = driver.find_element_by_class_name(name='getMore')
    actions = ActionChains(driver)
    actions.move_to_element(loadmore)
    actions.click(loadmore)
    actions.perform()
    html = driver.page_source
    soup = BeautifulSoup(html, 'lxml')
    newsRight = soup.find_all('div', attrs={'class': 'newsRight'})
    print(len(newsRight), newsRight)


dontai()
