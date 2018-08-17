import xlrd
from xlutils.copy import copy
from selenium import webdriver


def getdata(id):
    datas = []
    opt = webdriver.ChromeOptions()
    opt.set_headless()
    driver = webdriver.Chrome(options=opt)

    driver.get('https://www.bitstamp.net/ajax/news/?start={0}&limit=10'.format(id))
    driver.refresh()
    time = driver.find_elements_by_css_selector('body > article > a > hgroup > h3 > time')
    title = driver.find_elements_by_css_selector('body > article > a > section > h1 ')
    content = driver.find_elements_by_css_selector('body > article > a > section > div')
    # print(time)
    for a in time:
        data = {
            'time': a.text,
            'title': title[time.index(a)].text.replace('\r', '').replace('\n', ''),
            'content': content[time.index(a)].text.replace('\r', '').replace('\n', '').replace('*\t', ''),
        }
        datas.append(data)
    return datas


def write2(datas):
    col = 0
    rb = xlrd.open_workbook('data.xls')
    # 通过sheet_by_index()获取的sheet没有write()方法
    rs = rb.sheet_by_index(0)
    row = rs.nrows
    wb = copy(rb)
    # 通过get_sheet()获取的sheet有write()方法
    ws = wb.get_sheet(0)
    for data in datas:
        ws.write(row, col, data['time'])
        ws.write(row, col + 1, data['title'])
        ws.write(row, col + 2, data['content'])
        row += 1
    wb.save('data.xls')


if __name__ == '__main__':
    for i in range(0, 43):
        print('开始爬取第' + str(i + 1) + '页')
        datas = getdata(i * 10)
        print(datas)
        write2(datas)
        print('爬取第' + str(i + 1) + '成功')
