from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from time import sleep
import os, time
from urllib.parse import quote, unquote

time_btn = {}

dr = webdriver.Chrome()
dr.set_window_size(1280, 900)

# 输入需要查找的词
keyword = input('输入需要查找的词: ')
code_keyword = quote(keyword.encode('gb2312'))

# 需要请求的页面
url = 'https://index.baidu.com/?tpl=trend&word={0}'.format(code_keyword)
print(url)
dr.get("http://www.baidu.com")

# 模拟登陆
dr.delete_all_cookies()
c1 = "8CEE86D1614FB5CD1399BA44C0B67BF7:FG=1"
c2 = "R3UlFDVk9tLWRPcTAwRlptYTNyb0NLa2RWRDZBNTFzOU5DUHlqfkxCRjJKck5iQVFBQUFBJCQAAAAAAAAAAAEAAAAHvnFzxMfE6rfn0-oyNTgwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAHaZi1t2mYtbNk"
dr.add_cookie({'name': 'BAIDUID', 'value': c1})
dr.add_cookie({'name': 'BDUSS', 'value': c2})
dr.get(url)

svg = dr.find_element_by_tag_name("svg")
svg.screenshot(keyword + int(time.time()) + ".png")
input("截图完成")
dr.quit()
