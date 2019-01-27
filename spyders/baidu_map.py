import urllib.request

img_url = 'https://imgsa.baidu.com/forum/w%3D580/sign=6eb1809e16ce36d3a20483380af33a24/09a3cfd3572c11df9ce8c367602762d0f703c237.jpg'

urllib.request.urlretrieve(img_url, filename='test.jpg')
