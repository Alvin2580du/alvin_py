import re

salary_pattern = r'\d+\k\-\d+\k'

s = '自然语言处理、语音助手资深算法工程师/专家,20k-40k,美团点评,"20k-40k,经验3-5年,本科","移动互联网,O2O / D轮及以上","'

res = re.compile(salary_pattern).findall(s)
print(res)

