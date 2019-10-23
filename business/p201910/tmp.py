from textblob import TextBlob
import pandas as pd

text = "I am happy today. I feel sad today."
blob = TextBlob(text)
# 第一句的情感分析
first = blob.sentences.sentiment



