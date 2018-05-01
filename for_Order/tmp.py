import pandas as pd
from pyduyp.utils.utils import is_chinese
import re

data = pd.read_csv("D:\\rProject\\reviewscsv.csv")
total = []
for x, y in data.iterrows():
    try:
        # listing_id	id	date	reviewer_id	reviewer_name	comments

        rows = {}
        comments = y['comments']
        listing_id = y['listing_id']
        reviewer_id = y['reviewer_id']
        id = y['id']
        reviewer_name = y['reviewer_name']
        p = "[a-zA-Z]"
        res = []
        for one in comments.split(" "):
            comm = re.compile(p).findall(one)
            comms = "".join(comm)
            res.append(comms)
        if len(res) < 1:
            continue
        rows['comments'] = " ".join(res)
        rows['listing_id'] = listing_id
        rows['reviewer_id'] = reviewer_id
        rows['id'] = id
        rows['reviewer_name'] = reviewer_name
        total.append(rows)
    except:
        continue


df = pd.DataFrame(total)
df.to_csv("newcomments.csv", index=None, encoding='utf-8')


