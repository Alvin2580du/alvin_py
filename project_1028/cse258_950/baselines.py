import gzip
from collections import defaultdict


def readGz(f):
    for l in gzip.open(f):
        yield eval(l)


### Rating baseline: compute averages for each user, or return the global average if we've never seen the user before

allRatings = []
userRatings = defaultdict(list)
for l in readGz("train.json.gz"):
    user, business = l['reviewerID'], l['itemID']
    allRatings.append(l['rating'])
    userRatings[user].append(l['rating'])

globalAverage = sum(allRatings) / len(allRatings)
userAverage = {}
for u in userRatings:
    userAverage[u] = sum(userRatings[u]) / len(userRatings[u])

predictions = open("predictions_Rating.txt", 'w')
for l in open("pairs_Rating.txt"):
    if l.startswith("reviewerID"):
        # header
        predictions.write(l)
        continue
    u, i = l.strip().split('-')
    if u in userAverage:
        predictions.write(u + '-' + i + ',' + str(userAverage[u]) + '\n')
    else:
        predictions.write(u + '-' + i + ',' + str(globalAverage) + '\n')

predictions.close()

### Would-purchase baseline: just rank which businesses are popular and which are not,
#  and return '1' if a business is among the top-ranked

businessCount = defaultdict(int)
totalPurchases = 0

for l in readGz("train.json.gz"):
    user, business = l['reviewerID'], l['itemID']
    businessCount[business] += 1
    totalPurchases += 1

mostPopular = [(businessCount[x], x) for x in businessCount]
mostPopular.sort()
mostPopular.reverse()

return1 = set()
count = 0
for ic, i in mostPopular:
    count += ic
    return1.add(i)
    if count > totalPurchases / 2: break

predictions = open("predictions_Purchase.txt", 'w')
for l in open("pairs_Purchase.txt"):
    if l.startswith("reviewerID"):
        # header
        predictions.write(l)
        continue
    u, i = l.strip().split('-')
    if i in return1:
        predictions.write(u + '-' + i + ",1\n")
    else:
        predictions.write(u + '-' + i + ",0\n")

predictions.close()

### Category prediction baseline: Just consider some of the most common words from each category

catDict = {
    "Women": 0,
    "Men": 1,
    "Girls": 2,
    "Boys": 3,
    "Baby": 4
}

predictions = open("predictions_Category.txt", 'w')
predictions.write("reviewerID-reviewHash,category\n")
for l in readGz("test_Category.json.gz"):
    cat = catDict['Women']  # If there's no evidence, just choose the most common category in the dataset
    words = l['reviewText'].lower()
    if 'wife' in words:
        cat = catDict['Women']
    if 'husband' in words:
        cat = catDict['Men']
    if 'daughter' in words:
        cat = catDict['Girls']
    if 'son' in words:
        cat = catDict['Boys']
    if 'baby' in words:
        cat = catDict['Baby']
    predictions.write(l['reviewerID'] + '-' + l['reviewHash'] + "," + str(cat) + "\n")

predictions.close()
