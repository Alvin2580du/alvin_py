## 导入安装的包
import recordlinkage
import pandas as pd
import os
from saveLinkResult import save_linkage_set


# -----------------------------------------------------------------------------

def evalution(X_data, links_true):
    # 这里用逻辑回归分类器做分类，
    cl = recordlinkage.LogisticRegressionClassifier()
    cl.fit(X_data, links_true)
    # 用得到的模型做预测
    links_pred = cl.predict(X_data)
    print("links_pred:{}".format(links_pred.shape))
    # 输出混淆矩阵，confusion_matrix
    cm = recordlinkage.confusion_matrix(links_true, links_pred, total=len(X_data))
    print("Confusion matrix:\n", cm)
    # compute the F-score for this classification
    fscore = recordlinkage.fscore(cm)
    print('fscore', fscore)
    # compute recall for this classification
    recall = recordlinkage.recall(links_true, links_pred)
    print('recall', recall)
    # compute precision for this classification
    precision = recordlinkage.precision(links_true, links_pred)
    print('precision', precision)


def build(m):
    ## 读取数据集AB
    dfA = pd.read_csv(R"dataset-A.csv", index_col=0, encoding="utf_8")
    dfB = pd.read_csv(R"dataset-B.csv", index_col=0, encoding="utf_8")
    links_true = pd.read_csv("true-matches.csv", header=None)
    arrays = [links_true[0].values.tolist(), links_true[1].values.tolist()]
    links_true = pd.MultiIndex.from_arrays(arrays)
    # Indexation step
    indexer = recordlinkage.Index()
    ## 这里应该就是你的文档中的block,我用了下面的2个，可以改
    # rec_id,first_name,middle_name,last_name,gender,current_age,birth_date,street_address,suburb,postcode,state,phone,email

    indexer.block(("first_name", "last_name"))
    candidate_links = indexer.index(dfA, dfB)
    print("candidate_links:{}".format(candidate_links))
    """
    # 1  blocking:
            simpleBlocking, phoneticBlocking, slkBlocking, 可以选择simpleBlocking。
            	choice	of blocking	keys： 
    """
    # Comparison step
    compare_cl = recordlinkage.Compare()
    # 下面是匹配的条件，具体有哪些参考文档中第4点
    compare_cl.string('first_name', 'first_name', method=m, threshold=0.85, label='first_name')
    compare_cl.string('middle_name', 'middle_name', method=m, threshold=0.85, label='middle_name')
    compare_cl.string('last_name', 'last_name', label='last_name', method=m, threshold=0.85)
    compare_cl.string('street_address', 'street_address', label='street_address', method=m, threshold=0.85)
    compare_cl.exact('gender', 'gender', label='gender')
    compare_cl.string('birth_date', 'birth_date', label='birth_date', method=m, threshold=0.85)

    if not os.path.exists("./results"):
        os.makedirs("./results")

    features = compare_cl.compute(candidate_links, dfA, dfB)
    features.to_csv("./results/features_{}.csv".format(m))
    # 选择准确匹配的个数，满足阈值的决策为匹配成功的，
    num = 3
    matches = features[features.sum(axis=1) >= num]

    # 输出方法，直接调用文件夹中的代码，输出文件是out.csv
    save_linkage_set("./results/out_{}.csv".format(m), matches.index)
    evalution(X_data=matches, links_true=links_true)


for m in ['jaro', 'jarowinkler', 'levenshtein', 'damerau_levenshtein', 'qgram', 'cosine']:
    build(m)
    print("----------------{}-------------------".format(m))

