import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from xgboost import XGBClassifier


# 没事！周二之前就行！
def data_loader():
    train_data = pd.read_csv("pzo_training_data.csv")
    test2valid_data = pd.read_csv("pzo_test_data.csv")

    train_y = train_data['target']
    del train_data['target']
    train_x = train_data
    test_x = test2valid_data[test2valid_data['data_type'].isin(['test'])]
    del test_x['target']
    valid_x = test2valid_data[test2valid_data['data_type'].isin(['validation'])]
    valid_y = valid_x['target']
    del valid_x['target']
    del test_x['data_type']
    del valid_x['data_type']
    return train_x, train_y, valid_x, valid_y, test_x


train_x, train_y, valid_x, valid_y, test_x = data_loader()

print(train_x.shape, train_y.shape, valid_x.shape, valid_y.shape, test_x.shape)


def logistic():
    # build logistic regression model
    log_clf = LogisticRegression(penalty='l2', dual=False, tol=1e-9, C=0.1,
                                 fit_intercept=True, intercept_scaling=1, class_weight=None,
                                 random_state=1, solver='liblinear', max_iter=10000,
                                 multi_class='warn', verbose=False, warm_start=False, n_jobs=None)

    log_clf.fit(train_x, train_y)

    y_prediction = log_clf.predict_proba(valid_x)
    probabilities = y_prediction[:, 1]

    correct = [round(x) == y for (x, y) in zip(probabilities, valid_y)]
    print("Logistic Regression accuracy: ", sum(correct) / float(valid_x.shape[0]))

    log_loss = metrics.log_loss(valid_y, probabilities)
    print("Logistic Regression validation logloss:", log_loss)

    y_prediction = log_clf.predict(test_x)

    results_df = pd.DataFrame(data={'probability': y_prediction})
    results_df.to_csv("log_clf_submission.csv", index=False)


def gbtree():
    # build GradientBoostingClassifier  model
    model = XGBClassifier(learning_rate=0.1, n_estimators=10000, max_depth=6, min_child_weight=1,
                          gamma=0., subsample=0.8, colsample_btree=0.8, objective='multi:softmax',
                          scale_pos_weight=1, random_state=27, num_class=2)

    model.fit(train_x, train_y, eval_set=[(valid_x, valid_y)], eval_metric="mlogloss",
              early_stopping_rounds=10, verbose=False)

    y_pred = model.predict(valid_x)

    ### model evaluate
    accuracy = metrics.accuracy_score(valid_y, y_pred)
    print("xgboost accuarcy: %.2f%%" % (accuracy * 100.0))

    log_loss = metrics.log_loss(valid_y, y_pred)
    print("GradientBoostingClassifier validation logloss:", log_loss)

    y_prediction = model.predict_proba(test_x)
    results = y_prediction[:, 1]
    print(results)
    results_df = pd.DataFrame(data={'probability': results})
    results_df.to_csv("gb_submission.csv", index=False)


logistic()
gbtree()
