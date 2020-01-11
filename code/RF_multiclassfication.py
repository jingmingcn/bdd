# -*- coding: utf-8 -*-
"""
Created on Mon Jun 3 2019

@author: Brandon
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from keras.utils import np_utils
from scipy import interp


########################################
#os.chdir(r'D:\Project\Cheeloo-test-project\data')
# os.chdir(r'F:\Cheeloo-test-project\data')

positive = pd.read_csv(r"sample_pos_multi_class_0605.csv")
le = LabelEncoder()
positive['Result'] = le.fit_transform(positive['Result'])
positive['Sex'] = le.fit_transform(positive['Sex'])



X_train, X_test, y_train, y_test = train_test_split(positive.iloc[:, :24], positive.iloc[:, 24], test_size = 0.3, random_state = 0)

## 随机森林

param_test = {'n_estimators':range(20,50,10), 'max_depth':range(10,30,5)}

gsearch2 = GridSearchCV(estimator = RandomForestClassifier(max_features='sqrt' ,oob_score=True, random_state=0),
                        param_grid = param_test, iid=False, cv=5)
gsearch2.fit(X_train, y_train)

print("Best: %f using %s" % (gsearch2.best_score_,gsearch2.best_params_))
#grid_scores_：给出不同参数情况下的评价结果。best_params_：描述了已取得最佳结果的参数的组合
#best_score_：成员提供优化过程期间观察到的最好的评分
#具有键作为列标题和值作为列的dict，可以导入到DataFrame中。
#注意，“params”键用于存储所有参数候选项的参数设置列表。
means = gsearch2.cv_results_['mean_test_score']
params = gsearch2.cv_results_['params']
for mean,param in zip(means,params):
    print("%f  with:   %r" % (mean,param))


clf2 = RandomForestClassifier(n_estimators = 20, max_depth = 15, random_state = 0)
clf2.fit(X_train, y_train)
print(f'Model Accuracy: {clf2.score(X_train, y_train)}')


predict = clf2.predict(X_test)
print(metrics.accuracy_score(predict, y_test))
print(metrics.recall_score(predict, y_test, average = None))
print(metrics.precision_score(predict, y_test, average = None))
print(metrics.f1_score(predict, y_test, average = None))
print(metrics.classification_report(predict, y_test))
conf_matrix = metrics.confusion_matrix(predict, y_test)
print(conf_matrix)


y_test_score = clf2.predict_proba(X_test)
encoder = LabelEncoder()
encoded_Y = encoder.fit_transform(y_test).astype(np.float32)
y_test1 = np_utils.to_categorical(encoded_Y)
fpr = dict()
tpr = dict()
roc_auc = dict()
n_classes = 9
for i in range(n_classes):
    fpr[i], tpr[i], _ = metrics.roc_curve(y_test1[:, i], y_test_score[:, i])
    roc_auc[i] = metrics.auc(fpr[i], tpr[i])

fpr["micro"], tpr["micro"], _ = metrics.roc_curve(y_test1.ravel(), y_test_score.ravel())
roc_auc["micro"] = metrics.auc(fpr["micro"], tpr["micro"])

# Compute macro-average ROC curve and ROC area（方法一）
# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)

for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])
# Finally average it and compute AUC
mean_tpr /= n_classes
fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = metrics.auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
lw=2
plt.figure(dpi = 128)
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)
from itertools import cycle
colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic to multi-class')
plt.legend(loc="best")
plt.show()



scores2 = cross_val_score(clf2, X_train, y_train)
print(scores2.mean())


plt.figure()
lw = 2
plt.figure(figsize=(10,10))
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc) ###假正率为横坐标，真正率为纵坐标做曲线
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()

