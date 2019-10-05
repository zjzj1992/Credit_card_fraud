#!/usr/bin/env python
# coding: utf-8

# In[224]:


import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")


# * 这个数据是从Kaggle获取的，因为这类数据都算是涉密数据，所以下载的数据并不是原始数据

# In[225]:


data = pd.read_csv("C:/Users/zhang/Desktop/creditcardfraud/creditcard.csv")


# In[226]:


len(data)


# * 数据一共有284807条

# In[227]:


data.columns


# In[228]:


len(data.columns)


# In[229]:


set(data.Class)


# * 上面是数据中各列的特征名称，一共有31个特征，但是由于隐私的原因，并没有公开显示真实的含义是什么，而是只通过V+数字的形式来表示
# * 数据中对应的目标变量是Class，只有两种结果，0和1，其中1对应的是欺诈记录，而0对应的是非欺诈记录

# 下面统计了各特征值的缺失情况

# In[230]:


(data.isnull().sum())/len(data)


# * 不存在缺失的情况

# 下面先观察每个特征的取值范围，确实是否需要进行归一化的处理

# In[231]:


data.describe()


# * 从上面的结果可以看出，特征从V1-V28的取值范围都是比较相近的，且取值都很小，但是Time和Amount这两个特征对应的值与其他特征值的差距就很大，所以应该对其进行归一化的处理，并且归一化也可以很好的防止异常值的出现

# In[232]:


#不受异常值的影响的一种归一化方法
from sklearn.preprocessing import RobustScaler

#标准化处理
rs_scaler = RobustScaler()
data['scaler_amount'] = rs_scaler.fit_transform(data['Amount'].values.reshape(-1,1))
data['scaler_time'] = rs_scaler.fit_transform(data['Time'].values.reshape(-1,1))


# In[233]:


#删除原有的Amount和Time
data.drop(['Amount','Time'],axis=1,inplace=True)


# In[234]:


print("欺诈比例：",round(sum(data.Class == 1)/len(data) * 100,2),"%")
print("非欺诈比例：",round(sum(data.Class == 0)/len(data) * 100,2),"%")


# * 非欺诈记录占99.83%，而欺诈记录只占0.17%，说明这是一个非常不平衡的数据，两类数据的数量差距非常大
# * 我的目的是训练出一个可以识别欺诈交易的模型，但是这个数据中有很多的非欺诈数据，所以模型会更关注非欺诈数据，那么在识别欺诈数据时就不会准确了
# * 所以如何处理不平衡数据就显得非常重要了

# In[235]:


from sklearn.model_selection import StratifiedShuffleSplit


# In[236]:


X = data.drop('Class',axis=1)
y = data['Class']

#获取数据
s = StratifiedShuffleSplit(n_splits=5,random_state=None)

for train_index,test_index in s.split(X,y):
    X_train,X_test = X.iloc[train_index],X.iloc[test_index]
    y_train,y_test = y.iloc[train_index],y.iloc[test_index]
    
X_train = X_train.values
X_test = X_test.values
y_train = y_train.values
y_test = y_test.values


# 下面使用欠采样的方式获取数据

# In[144]:


#随机打乱
data = data.sample(frac=1)
fraud_data = data.loc[data.Class == 1]
non_fraud_data = data.loc[data.Class == 0][:492]

#组成出平衡数据
new_data = pd.concat([fraud_data,non_fraud_data])
new_data = new_data.sample(frac=1,random_state=42)


# In[145]:


print(new_data['Class'].value_counts()/len(new_data))


# * 上面是使用了一种最普通的欠采样的方式来获取一个平衡数据，这种方式是删除多余的数据来达到平衡
# * 重新采样之后的数据符合了平衡数据的标准
# * 现在有了一个平衡数据后就可以将重心放在选择合适的特征上了
# * 为了构建一个可以预测欺诈行为的模型，应该筛选一下哪些特征与欺诈行为之间的相关性更高

# 下面通过箱型图的方式找出哪些特征与目标变量之间是有关系的

# In[146]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[147]:


f,axes = plt.subplots(ncols=4,figsize=(20,4))

sns.boxplot(x='Class',y='V1',data=new_data,ax=axes[0])
sns.boxplot(x='Class',y='V2',data=new_data,ax=axes[1])
sns.boxplot(x='Class',y='V3',data=new_data,ax=axes[2])
sns.boxplot(x='Class',y='V4',data=new_data,ax=axes[3])


# * 从上面这四个特征可以看出，当V1和V3的值越低时，则越有可能是欺诈行为；而V2和V4的值是越高则越有可能是欺诈行为

# In[148]:


f,axes = plt.subplots(ncols=4,figsize=(20,4))

sns.boxplot(x='Class',y='V5',data=new_data,ax=axes[0])
sns.boxplot(x='Class',y='V6',data=new_data,ax=axes[1])
sns.boxplot(x='Class',y='V7',data=new_data,ax=axes[2])
sns.boxplot(x='Class',y='V8',data=new_data,ax=axes[3])


# * 从上面这四个特征可以看出，当V5、V6和V7的值越低时，则越有可能是欺诈行为；而V8与欺诈行为的关系不明显

# In[149]:


f,axes = plt.subplots(ncols=4,figsize=(20,4))

sns.boxplot(x='Class',y='V9',data=new_data,ax=axes[0])
sns.boxplot(x='Class',y='V10',data=new_data,ax=axes[1])
sns.boxplot(x='Class',y='V11',data=new_data,ax=axes[2])
sns.boxplot(x='Class',y='V12',data=new_data,ax=axes[3])


# * 从上面这四个特征可以看出，当V9、V10和V12的值越低时，则越有可能是欺诈行为；而V11的值是越高则越有可能是欺诈行为

# In[150]:


f,axes = plt.subplots(ncols=4,figsize=(20,4))

sns.boxplot(x='Class',y='V13',data=new_data,ax=axes[0])
sns.boxplot(x='Class',y='V14',data=new_data,ax=axes[1])
sns.boxplot(x='Class',y='V15',data=new_data,ax=axes[2])
sns.boxplot(x='Class',y='V16',data=new_data,ax=axes[3])


# * 从上面这四个特征可以看出，当V14和V16的值越低时，则越有可能是欺诈行为；而V13和V15与欺诈行为关系不明显

# In[151]:


f,axes = plt.subplots(ncols=4,figsize=(20,4))

sns.boxplot(x='Class',y='V17',data=new_data,ax=axes[0])
sns.boxplot(x='Class',y='V18',data=new_data,ax=axes[1])
sns.boxplot(x='Class',y='V19',data=new_data,ax=axes[2])
sns.boxplot(x='Class',y='V20',data=new_data,ax=axes[3])


# * 从上面这四个特征可以看出，当V17和V18的值越低时，则越有可能是欺诈行为；而V19的值是越高则越有可能是欺诈行为，而V20与欺诈行为关系不明显

# In[152]:


f,axes = plt.subplots(ncols=4,figsize=(20,4))

sns.boxplot(x='Class',y='V21',data=new_data,ax=axes[0])
sns.boxplot(x='Class',y='V22',data=new_data,ax=axes[1])
sns.boxplot(x='Class',y='V23',data=new_data,ax=axes[2])
sns.boxplot(x='Class',y='V24',data=new_data,ax=axes[3])


# * 上面这四个特征与欺诈行为的关系都不明显

# In[153]:


f,axes = plt.subplots(ncols=4,figsize=(20,4))

sns.boxplot(x='Class',y='V25',data=new_data,ax=axes[0])
sns.boxplot(x='Class',y='V26',data=new_data,ax=axes[1])
sns.boxplot(x='Class',y='V27',data=new_data,ax=axes[2])
sns.boxplot(x='Class',y='V28',data=new_data,ax=axes[3])


# * 上面这四个特征与欺诈行为的关系依然不明显

# In[154]:


f,axes = plt.subplots(ncols=2,figsize=(20,4))

sns.boxplot(x='Class',y='scaler_amount',data=new_data,ax=axes[0])
sns.boxplot(x='Class',y='scaler_time',data=new_data,ax=axes[1])


# * 上面这两个特征对于判断是否是欺诈行为也是不明显的

# * 所以，与欺诈相关性比较高的特征有V1、V2、V3、V4、V5、V6、V7、V9、V10、V11、V12、V14、V16、V17、V18和V19这16个特征

# * 从上面可视化的结果可以看出，大多数特征的箱型图中都可以看出是含有异常值的，所以下面要先定义一个规则来将异常值去掉，否则异常值会影响模型的准确性

# 下面针对上面提到的16个特征进行异常值的剔除

# * 对异常值的惩罚系数设定为1.5，这个值越大则对异常值的惩罚力度就越大

# In[155]:


from scipy.stats import norm


# In[156]:


#定义异常值移除函数
def remove_outliers(column,data):
    v = data[column].loc[data['Class'] == 1].values
    q25 = np.percentile(v,25)
    q75 = np.percentile(v,75)
    
    v_lower = q25 - (q75 - q25) * 1.5
    v_upper = q75 + (q75 - q25) * 1.5
    
    new_data = data.drop(data.loc[(data[column] < v_lower) | (data[column] > v_upper)].index)
    return new_data


# In[157]:


#分布图
def draw(data,column):
    f,ax = plt.subplots(1,figsize=(5,5))
    v = data[column].loc[data['Class'] == 1].values
    sns.distplot(v,ax=ax,fit=norm)
    ax.set_title(column)


# V1

# In[158]:


draw(new_data,'V1')


# * 上图可以看出V1有严重的偏移，很不符合正态分布，有很多偏小的异常值出现

# In[159]:


new_data = remove_outliers('V1',new_data)


# In[160]:


draw(new_data,'V1')


# * 剔除异常值以后，数据更规整，更符合正态分布

# V2

# In[161]:


draw(new_data,'V2')


# * V2两端都有异常情况出现

# In[162]:


new_data = remove_outliers('V2',new_data)


# In[163]:


draw(new_data,'V2')


# V3

# In[164]:


draw(new_data,'V3')


# * V3中有很多过小的异常值

# In[165]:


new_data = remove_outliers('V3',new_data)


# In[166]:


draw(new_data,'V3')


# V4

# In[167]:


draw(new_data,'V4')


# * 从箱型图和分布图来看，这个特征的异常值并不明显，所以可以不进行处理

# V5

# In[168]:


draw(new_data,'V5')


# * V5特征存在异常值

# In[169]:


new_data = remove_outliers('V5',new_data)


# V6

# In[170]:


draw(new_data,'V6')


# * V6存在异常值

# In[171]:


new_data = remove_outliers('V6',new_data)


# V7

# In[172]:


draw(new_data,'V7')


# * V7存在异常值

# In[173]:


new_data = remove_outliers('V7',new_data)


# V9

# In[174]:


draw(new_data,'V9')


# * V9存在异常值

# In[175]:


new_data = remove_outliers('V9',new_data)


# V10

# In[176]:


draw(new_data,'V10')


# * V10存在异常值

# In[177]:


new_data = remove_outliers('V10',new_data)


# V11

# In[178]:


draw(new_data,'V11')


# In[179]:


new_data = remove_outliers('V11',new_data)


# V12

# In[180]:


draw(new_data,'V12')


# In[181]:


new_data = remove_outliers('V12',new_data)


# V13

# In[182]:


draw(new_data,'V13')


# In[183]:


new_data = remove_outliers('V13',new_data)


# V14

# In[184]:


draw(new_data,'V14')


# In[185]:


new_data = remove_outliers('V14',new_data)


# V16

# In[186]:


draw(new_data,'V16')


# In[187]:


new_data = remove_outliers('V16',new_data)


# V17

# In[188]:


draw(new_data,'V17')


# In[189]:


new_data = remove_outliers('V17',new_data)


# V18

# In[190]:


draw(new_data,'V18')


# In[191]:


new_data = remove_outliers('V18',new_data)


# V19

# In[192]:


draw(new_data,'V19')


# In[193]:


new_data = remove_outliers('V19',new_data)


# * 到这里所有特征的异常值就处理结束了，但是对于异常值的惩罚系数还可以根据后面的结果进行调整

# In[74]:


from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score


# In[75]:


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


# In[198]:


X = new_data.drop('Class',axis=1)
y = new_data['Class']


# In[200]:


new_X_train,new_X_test,new_y_train,new_y_test = train_test_split(X, y, test_size=0.2, random_state=42)

new_X_train = new_X_train.values
new_X_test = new_X_test.values
new_y_train = new_y_train.values
new_y_test = new_y_test.values


# In[205]:


classifier = {"LR":LogisticRegression(),
             "KNN":KNeighborsClassifier(),
             "SVC":SVC(),
             "DTC":DecisionTreeClassifier()
             }


# In[206]:


from sklearn.model_selection import cross_val_score


for key,classifier in classifier.items():
    classifier.fit(new_X_train,new_y_train)
    training_score = cross_val_score(classifier,new_X_train,new_y_train, cv=5)
    print("Classifiers: ", classifier.__class__.__name__,round(training_score.mean(), 2) * 100, "% accuracy score")


# In[207]:


from sklearn.model_selection import GridSearchCV


# In[208]:


#lr_gird
lr_params = {"penalty":['l1','l2'],'C':[0.001,0.01,0.1,1,10,100,1000]}
grid_lr = GridSearchCV(LogisticRegression(),lr_params)
grid_lr.fit(new_X_train,new_y_train)
log_lr = grid_lr.best_estimator_
lr_score = cross_val_score(log_lr,new_X_train,new_y_train,cv=5)
print("LR accuracy:",round(lr_score.mean() * 100,2).astype(str)+"%")

#knn_gird
knn_params = {"n_neighbors":list(range(2,5,1)),'algorithm':['auto','ball_tree','kd_tree','brute']}
grid_knn = GridSearchCV(KNeighborsClassifier(),knn_params)
grid_knn.fit(new_X_train,new_y_train)
log_knn = grid_knn.best_estimator_
knn_score = cross_val_score(log_knn,new_X_train,new_y_train,cv=5)
print("KNN Accuracy:",round(knn_score.mean() * 100,2).astype(str) + '%')

#svc_grid
svc_params = {'C':[0.5,0.7,0.9,1],'kernel':['rbf','poly','sigmoid','linear']}
grid_svc = GridSearchCV(SVC(),svc_params)
grid_svc.fit(new_X_train,new_y_train)
log_svc = grid_svc.best_estimator_
svc_socre = cross_val_score(log_svc,new_X_train,new_y_train,cv=5)
print("SVC Accuracy:",round(svc_socre.mean() * 100,2).astype(str)+'%')

#tree_grid
tree_params = {"criterion":['gini','entropy'],"max_depth":list(range(2,4,1)),"min_samples_leaf":list(range(5,7,1))}
grid_tree = GridSearchCV(DecisionTreeClassifier(),tree_params)
grid_tree.fit(new_X_train,new_y_train)
log_tree = grid_tree.best_estimator_
tree_socre = cross_val_score(log_tree,new_X_train,new_y_train,cv=5)
print("Tree Accuracy:",round(tree_socre.mean() * 100,2).astype(str)+'%')


# * 上面准确率最高的是SVC模型，而LR模型的准确率也很高

# 下面导入两种处理不平衡数据的方法

# In[210]:


from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss
from imblearn.pipeline import make_pipeline
from sklearn.metrics import precision_score,recall_score,f1_score,roc_auc_score,accuracy_score,classification_report


# 下面使用NearMiss对数据进行重新采样

# In[238]:


undersample_X = data.drop('Class',axis=1)
undersample_y = data['Class']

for train_index,test_index in s.split(undersample_X,undersample_y):
    undersample_X_train,undersample_X_test = undersample_X.iloc[train_index],undersample_X.iloc[test_index]
    undersample_y_train,undersample_y_test = undersample_y.iloc[train_index],undersample_y.iloc[test_index]

undersample_X_train = undersample_X_train.values
undersample_X_test = undersample_X_test.values
undersample_y_train = undersample_y_train.values
undersample_y_test = undersample_y_test.values

undersample_acc = []
undersample_pre = []
undersample_recall = []
undersample_f1 = []
undersample_auc = []

#输出KNN模型对应的各项指标
for train_index,test_index in s.split(undersample_X_train,undersample_y_train):
    undersample_pipeline = make_pipeline(NearMiss(sampling_strategy='majority'),log_lr)
    undersample_m = undersample_pipeline.fit(undersample_X_train[train_index],undersample_y_train[train_index])
    undersample_prediction = undersample_m.predict(undersample_X_train[test_index])
    undersample_acc.append(undersample_pipeline.score(X_train[test_index],y_train[test_index]))
    undersample_pre.append(precision_score(y_train[test_index],undersample_prediction))
    undersample_recall.append(recall_score(y_train[test_index],undersample_prediction))
    undersample_f1.append(f1_score(y_train[test_index],undersample_prediction))
    undersample_auc.append(roc_auc_score(y_train[test_index],undersample_prediction))


# 使用交叉验证的方式进行预测

# In[239]:


from sklearn.metrics import roc_curve
from sklearn.model_selection import cross_val_predict


# In[240]:


#使用原始数据来评估模型的好坏
lr_predct = cross_val_predict(log_lr,new_X_train,new_y_train,cv=5,method='decision_function')

knn_predict = cross_val_predict(log_knn,new_X_train,new_y_train,cv=5)

svc_predict = cross_val_predict(log_svc,new_X_train,new_y_train,cv=5,method='decision_function')

tree_predict = cross_val_predict(log_tree,new_X_train,new_y_train,cv=5)


# 输出AUC面积，越大越好

# In[241]:


from sklearn.metrics import roc_auc_score

print("lr roc_auc:",roc_auc_score(new_y_train,lr_predct))
print("knn roc_auc:",roc_auc_score(new_y_train,knn_predict))
print("svc roc_auc:",roc_auc_score(new_y_train,svc_predict))
print("tree roc_auc:",roc_auc_score(new_y_train,tree_predict))


# * 根据AUC面积来看，分类效果最好的是LR模型

# 使用ROC评估模型

# In[242]:


#计算ROC曲线的相关参数
lr_fpr,lr_tpr,lr_thresold = roc_curve(new_y_train,lr_predct)
knn_fpr,knn_tpr,knn_thresold = roc_curve(new_y_train,knn_predict)
svc_fpr,svc_tpr,svc_thresold = roc_curve(new_y_train,svc_predict)
tree_fpr,tree_tpr,tree_thresold = roc_curve(new_y_train,tree_predict)


# In[243]:


plt.figure(figsize=(16,8))
plt.title("ROC")
plt.plot(lr_fpr,lr_tpr,label='lr score:{:.4f}'.format(roc_auc_score(new_y_train,lr_predct)))
plt.plot(knn_fpr,knn_tpr,label='knn score:{:.4f}'.format(roc_auc_score(new_y_train,knn_predict)))
plt.plot(svc_fpr,svc_tpr,label='svc score:{:.4f}'.format(roc_auc_score(new_y_train,svc_predict)))
plt.plot(tree_fpr,tree_tpr,label='tree score:{:.4f}'.format(roc_auc_score(new_y_train,tree_predict)))
plt.plot([0, 1], [0, 1], 'k--')
plt.axis([-0.01, 1, 0, 1])
plt.legend()


# * 上面是不同模型对应的ROC曲线，这个曲线越接近左上角则代表就越好，而上面最好的是LR模型

# 使用精确率、召回率和F1指标评估LR模型

# In[244]:


from sklearn.metrics import precision_recall_curve

precision,recall,threshold = precision_recall_curve(new_y_train,lr_predct)


# In[245]:


from sklearn.metrics import recall_score,precision_score,f1_score,accuracy_score
y_pred = log_lr.predict(new_X_test)

print('Recall Score: {:.2f}'.format(recall_score(new_y_test, y_pred)))
print('Precision Score: {:.2f}'.format(precision_score(new_y_test, y_pred)))
print('F1 Score: {:.2f}'.format(f1_score(new_y_test, y_pred)))
print('Accuracy Score: {:.2f}'.format(accuracy_score(new_y_test, y_pred)))

print('---' * 45)

print("Recall Average Score: {:.2f}".format(np.mean(undersample_recall)))
print("Precision Average Score: {:.2f}".format(np.mean(undersample_pre)))
print("F1 Average Score: {:.2f}".format(np.mean(undersample_f1)))
print("Accuracy Average Score: {:.2f}".format(np.mean(undersample_acc)))
print('---' * 45)


# In[246]:


undersample_y_score = log_lr.decision_function(X_test)

from sklearn.metrics import average_precision_score

#平均精确率
undersample_average_pre = average_precision_score(y_test,undersample_y_score)
print('Average precision-recall score: {0:0.2f}'.format(undersample_average_pre))


# In[247]:


p,r,_ = precision_recall_curve(y_test,undersample_y_score)
fig = plt.figure(figsize=(12,6))

plt.step(r,p,color='r',alpha=0.2,where='post')
plt.fill_between(r,p,step='post',alpha=0.2,color='#F59B00')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Average Precision-Recall Score ={0:0.2f}'.format(undersample_average_pre), fontsize=16)
plt.show()


# 上面是使用NearMiss处理过的数据，但是效果并不理想

# 下面使用SMOTE方法

# In[248]:


from sklearn.model_selection import RandomizedSearchCV


# In[252]:


acc_list = []
pre_list = []
recall_list = []
f1_list = []
auc_list = []

lr_sm = LogisticRegression()

lr_sm_params = {'penalty':['l1','l2'],'C':[0.001,0.01,0.1,1,10,100,1000]}
rand_lr_sm = RandomizedSearchCV(LogisticRegression(),lr_sm_params,n_iter=4)

for train_index,test_index in s.split(X_train,y_train):
    pp = make_pipeline(SMOTE(sampling_strategy='minority'),rand_lr_sm)
    model = pp.fit(X_train[train_index],y_train[train_index])
    best = rand_lr_sm.best_estimator_
    predict = best.predict(X_train[test_index])
    
    acc_list.append(pp.score(X_train[test_index],y_train[test_index]))
    pre_list.append(precision_score(y_train[test_index],predict))
    recall_list.append(recall_score(y_train[test_index],predict))
    f1_list.append(f1_score(y_train[test_index],predict))
    auc_list.append(roc_auc_score(y_train[test_index],predict))
    
print("accuracy: {}".format(np.mean(acc_list)))
print("precision: {}".format(np.mean(pre_list)))
print("recall: {}".format(np.mean(recall_list)))
print("f1: {}".format(np.mean(f1_list)))


# In[253]:


labels = ['No Fraud', 'Fraud']
smote_prediction = best.predict(X_test)
print(classification_report(y_test, smote_prediction,target_names=labels))


# In[254]:


y_score = best.decision_function(X_test)


# In[255]:


from sklearn.metrics import average_precision_score

ave_pre = average_precision_score(y_test,y_score)
print("ave_pre_recall score:{:.2f}".format(ave_pre))


# In[256]:


p,r,_ = precision_recall_curve(y_test,y_score)
fig = plt.figure(figsize=(12,6))

plt.step(r,p,color='r',alpha=0.2,where='post')
plt.fill_between(r,p,step='post',alpha=0.2,color='#F59B00')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Average Precision-Recall Score ={0:0.2f}'.format(ave_pre), fontsize=16)


# 上面较NearMiss方法来说，平均精确率有所提升

# 下面使用SMOTE处理原始数据

# In[257]:


sm = SMOTE(ratio='minority',random_state=42)

X_sm_train,y_sm_train = sm.fit_sample(X_train,y_train)


# In[258]:


log_sm = grid_lr.best_estimator_
log_sm.fit(X_sm_train,y_sm_train)


# In[259]:


from sklearn.metrics import confusion_matrix

#sm
y_pred_sm = log_sm.predict(new_X_test)

#knn
y_pred_knn = log_knn.predict(new_X_test)

#svc
y_pred_svc = log_svc.predict(new_X_test)

#tree
y_pred_tree = log_tree.predict(new_X_test)


# In[260]:


print('LR:')
print(classification_report(new_y_test, y_pred_sm))
print("-----------------------------------------------------")

print('KNN:')
print(classification_report(new_y_test, y_pred_knn))
print("-----------------------------------------------------")

print('SVC:')
print(classification_report(new_y_test, y_pred_svc))
print("-----------------------------------------------------")

print('TREE:')
print(classification_report(new_y_test, y_pred_tree))


# 这个模型是用来预测欺诈行为的，所以应该更关注欺诈行为的预测效果，所以从上面的各个模型的表现来看，LR模型在预测欺诈行为的准确率是最高的

# In[261]:


# Final Score in the test set of logistic regression
from sklearn.metrics import accuracy_score

# Logistic Regression with Under-Sampling
y_pred = log_sm.predict(new_X_test)
undersample_score = accuracy_score(new_y_test, y_pred)


# Logistic Regression with SMOTE Technique (Better accuracy with SMOTE t)
y_pred_sm = best.predict(X_test)
oversample_score = accuracy_score(y_test, y_pred_sm)


d = {'Technique': ['Random UnderSampling', 'Oversampling (SMOTE)'], 'Score': [undersample_score, oversample_score]}
final_df = pd.DataFrame(data=d)

# Move column
score = final_df['Score']
final_df.drop('Score', axis=1, inplace=True)
final_df.insert(1, 'Score', score)

# Note how high is accuracy score it can be misleading! 
final_df


# SMOTE这种对不平衡数据的处理方式可以带来更好的效果
