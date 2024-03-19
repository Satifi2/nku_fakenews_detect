from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
import pandas as pd

df = pd.read_csv('./Data/train.news.csv') #读取训练集,df是一个DataFrame的简称，类似于表格

df_sample = df.sample(n=2000, random_state=42) #随机抽取2000个样本

train_df, validation_df = train_test_split(df_sample, test_size=0.2, random_state=42) #划分训练集和验证集

vectorizer = TfidfVectorizer() #定义一个TfidfVectorizer对象，目的是将文本向量化
X_train = vectorizer.fit_transform(train_df['Title']) #将训练集的Title列向量化
y_train = train_df['label'] #训练集的标签，用来监督学习
X_validation = vectorizer.transform(validation_df['Title']) #将验证集的Title列向量化
y_validation = validation_df['label'] #验证集的标签，用来评价模型

models = {
    "KNeighborsClassifier": KNeighborsClassifier(),#K近邻
    "RandomForestClassifier": RandomForestClassifier(random_state=42),#随机森林
    "SVC": SVC(random_state=42),#支持向量机
    "AdaBoostClassifier": AdaBoostClassifier(random_state=42),#AdaBoost
    "LinearSVC": LinearSVC(random_state=42),#线性支持向量机
    "MultinomialNB": MultinomialNB(),#朴素贝叶斯
    "DecisionTreeClassifier": DecisionTreeClassifier(random_state=42),#决策树
    "GradientBoostingClassifier": GradientBoostingClassifier(random_state=42),#梯度提升
    "MLPClassifier": MLPClassifier(random_state=42),#神经网络
    "LogisticRegression": LogisticRegression(random_state=42)#逻辑回归
}

for name, model in models.items():#遍历models字典
    model.fit(X_train, y_train) #训练模型
    train_score = model.score(X_train, y_train) #训练集准确率
    validation_score = model.score(X_validation, y_validation) #验证集准确率
    print(f"{name} has a training score of {train_score} and a validation score of {validation_score}")


