# 任务描述

#### 一、数据

数据集是中文微信消息，包括微信消息的Official Account Name，Title，News Url，Image Url，Report Content，label。Title是微信消息的标题，label是消息的真假标签（0是real消息，1是fake消息）。训练数据保存在`train.news.csv`，测试数据保存在`test.news.csv`。

![image-20240319103733684](./assets/image-20240319103733684.png)

#### 二、任务(建议使用Scikit Learn和PyTorch实现)

**1. (数据统计)分别统计训练集和测试集中真假样本的数量，填写下表1。**

**Table 1. statistics of data**

|                  | #real | #fake | #total |
| ---------------- | ----- | ----- | ------ |
| `train.news.csv` |       |       |        |
| `test.news.csv`  |       |       |        |

#### 2. (验证集上调节参数)只根据微信消息的title文字，预测消息的真假。

从`train.news.csv`中随机抽取2000个样本作为验证集(validation set)，剩余的数据作为训练集(training set)，使用`test.news.csv`中的样本作为测试集(testing set)。使用中文预训练词向量语料库[建议使用[https://github.com/Embedding/Chinese-Word-Vectors](https://github.com/Embedding/Chinese-Word-Vectors)中的微博语料库，也可以使用其他。]把中文微信消息的title表示成词向量。根据训练集数据，训练两层的RNN, LSTM和GRU模型。模型隐层节点数量参数值的调节范围是{64,128,256}，通过在验证集上评价，得到的结果填入表2。

**Table 2. results of intermediate tasks on validation set**

| Model | Hidden | Label | Precision | Recall | F1   | Accuracy | AUC  |
| ----- | ------ | ----- | --------- | ------ | ---- | -------- | ---- |
| RNN   | 64     | Real  |           |        |      |          |      |
|       |        | Fake  |           |        |      |          |      |
| LSTM  | 64     | Real  |           |        |      |          |      |
|       |        | Fake  |           |        |      |          |      |
| GRU   | 64     | Real  |           |        |      |          |      |
|       |        | Fake  |           |        |      |          |      |
| RNN   | 128    | Real  |           |        |      |          |      |
|       |        | Fake  |           |        |      |          |      |
| LSTM  | 128    | Real  |           |        |      |          |      |
|       |        | Fake  |           |        |      |          |      |
| GRU   | 128    | Real  |           |        |      |          |      |
|       |        | Fake  |           |        |      |          |      |
| RNN   | 256    | Real  |           |        |      |          |      |
|       |        | Fake  |           |        |      |          |      |
| LSTM  | 256    | Real  |           |        |      |          |      |
|       |        | Fake  |           |        |      |          |      |
| GRU   | 256    | Real  |           |        |      |          |      |
|       |        | Fake  |           |        |      |          |      |

#### 3. (测试集上评价模型)根据表2中的结果,选择在验证集上效果最好的RNN, LSTM和GRU模型，并在测试集上评价这些模型。

然后在测试集上评价得到accuracy、recall, F1，并计算得出Accuracy和AUC评价结果，填入表3。

**Table 3. results of intermediate tasks on testing set**

| Model | Hidden | Label | Precision | Recall | F1 | Accuracy | AUC |
|-------|--------|-------|-----------|--------|----|----------|-----|
| RNN   |        | Real  |           |        |    |          |     |
|       |        | Fake  |           |        |    |          |     |
| LSTM  |        | Real  |           |        |    |          |     |
|       |        | Fake  |           |        |    |          |     |
| GRU   |        | Real  |           |        |    |          |     |
|       |        | Fake  |           |        |    |          |     |

# 代码

#### 统计label数

**已知：**

- 在一个机器学习任务当中，数据集是中文微信消息，包括微信消息的Official Account Name，Title，News Url，Image Url，Report Content，label。Title是微信消息的标题，label是消息的真假标签（0是real消息，1是fake消息）。训练数据保存在`./Data/train.news.csv`，测试数据保存在`./Data/test.news.csv`。

- 其中csv格式：行间用换行隔开，行内用英文逗号隔开，格子内用中文逗号。

- csv文件的第一行:`Ofiicial Account Name,Title,News Url,Image Url,Report Content,label`

![image-20240319105004619](./assets/image-20240319105004619.png)

**任务：统计label数**

- 用panda读取训练集和测试集，分别统计训练集和测试集中真样本的数量、假样本的数量、样本总数。

```py
import pandas as pd #导入pandas库

train_data_path = './Data/train.news.csv' #训练集路径
test_data_path = './Data/test.news.csv' #测试集路径

train_data = pd.read_csv(train_data_path) #读取训练集
test_data = pd.read_csv(test_data_path) #读取测试集

print("训练集标签分布")
print(train_data['label'].value_counts()) #输出训练集标签分布
print(len(train_data)) #输出训练集样本个数
print("测试集标签分布") 
print(test_data['label'].value_counts()) #输出测试集标签分布
print(len(test_data)) #输出测试集样本个数
```

![image-20240319104330627](./assets/image-20240319104330627.png)

#### 传统机器学习-sklearn

**任务：用Title预测label**

- 用panda读取训练数据`./Data/train.news.csv`和测试数据`./Data/test.news.csv`。，从`train.news.csv`中随机抽取2000个样本作为验证集(validation set)，剩余的数据作为训练集(training set)。
- 使用sklearn的库对新闻标题Title进行向量化处理，用来作为输入，而label作为监督。

- 用sklearn的机器学习库，定义十个模型模型：
  ` KNeighborsClassifier、RandomForestClassifier、SVC、AdaBoostClassifier、LinearSVC、MultinomialNB、DecisionTreeClassifier、GradientBoostingClassifier、MLPClassifier、LogisticRegression `
  为了试验的可重复性，定义模型的时候需要设置种子（如果模型有`random_state`参数）。
- 遍历所有的模型，在训练集上训练，给出模型在训练集和验证集上的score。

```py
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
import pandas as pd #导入pandas库

# Load the training data
train_df = pd.read_csv('./Data/train.news.csv')

# Splitting the training data into training and validation sets
train_set, validation_set = train_test_split(train_df, test_size=2000, random_state=42)

# Vectorizing the titles in the training and validation sets
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(train_set['Title'])
X_validation = vectorizer.transform(validation_set['Title'])

y_train = train_set['label']
y_validation = validation_set['label']

# Define the models with a common random state where applicable
models = {
    "KNeighborsClassifier": KNeighborsClassifier(),
    "RandomForestClassifier": RandomForestClassifier(random_state=42),
    "SVC": SVC(random_state=42),
    "AdaBoostClassifier": AdaBoostClassifier(random_state=42),
    "LinearSVC": LinearSVC(random_state=42),
    "MultinomialNB": MultinomialNB(),
    "DecisionTreeClassifier": DecisionTreeClassifier(random_state=42),
    "GradientBoostingClassifier": GradientBoostingClassifier(random_state=42),
    "MLPClassifier": MLPClassifier(random_state=42),
    "LogisticRegression": LogisticRegression(random_state=42)
}

# Training and evaluating each model
scores = {}
for model_name, model in models.items():
    model.fit(X_train, y_train)
    train_score = model.score(X_train, y_train)
    validation_score = model.score(X_validation, y_validation)
    scores[model_name] = (train_score, validation_score)
    print(f"{model_name} training score: {train_score}")
```

然后遍历所有模型进行训练，在测试集上面测试后输出score。

**方案：**

首先，为了使用`sklearn`的机器学习库定义各个模型并设置可重复性的种子，我们可以初始化每个模型时指定其`random_state`参数（对于那些支持这一参数的模型）。不是所有模型都有`random_state`参数，比如`KNeighborsClassifier`和`MultinomialNB`就没有。对于这些模型，我们将直接初始化它们，因为它们的结果通常是确定的或者不依赖于随机性。以下是如何定义每个模型：

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression

# 定义模型列表
models = [
    KNeighborsClassifier(),
    RandomForestClassifier(random_state=42),
    SVC(random_state=42),
    AdaBoostClassifier(random_state=42),
    LinearSVC(random_state=42),
    MultinomialNB(),
    DecisionTreeClassifier(random_state=42),
    GradientBoostingClassifier(random_state=42),
    MLPClassifier(random_state=42),
    LogisticRegression(random_state=42)
]

# 模型名称列表，用于打印结果
model_names = [
    "KNeighborsClassifier",
    "RandomForestClassifier",
    "SVC",
    "AdaBoostClassifier",
    "LinearSVC",
    "MultinomialNB",
    "DecisionTreeClassifier",
    "GradientBoostingClassifier",
    "MLPClassifier",
    "LogisticRegression"
]
```

接下来，我们需要对文本数据进行预处理（比如向量化），然后训练这些模型，并在测试集上测试它们的性能。这里，假设你的文本数据已经是特征向量化后的结果，或者你将进行这样的转换。然后，可以遍历所有模型进行训练，并在测试集上输出它们的`score`。

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# 假设'Title'列是我们要向量化的文本
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(train_data['Title'])
y_train = train_data['label']
X_test = vectorizer.transform(test_data['Title'])
y_test = test_data['label']

# 遍历模型进行训练和测试
for model, name in zip(models, model_names):
    model.fit(X_train, y_train)  # 训练模型
    y_pred = model.predict(X_test)  # 在测试集上进行预测
    score = accuracy_score(y_test, y_pred)  # 计算准确率
    print(f"{name} accuracy: {score}")
```

请注意，`LinearSVC`和`MLPClassifier`等模型在大数据集上可能需要较长的训练时间，而且`MLPClassifier`的训练过程可能特别慢。此外，`LinearSVC`可能会警告关于收敛的问题，这可以通过增加`max_iter`参数的值来解决。如果你的数据是非向量化的文本数据，你需要先进行向量化或其他形式的特征提取/转换。
