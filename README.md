## 介绍

- 利用自然语言处理技术，执行文本分类任务。

- 使用了机器学习的若干种方法以及CNN、RNN、GRU、LSTM，在测试集上准确率超过了90%。并提供了简单版、GPU训练版，在CNN、RNN上提供了纯手写版（即不调库进行张量计算）。
- 基本步骤：加载训练集、验证集、测试集，将文本词向量化(通过读取预训练的词向量)，构建模型，训练，验证，调参，测试。

##### 下载预训练词向量

从`https://github.com/Embedding/Chinese-Word-Vectors`当中找到搜狗新闻，选择`Word+Character+Ngram`这一类进行下载，之后解压，将得到的`sgns.sogounews.bigram-char`存入Data文件夹中。
![image-20240321222518261](./assets/image-20240321222518261.png)

#### 下载`bert`模型

去hugging-face官网下载这五个文件

![image-20240321225621951](./assets/image-20240321225621951.png)

放到`./model/bert-base-chinese`当中。

![image-20240321231053516](./assets/image-20240321231053516.png)



## 任务描述

#### 一、数据

数据集是中文微信消息，包括微信消息的Official Account Name，Title，News Url，Image Url，Report Content，label。Title是微信消息的标题，label是消息的真假标签（0是real消息，1是fake消息）。训练数据保存在`train.news.csv`，测试数据保存在`test.news.csv`。

![image-20240319103733684](./assets/image-20240319103733684.png)

#### 二、任务

**(建议使用Scikit Learn和PyTorch实现)**

**1. (数据统计)分别统计训练集和测试集中真假样本的数量，填写下表1。**

**Table 1. statistics of data**

|                  | #real | #fake | #total |
| ---------------- | ----- | ----- | ------ |
| `train.news.csv` |       |       |        |
| `test.news.csv`  |       |       |        |

**2. (验证集上调节参数)只根据微信消息的title文字，预测消息的真假。**

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

**3. (测试集上评价模型)根据表2中的结果,选择在验证集上效果最好的RNN, LSTM和GRU模型，并在测试集上评价这些模型。**

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

## 代码

#### 统计label数

**已知：**

- 在一个机器学习任务当中，数据集是中文微信消息，包括微信消息的Official Account Name，Title，News Url，Image Url，Report Content，label。Title是微信消息的标题，label是消息的真假标签（0是real消息，1是fake消息）。训练数据保存在`./Data/train.news.csv`，测试数据保存在`./Data/test.news.csv`。

- 其中csv格式：行间用换行隔开，行内用英文逗号隔开，格子内用中文逗号。

- csv文件的第一行:`Ofiicial Account Name,Title,News Url,Image Url,Report Content,label`

![image-20240319105004619](./assets/image-20240319105004619.png)

**任务：统计label数**

- 用pandas读取训练集和测试集，分别统计训练集和测试集中真样本的数量、假样本的数量、样本总数。

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

![image-20240320111846594](./../nku_fakenews_detect - 副本/assets/image-20240320111846594.png)

**任务：用Title预测label**

- 用pandas读取数据`./Data/train.news.csv`。从`train.news.csv`中随机抽取2000个样本作为验证集(validation set)，剩余的数据作为训练集(training set)。
- 使用sklearn的库对新闻标题Title进行向量化处理，用来作为输入，而label作为监督。

- 用sklearn的机器学习库，定义十个模型模型：
  ` KNeighborsClassifier、RandomForestClassifier、SVC、AdaBoostClassifier、LinearSVC、MultinomialNB、DecisionTreeClassifier、GradientBoostingClassifier、MLPClassifier、LogisticRegression `
  为了试验的可重复性，定义模型的时候需要设置种子（如果模型有`random_state`参数）。
- 遍历所有的模型，在训练集上训练，之后给出模型在训练集和验证集上的score。

```py
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

```

![image-20240319114536044](./assets/image-20240319114536044.png)

### rnn

**已知:**

在路径`Data/sgns.sogounews.bigram-char`[没有后缀的纯文本]下有一个预训练好的中文词向量语料库，每个词向量有300个元素。

格式：第一行包含两个整数，代表词向量总数，和词向量维度，中间空格隔开。从第二行开始才是词向量，每行的开头位置是词向量对应的词语，例如"山峰"。然后是词向量当中的元素，例如-1.224051 0.255035...(中间也用空格隔开)。注意，有的词内含有空格，会导致报错，需要用try在报错时跳过。

**任务：用pytorch构建一个rnn，用训练集训练，在验证集测试，用Title作为输入，用label作为监督**

- 数据集划分，将`./Data/train.news.csv`当中的随机2000个样本划分为验证集，剩下的作为训练集，将"Title"这一列作为输入X,将"label"这一列作为预测y。构建出：X_train，y_train，X_validation，y_validation。

- 读取预训练好的词向量，构建起{词语:词向量}字典

- 数据预处理，使用分词工具jieba，对X_train和X_validation（对应"Title"这一列）中的文本进行分词，将每一个Title拆分为若干个词，然后在{词语:词向量}字典当中找到对应词向量，将一个Title当中含有的多个词向量堆叠起来，使得一个Title对应一个词矩阵。

- 由于不同的Title含有的词数可能不同，需要求出Title含有的最大词数，然后对于每一个title，如果其中含有的词数小于最大词数，需要补充全0的词向量，目的是保证所有Title都对应一个相同形状的词矩阵。到此为止，我们构建好了模型输入X，应该是一个三维张量，形状为:[Title数，Title含有的最大词数，词向量维度数]

- 构建一个网络，包含两层RNN层，其中hidden的大小都为64。然后是一个全连接层，用softmax输出标签的概率分布，对X_train进行训练。

- 对于已经训练好的模型，用X_validation进行测试，调用库函数进行评估。


![image-20240320171452512](./assets/image-20240320171452512.png)

#### 注释版代码

```python
import pandas as pd #用来读取csv
import numpy as np
import torch #用来构建模型
from torch.utils.data import DataLoader, TensorDataset
import jieba #用来分词

# 读取数据
train_df = pd.read_csv('./Data/train.news.csv') #读取csv作为一个DataFrame

# 数据集划分
validation_df = train_df.sample(n=2000) #随机抽取2000个样本作为验证集
train_df = train_df.drop(validation_df.index) #剩下的作为训练集

jieba.initialize() #初始化jieba分词

def load_embeddings(path, dimension=300): #加载预训练词向量，构建从词到向量的映射
    word_vecs = {} #词到向量的映射
    with open(path, 'r', encoding='UTF-8') as f: #打开文件
        n, _ = map(int, f.readline().strip().split()) #读取词向量的个数和维度
        for line in f: #遍历文件的每一行，从第二行开始
            parts = line.strip().split() #strip会移除头尾的空白字符，split会将每行按空格分割，parts形如["山峰","0.124","-0.234",...]
            word = ' '.join(parts[:-dimension])  #从parts列表中选择除了最后dimension个元素之外的所有元素，不用parts[0]是因为有些词可能包含空格，join是将上一步选出的元素（词或短语的部分）用空格连接起来
            vec = list(map(float, parts[-dimension:]))#将parts列表中最后dimension个元素转换为浮点数，list是将map的结果转换为列表
            word_vecs[word] = vec #将词到向量的映射存入字典
    return word_vecs #word_vecs形如{"山峰":[0.124,-0.234,...],"河流":[0.234,0.123,...],...}

word_vecs = load_embeddings('Data/sgns.sogounews.bigram-char')#调用刚才的函数，加载预训练词向量

def text_to_matrix(text, word_vecs, max_words):#text类似于"山峰河流"
    words = jieba.lcut(text) #words类似于["山峰","河流"]
    matrix = np.zeros((max_words, 300)) #matrix形如[[0,0,...,0],[0,0,...,0],...]，有40行，每行有300个0
    for i, word in enumerate(words[:max_words]): #word形如"山峰"，i是下标例如7，:max_words是防止越界
        if word in word_vecs: #如果"山峰"在word_vecs中
            matrix[i] = word_vecs[word] #得到"山峰"的词向量，存入matrix的第i行
    return matrix #matrix形如[[0.124,-0.234,...],[0.234,0.123,...],...]，对应"山峰河流"...，如果words较少，后面的行全是0

# 计算最大词数，我试验的结果是40，也就是说最长的标题有40个词
max_words = max(train_df['Title'].apply(lambda x: len(jieba.lcut(x))).max(),#对于Title这一列中的每一个元素x，用jieba分词，记录总词数，取最大值
                validation_df['Title'].apply(lambda x: len(jieba.lcut(x))).max())
print("max_words:",max_words)

# 预处理文本数据,得到的X_train是一个三维数组
X_train = np.array([text_to_matrix(title, word_vecs, max_words) for title in train_df['Title']])#对于train_df中Title列的每一个Title，用text_to_matrix函数处理，得到一个矩阵，存入X_train
#X_train形如[[[0.124,-0.234,...],[0.234,0.123,...],...],[[0.789,-0.678,...],[0.679,0.798,...],...],...]，对应每一个Title，例如"山峰河流"，"惊天消息"
y_train = train_df['label'].values#y_train是train_df中label列的值，也就是标签

X_validation = np.array([text_to_matrix(title, word_vecs, max_words) for title in validation_df['Title']])#对于验证集方法一模一样
y_validation = validation_df['label'].values

# 转换为PyTorch张量，直接用torch.tensor函数，输入是numpy数组，dtype是数据类型
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
X_validation = torch.tensor(X_validation, dtype=torch.float32)
y_validation = torch.tensor(y_validation, dtype=torch.long)
print("shape of 4 tensors:",X_train.shape,y_train.shape,X_validation.shape,y_validation.shape)
#我的结果:shape of 4 tensors: torch.Size([8587, 40, 300]) torch.Size([8587]) torch.Size([2000, 40, 300]) torch.Size([2000])
#解释：X_train有8587个Title，每个Title固定40个词(不够的补0)，每个词是300维的词向量；y_train有8587个标签；X_validation有2000个Title，每个Title固定40个词(不够的补0)，每个词是300维的词向量；y_validation有2000个标签

train_dataset = TensorDataset(X_train, y_train)#到时候可以取出来Title和label
validation_dataset = TensorDataset(X_validation, y_validation)
print("shape of train_dataset:",len(train_dataset),len(train_dataset[0]),len(train_dataset[0][0]),len(train_dataset[0][0][0]))
#输出：shape of train_dataset: 8587 2 40 300，说明train_dataset有8587个样本，每个样本包含Title和label，Title有40个词，每个词是300维的词向量，每个label是一个常量
print("train_dataset:",train_dataset[0])
#输出train_dataset当中的第一个样本，包含Title和label

train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)#DataLoader是一个迭代器，每次迭代会返回一个batch的数据,其中dataset是数据集，batch_size是每个batch的大小，shuffle是是否打乱数据
# for texts, labels in train_loader:#其中texts: torch.Size([64, 40, 300])相当于64个标题 。labels: torch.Size([64])相当于64个标签
validation_loader = DataLoader(dataset=validation_dataset, batch_size=64, shuffle=False)#千万不要打乱，否则预测和实际标签无法对应

import torch.nn as nn
class RNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(input_dim, hidden_dim, n_layers, batch_first=True)
        #input_dim是输入的特征维度，hidden_dim是隐藏层的维度，n_layers是RNN的层数，batch_first=True表示期待输入的张量形状是(batch, seq, feature)
        self.fc = nn.Linear(hidden_dim, output_dim)#全连接层，将隐藏层的输出映射到输出维度

    def forward(self, x):
        out, hidden = self.rnn(x)
        #print("shape of out:",out.shape)#输出shape of out: torch.Size([64, 40, 64])，表示64个标题，每个标题40个词，每个词64维的隐藏层输出
        #print("shape of out[:, -1, :]:",out[:, -1, :].shape)#输出shape of out[:, -1, :]: torch.Size([64, 64])，表示64个标题，每个标题最后一个词的隐藏层输出
        #print("shape of hidden" ,hidden.shape)#输出shape of hidden torch.Size([2, 64, 64])，表示2层RNN，每层64个标题，每个标题64维的隐藏层输出
        out = self.fc(out[:, -1, :])
        return out

model = RNNModel(input_dim=300, hidden_dim=64, output_dim=2, n_layers=2)

import torch.optim as optim
from sklearn.metrics import accuracy_score

criterion = nn.CrossEntropyLoss()#对于分类任务，一般用 CrossEntropyLoss 即交叉熵损失，网络层没有用softmax，因为CrossEntropyLoss已经包含了softmax
optimizer = optim.Adam(model.parameters()) # Adam优化器，没有定义学习率，因为有默认值

for epoch in range(10):  # 进行多轮训练
    for texts, labels in train_loader:
        # print("texts:",texts.shape,"labels:",labels.shape)#输出texts: torch.Size([64, 40, 300])，总共有64个样本，每个样本是一个词矩阵 labels: torch.Size([64])总共有64个标签
        optimizer.zero_grad()#清除之前的梯度，否则梯度将累加到一起
        outputs = model(texts)#前向传播，得到的outputs的shape为(64, 2)，表示64个标题的预测结果，2表示两个类别，outputs形如[[0.1,0.9],[0.2,0.8],...],对应分类为[1,1...]
        loss = criterion(outputs, labels)#计算损失
        loss.backward()#反向传播
        optimizer.step()#更新参数，主要工作之一确实是梯度下降
    print(f'Epoch {epoch+1}, Loss: {loss.item()}')#打印每个epoch的损失值

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score
# 模型评估
model.eval()#模型评估模式，因为有些层在训练和评估时行为不同，例如Dropout和BatchNorm
with torch.no_grad(): #不需要计算梯度
    y_pred = [] #预测的标签，用列表存储
    for texts, labels in validation_loader: #texts形如 torch.Size([64, 40, 300])，表示64个标题的词矩阵，labels形如 torch.Size([64])
        outputs = model(texts) #用texts作为输入，outputs形状是torch.Size([64, 2])，表示64个标题的预测结果，2表示两个类别，outputs形如[[0.1,0.9],[0.2,0.8],...],对应分类为[1,1...]
        _, predicted = torch.max(outputs.data, 1) #取出每个标题的预测结果中概率最大的那个类别，predicted形状是torch.Size([64])，形如[1,1,...]
        #注：torch.max(a,1)表示沿着第1维度取最大值，返回最大值和最大值的索引，这里的第1维度是类别维度
        y_pred.extend(predicted.numpy()) #将predicted转换为numpy数组，然后加入y_pred
    print("shape of y_pred:",len(y_pred))#输出shape of y_pred: 2000,之前说过，y_validation形状为torch.Size([2000])
    print("y_pred:",y_pred)#输出形如y_pred:[1,1,0,1,0,0,1,0,1,0,...]，2000个元素，每个元素是0或1，表示预测的标签
    print("y_validation:",y_validation)#输出形如y_validation:tensor([1, 1, 0, 1, 0, 0, 1, 0, 1, 0,...])，2000个元素，每个元素是0或1，表示真实的标签
    accuracy = accuracy_score(y_validation, y_pred)#和真实标签比较，计算准确率，accuracy_score来自sklearn.metrics
    print(f'Validation Accuracy: {accuracy}')
    #有一个细节问题，遍历validation_loader时，最后一个batch可能不是64，而是小于64，但是总的还是2000个样本，所以y_pred的长度是2000
    #另外，遍历validation_loader时，每个batch的顺序是固定的，所以y_pred的顺序和y_validation的顺序是一样的，否则预测和实际标签无法对应
```

#### GPU训练-调参完整版

**任务：添加测试指标**

- 在上述代码中，`print("y_pred:",y_pred)#输出y_pred:[1,1,0,1,0,0,1,0,1,0,...]，2000个元素，每个元素是0或1，表示预测的标签`
- ``print("y_validation:",y_validation)#输出形如y_validation:tensor([1, 1, 0, 1, 0, 0, 1, 0, 1, 0,...])，2000个元素，每个元素是0或1，表示真实的标签`
- `outputs形状是torch.Size([64, 2])，代表有64个样本，outputs[0]类似于：[2,1],取softmax([2,1])=[0.73,0.27]表示预测label=0的概率是0.73，label=1的概率是0.1 `
- 任务是定义一个`probabilities`列表，在for循环当中用`softmax`计算出2000个样本标签是1的概率，定义一个`y_pred = []`，来存储2000个真实的标签。之后调用`sklearn`计算标签1的精确率、召回率、`F1`；标签0的精确率、召回率、`F1`，使用`probabilities`和`y_pred` 计算出`auc`指标，`accuracy`指标。

**任务：在上述代码基础上，调库用`y_true, probabilities`画出ROC曲线。**

**任务：在上述代码的基础上，在测试集上测试** 

- 从`./Data/test.news.csv`当中加载测试集，格式和验证集完全一样。
-  将刚才在验证集上测试的代码封装成一个函数，分别输入测试集和验证集查看效果（指标不变）。

**任务：分层采样[可能不会改善效果]**
上述代码当中，`validation_df = train_df.sample(n=2000, random_state=random_seed)`，是随机采样的，然而，在`./Data/train.news.csv`当中有10587个样本，其中仅有7844个样本label是0，2743个样本label是0，经过我的测试，不同的`random_state`在验证集上的`accuracy`从86%到92%不等，因此我们需要分层采样来提高模型性能，需要给出改进之后的代码。

```py
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import jieba
import random
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score ,roc_curve
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
jieba.initialize()
#好种子46 (45、47、48)
random_seed = 46
np.random.seed(random_seed)
random.seed(random_seed)
torch.manual_seed(random_seed)

train_df = pd.read_csv('./Data/train.news.csv')
validation_df = train_df.sample(n=2000, random_state=48)
train_df = train_df.drop(validation_df.index)

def load_embeddings(path, dimension=300):
    word_vecs = {}
    with open(path, 'r', encoding='UTF-8') as f:
        n, _ = map(int, f.readline().strip().split())
        for line in f:
            parts = line.strip().split()
            word = ' '.join(parts[:-dimension])  # 假设最后dimension个元素是向量，其余的是词
            vec = list(map(float, parts[-dimension:]))  # 只取最后dimension个元素作为向量
            word_vecs[word] = vec
    return word_vecs
word_vecs = load_embeddings('Data/sgns.sogounews.bigram-char')

def text_to_matrix(text, word_vecs, max_words):
    words = jieba.lcut(text)
    matrix = np.zeros((max_words, len(next(iter(word_vecs.values())))))
    for i, word in enumerate(words[:max_words]):
        if word in word_vecs:
            matrix[i] = word_vecs[word]
    return matrix

max_words = max(train_df['Title'].apply(lambda x: len(jieba.lcut(x))).max(),
                validation_df['Title'].apply(lambda x: len(jieba.lcut(x))).max())

X_train = np.array([text_to_matrix(title, word_vecs, max_words) for title in train_df['Title']])
y_train = train_df['label'].values
X_validation = np.array([text_to_matrix(title, word_vecs, max_words) for title in validation_df['Title']])
y_validation = validation_df['label'].values

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
X_validation = torch.tensor(X_validation, dtype=torch.float32)
y_validation = torch.tensor(y_validation, dtype=torch.long)

# DataLoader
train_dataset = TensorDataset(X_train, y_train)
validation_dataset = TensorDataset(X_validation, y_validation)
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
validation_loader = DataLoader(dataset=validation_dataset, batch_size=64, shuffle=False)

class RNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(input_dim, hidden_dim, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])
        return out

model = RNNModel(input_dim=300, hidden_dim=64, output_dim=2, n_layers=2)

# 检查CUDA是否可用，然后选择设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

# 将模型转移到选定的设备
model.to(device)

# 修改数据加载部分，确保张量也被送到正确的设备
def to_device(data, device):
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

for epoch in range(20):
    for texts, labels in train_loader:
        texts, labels = to_device(texts, device), to_device(labels, device)# 训练循环中，确保数据和模型都在同一个设备
        optimizer.zero_grad()
        outputs = model(texts)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch + 1}, Loss: {loss.item()}')

# 封装测试代码为一个函数
def evaluate_model(data_loader, model):
    model.eval()
    y_pred = []
    probabilities = []
    y_true = []
    with torch.no_grad():
        for texts, labels in data_loader:
            texts, labels = to_device(texts, device), to_device(labels, device)
            outputs = model(texts)
            _, predicted = torch.max(outputs.data, 1)
            y_pred.extend(predicted.cpu().numpy())
            probs = F.softmax(outputs, dim=1)
            probabilities.extend(probs[:, 1].cpu().numpy())
            y_true.extend(labels.cpu().numpy())
    print(probabilities[:20],"\n", y_true[:20],"\n", y_pred[:20],"\n",y_validation[:20])

    precision_0 = precision_score(y_true, y_pred, pos_label=0)
    recall_0 = recall_score(y_true, y_pred, pos_label=0)
    f1_0 = f1_score(y_true, y_pred, pos_label=0)
    precision_1 = precision_score(y_true, y_pred, pos_label=1)
    recall_1 = recall_score(y_true, y_pred, pos_label=1)
    f1_1 = f1_score(y_true, y_pred, pos_label=1)
    accuracy = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, probabilities)
    print(f"Precision on Real Label (0): {precision_0}")
    print(f"Recall on Real Label (0): {recall_0}")
    print(f"F1 Score on Real Label (0): {f1_0}")
    print(f"Precision on False Label (1): {precision_1}")
    print(f"Recall on False Label (1): {recall_1}")
    print(f"F1 Score on False Label (1): {f1_1}")
    print(f"Overall Accuracy: {accuracy}")
    print(f"AUC: {auc}")

    # 使用matplotlib绘制ROC曲线
    fpr, tpr, thresholds = roc_curve(y_true, probabilities)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.show()

# 在验证集上评估模型
print("Evaluation on Validation Set:")
evaluate_model(validation_loader, model)

# 加载测试集
def load_dataset(file_path):
    df = pd.read_csv(file_path)
    X = np.array([text_to_matrix(title, word_vecs, max_words) for title in df['Title']])
    y = df['label'].values
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)
    return X, y

X_test, y_test = load_dataset('./Data/test.news.csv')
test_dataset = TensorDataset(X_test, y_test)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

# 在测试集上评估模型
print("Evaluation on Test Set:")
evaluate_model(test_loader, model)


```

#### 验证集评估

```py
Precision on Real Label (0): 0.9177090190915076
Recall on Real Label (0): 0.9495912806539509
F1 Score on Real Label (0): 0.9333779712085705
Precision on False Label (1): 0.8461538461538461
Recall on False Label (1): 0.7650375939849624
F1 Score on False Label (1): 0.8035538005923001
Overall Accuracy: 0.9005
AUC: 0.9197004773514166
```

对于标签（0）的性能：

- **精确率（Precision）**为0.9177，意味着当模型**预测一个样本为类别0时，有91.77%的几率是正确的。**
- **召回率（Recall）**为0.9496，意味着**实际为类别0的样本中，有94.96%被模型正确识别。**
- **F1分数（F1 Score）**为0.9334，是精确率和召回率的**调和**平均数，表明模型在类别0上的综合性能很高。

对于标签（1）的性能：

- **精确率（Precision）**为0.8462，意味着当模型预测一个样本为类别1时，有84.62%的几率是正确的。
- **召回率（Recall）**为0.7650，意味着实际为类别1的样本中，有76.50%被模型正确识别。
- **F1分数（F1 Score）**为0.8036，表明模型在类别1上的综合性能也相当不错，虽然相比类别0稍低一些。

总体性能：

- **整体准确率（Overall Accuracy）**为0.9005，这表示**模型正确预测了90.05%的样本**。这是一个很高的准确率，表明模型整体上表现良好。
- **AUC(Area Under the Curve)**ROC曲线下面积。如果模型很好地进行了分类，AUC值应接近于1；如果模型仅做随机猜测，AUC值应接近于0.5。
  `probabilities`数组用于存储每个样本属于类别1的预测概率，而`y_true`数组用于存储相应的真实标签。这两个数组被用于`roc_auc_score`函数来计算AUC指标。
  AUC: 0.9197说明模型有区分能力而非随机猜测。

- **ROC曲线**：
  **解释：**

  - 如果在ROC曲线上有一个点是 (0.8, 0.7)，这表示在某个特定的分类阈值【如果模型预测是正类的概率大于或等于这个阈值，我们将样本分类为正类（1），反之为负类】下，模型预测结果具有以下特性：
    - 横坐标0.7表示假正例率（FPR）是0.7。这意味着，在所有实际为标签0的样本中，有70%被模型错误地预测为标签1。
    - 纵坐标0.9表示真正例率（TPR）是0.9。这意味着，在所有实际为标签1的样本中，有90%被模型正确地预测为标签1。
  - **理想情况下**，一个完美的分类器的ROC曲线将会紧贴左上角，表示它能**以最小的FPR获得最大的TPR**。一个完全随机的分类器的ROC曲线会沿着对角线（从(0, 0)到(1, 1)）增长，表示其分类效果不比随机猜测好。

-  ![image-20240321100717749](./assets/image-20240321100717749.png)

  #### 加快训练

  我发现训练的时候加载词向量非常费时间，但实际上训练和测试用到的词向量都少，所以我打算构建一个词向量表，其中只存储有用的词向量。
  方法也非常简单，每一次Title转词向量的时候，都会有一次查表的过程，被查表的词向量全部存在`useful_word_vecs`这个字典里，然后把字典存在本地即可。由于所有用到的文本都要进行这个步骤，因此最后存下来的只有用得到的词向量。

  ```py
  useful_word_vecs={}#记录有用的词向量的字典
  def text_to_matrix(text, word_vecs, max_words):
      words = jieba.lcut(text)
      matrix = np.zeros((max_words, len(next(iter(word_vecs.values())))))
      for i, word in enumerate(words[:max_words]):
          if word in word_vecs:
              matrix[i] = word_vecs[word]
              useful_word_vecs[word]=word_vecs[word]#把有用的词向量记录下来
      return matrix
  
  # 遍历字典并写入文件
  with open('./Data/usefulWordVec.txt', 'w', encoding='utf-8') as f:
      for word, vec in useful_word_vecs.items():
          line = f"{word} {' '.join(map(str, vec))}\n"
          f.write(line)
  ```

经过我测试，所有指标数据都是一样的，说明没有问题。
#### 注意：种子对实验有很大影响

一个好的种子（验证集准确率达到94%）vs一个坏的种子（验证集准确率72.3%），参数不对甚至无法收敛，在实验数据文件里记录了这些信息。

![image-20240321154038324](./assets/image-20240321154038324.png)![image-20240321154050417](./assets/image-20240321154050417.png)



### GRU模型

只需要改变模型类即可，不再赘述

```python
class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, hidden = self.gru(x)
        out = self.fc(out[:, -1, :])
        return out

model = GRUModel(input_dim=300, hidden_dim=hidden_dim, output_dim=2, n_layers=2)
```

### LSTM模型

只需要改变模型类即可，不再赘述

```python
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, (hidden, cell) = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

model = LSTMModel(input_dim=300, hidden_dim=hidden_dim, output_dim=2, n_layers=2)
```

### CNN模型

```py
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=300, out_channels=64, kernel_size=3, stride=1, padding=1)
        # 假设卷积操作后不改变长度（由于padding=1），则输出形状为(batch_size, 64, 300)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        # 经过池化层后，长度减半，输出形状为(batch_size, 64, 150)
        self.fc = nn.Linear(1280, 2)  # 全连接层，将卷积层输出平铺后输入，输出形状为(batch_size, 2)

    def forward(self, x):
        # print("x.shape:", x.shape)  # x.shape: torch.Size([64, 40, 300])
        # 保持x的形状不变，直接用于卷积层
        x = x.permute(0, 2, 1)  # 现在调整为(batch_size, channels=300, length=40)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # 平铺操作，为全连接层准备
        # print("x.shape:", x.shape)  # x.shape: torch.Size([64, 1280])
        x = self.fc(x)
        return x

model = SimpleCNN()
```



#### 微调Bert模型之后预测

如果之前没有安装过`transformers`

```css
pip install transformers
```













