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