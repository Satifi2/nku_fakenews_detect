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

useful_word_vecs={}#记录有用的词向量的字典
def text_to_matrix(text, word_vecs, max_words):
    words = jieba.lcut(text)
    matrix = np.zeros((max_words, len(next(iter(word_vecs.values())))))
    for i, word in enumerate(words[:max_words]):
        if word in word_vecs:
            matrix[i] = word_vecs[word]
            useful_word_vecs[word]=word_vecs[word]
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

# 遍历字典并写入文件
with open('./Data/usefulWordVec.txt', 'w', encoding='utf-8') as f:
    for word, vec in useful_word_vecs.items():
        line = f"{word} {' '.join(map(str, vec))}\n"
        f.write(line)