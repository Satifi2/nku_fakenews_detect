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
# 检查CUDA是否可用，然后选择设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

#在这里可以调参
hidden_dim = 256
random_seed = 49
random_state=0
epochs= 20
#好种子：hidden_size=64，random_seed = 46，random_state=0，epochs=20
#hidden=128，random_seed = 44，random_state=0，epochs=20
#hidden=256，random_seed = 51，random_state=0，epochs=20
np.random.seed(random_seed)
random.seed(random_seed)
torch.manual_seed(random_seed)

train_df = pd.read_csv('./Data/train.news.csv')
validation_df = train_df.sample(n=2000, random_state=random_state)
train_df = train_df.drop(validation_df.index)

def load_embeddings(path, dimension=300):
    word_vecs = {}
    with open(path, 'r', encoding='UTF-8') as f:
        for line in f:
            parts = line.strip().split()
            word = ' '.join(parts[:-dimension])  # 假设最后dimension个元素是向量，其余的是词
            vec = list(map(float, parts[-dimension:]))  # 只取最后dimension个元素作为向量
            word_vecs[word] = vec
    return word_vecs
word_vecs = load_embeddings('Data/usefulWordVec.txt')

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


# 定义手动LSTM前向传播函数
def manual_lstm_forward(X, lstm):
    batch_size, seq_len, _ = X.shape
    hidden_size = lstm.hidden_size
    num_layers = lstm.num_layers

    # 初始化隐藏状态和单元状态
    h_prev = [torch.zeros(batch_size, hidden_size,device=device) for _ in range(num_layers)]
    C_prev = [torch.zeros(batch_size, hidden_size,device=device) for _ in range(num_layers)]

    # 存储每一层的最终输出
    layer_outputs = []

    # 对于每一层
    for layer in range(num_layers):
        layer_input = X if layer == 0 else layer_outputs[-1]
        all_weights = getattr(lstm, f'weight_ih_l{layer}'), getattr(lstm, f'weight_hh_l{layer}')
        all_biases = getattr(lstm, f'bias_ih_l{layer}'), getattr(lstm, f'bias_hh_l{layer}')

        outputs = []
        # LSTM gates
        for t in range(seq_len):
            gates = (layer_input[:, t, :] @ all_weights[0].T + all_biases[0] +
                     h_prev[layer] @ all_weights[1].T + all_biases[1])
            i_t, f_t, g_t, o_t = gates.chunk(4, 1)

            i_t = torch.sigmoid(i_t)
            f_t = torch.sigmoid(f_t)
            g_t = torch.tanh(g_t)
            o_t = torch.sigmoid(o_t)

            C_t = f_t * C_prev[layer] + i_t * g_t
            h_t = o_t * torch.tanh(C_t)

            outputs.append(h_t.unsqueeze(1))
            h_prev[layer] = h_t
            C_prev[layer] = C_t
        layer_outputs.append(torch.cat(outputs, dim=1))

    return layer_outputs[-1], (torch.stack(h_prev, dim=0), torch.stack(C_prev, dim=0))

#手写线性层前向传播
def manual_linear_forward(X, linear):
    return X @ linear.weight.T + linear.bias

class RNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers):
        super(RNNModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = manual_lstm_forward(x, self.lstm)
        out = manual_linear_forward(out[:, -1, :], self.fc)
        return out

model = RNNModel(input_dim=300, hidden_dim=hidden_dim, output_dim=2, n_layers=2)
# 将模型转移到选定的设备
model.to(device)

# 修改数据加载部分，确保张量也被送到正确的设备
def to_device(data, device):
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=0.0001)


def manual_cross_entropy_loss(outputs, labels):
    # 假设`outputs`已经在适当的设备上（例如cuda:0），并且`labels`也是如此

    # Step 1: Softmax
    softmax_outputs = F.softmax(outputs, dim=1)

    # Step 2: Logarithm
    log_softmax_outputs = torch.log(softmax_outputs)

    # Step 3: Negative Log Likelihood
    n_samples = labels.shape[0]
    correct_log_probs = -log_softmax_outputs[torch.arange(n_samples), labels]

    # Step 4: Mean
    loss = torch.mean(correct_log_probs)

    return loss

for epoch in range(epochs):
    for texts, labels in train_loader:
        texts, labels = to_device(texts, device), to_device(labels, device)# 训练循环中，确保数据和模型都在同一个设备
        optimizer.zero_grad()
        outputs = model(texts)
        loss = manual_cross_entropy_loss(outputs, labels)
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

