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
#在这里可以调参
hidden_dim = 256
random_seed = 49
random_state=0
epochs= 20
np.random.seed(random_seed)
random.seed(random_seed)
torch.manual_seed(random_seed)

# 检查CUDA是否可用，然后选择设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

from transformers import AutoModel, AutoTokenizer, BertModel

class BertBasedClassifier(nn.Module):
    def __init__(self, bert_model_name='bert-base-chinese', output_dim=2):
        super(BertBasedClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.classifier = nn.Linear(self.bert.config.hidden_size, output_dim)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.pooler_output  # 使用[CLS] token的输出
        return self.classifier(cls_output)


model = BertBasedClassifier(bert_model_name='./model/bert-base-chinese', output_dim=2).to(device)

# 将模型转移到选定的设备
model.to(device)

# 示例输入 (这里使用tokenizer将文本转换为BERT的输入格式)
tokenizer = AutoTokenizer.from_pretrained('./model/bert-base-chinese')

train_df = pd.read_csv('./Data/train.news.csv')
validation_df = train_df.sample(n=2000, random_state=random_state)
train_df = train_df.drop(validation_df.index)

def text_to_matrix(text, word_vecs, max_words):
    words = jieba.lcut(text)
    matrix = np.zeros((max_words, len(next(iter(word_vecs.values())))))
    for i, word in enumerate(words[:max_words]):
        if word in word_vecs:
            matrix[i] = word_vecs[word]
    return matrix

max_words = 40

def encode_texts(texts, tokenizer, max_length):
    return tokenizer(texts, padding='max_length', truncation=True, max_length=max_length, return_tensors="pt")


# 对训练集和验证集文本进行编码
encoded_train = encode_texts(train_df['Title'].tolist(), tokenizer, max_length=max_words)
encoded_validation = encode_texts(validation_df['Title'].tolist(), tokenizer, max_length=max_words)

y_train = train_df['label'].values
y_validation = validation_df['label'].values
y_train = torch.tensor(y_train, dtype=torch.long)
y_validation = torch.tensor(y_validation, dtype=torch.long)

# 更新DataLoader，以使用编码后的文本和标签
train_dataset = TensorDataset(encoded_train['input_ids'], encoded_train['attention_mask'], y_train)
validation_dataset = TensorDataset(encoded_validation['input_ids'], encoded_validation['attention_mask'], y_validation)
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
validation_loader = DataLoader(dataset=validation_dataset, batch_size=64, shuffle=False)

# 修改数据加载部分，确保张量也被送到正确的设备
def to_device(data, device):
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

optimizer = optim.Adam(model.parameters(),lr=0.0001)  # 尝试使用更大的学习率
criterion = nn.CrossEntropyLoss()

for epoch in range(10):
    for batch in train_loader:
        input_ids, attention_mask, labels = [to_device(x, device) for x in batch]
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask)
        # print(outputs.size())#结果就是[64,2]，64是batch_size，2是输出的维度
        # 提取pooler_output作为损失计算的输入
        # print(outputs)
        # print(labels)
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
        for batch in data_loader:
            input_ids, attention_mask, labels = [to_device(x, device) for x in batch]
            outputs = model(input_ids, attention_mask=attention_mask)
            probs = F.softmax(outputs, dim=1)
            _, predicted = torch.max(probs, 1)
            y_pred.extend(predicted.cpu().numpy())
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


# 读取测试集数据
test_df = pd.read_csv('./Data/test.news.csv')

# 对测试集文本进行编码
encoded_test = encode_texts(test_df['Title'].tolist(), tokenizer, max_length=max_words)

# 获取测试集标签
y_test = test_df['label'].values
y_test = torch.tensor(y_test, dtype=torch.long)

# 创建测试集的TensorDataset
test_dataset = TensorDataset(encoded_test['input_ids'], encoded_test['attention_mask'], y_test)
# 创建DataLoader用于测试集
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

# 在测试集上评估模型
print("Evaluation on Test Set:")
evaluate_model(test_loader, model)
