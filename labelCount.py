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