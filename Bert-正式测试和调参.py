from transformers import BertModel, BertTokenizer
import torch.nn as nn


class BertBasedClassifier(nn.Module):
    def __init__(self, bert_model_name='bert-base-uncased', output_dim=2):
        super(BertBasedClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.classifier = nn.Linear(self.bert.config.hidden_size, output_dim)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.pooler_output  # 使用[CLS] token的输出
        return self.classifier(cls_output)


# 实例化模型
model = BertBasedClassifier()

# 示例输入 (这里使用tokenizer将文本转换为BERT的输入格式)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
input_text = ["Here is some input text for classification"] * 64  # 假设有64个样本
inputs = tokenizer(input_text, padding=True, truncation=True, return_tensors="pt", max_length=40)

# 模型前向传播
output = model(inputs.input_ids, inputs.attention_mask)
print(output.shape)  # 预期输出形状为: (64, 2)
