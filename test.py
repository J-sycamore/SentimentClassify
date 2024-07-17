from transformers import AutoModelForSequenceClassification
import pandas as pd
import torch
from transformers import AutoTokenizer


def judge(x):
    if x > 0:
        return 1
    else:
        return 0


if __name__ == '__main__':
    model = AutoModelForSequenceClassification.from_pretrained('./checkpoint').to('cuda')
    tokenizer1 = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')
    tokenizer2 = AutoTokenizer.from_pretrained('bert-base-chinese')
    sentence1 = ['So cool', 'You are beautiful', 'It tastes good']
    sentence2 = ['相当不错的热干面，配冰可乐巨爽', '米饭太硬了', '卤肉饭没有肉']
    encoded_input1 = tokenizer1(sentence1, padding=True, truncation=True, return_tensors='pt').to('cuda')
    encoded_input2 = tokenizer2(sentence2, padding=True, truncation=True, return_tensors='pt',
                                return_token_type_ids=False).to('cuda')

    with torch.no_grad():
        model_output1 = model(**encoded_input1)
        model_output2 = model(**encoded_input2)

    for i in range(0, 3):
        x = judge(sum(model_output1[0][i]))
        print(x)

    for i in range(0, 3):
        x = judge(sum(model_output2[0][i]))
        print(x)