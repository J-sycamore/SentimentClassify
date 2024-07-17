from transformers import AutoModelForSequenceClassification
import pandas as pd
import torch
from transformers import AutoTokenizer, BertTokenizer


def judge(x):
    if x > 0:
        return 1
    else:
        return 0


if __name__ == '__main__':
    model = AutoModelForSequenceClassification.from_pretrained('./checkpoint').to('cuda')
    # tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')
    tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')
    fdata = pd.read_csv('./testdata/waimai.csv', encoding='gb18030')
    # fdata = pd.read_csv('./testdata/IMDB Dataset.csv')
    print(fdata)
    sentence = []
    for i in fdata['review']:
        sentence.append(i)
    encoded_input = tokenizer(sentence[3900:4100], padding=True, truncation=True, return_tensors='pt',
                              return_token_type_ids=False).to('cuda')
    with torch.no_grad():
        model_output = model(**encoded_input)

    datasum = model_output[0].shape[0]
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for i in range(0, datasum):
        x = judge(sum(model_output[0][i]))
        if x == 1 and fdata['label'][i] == 1:
            TP += 1
        elif x == 1 and fdata['label'][i] == 0:
            FP += 1
        elif x == 0 and fdata['label'][i] == 1:
            FN += 1
        else:
            FN += 1

    acc = (TP + TN) / 200
    pre = TP / (TP + FP)
    recall = TP / (TP + FN)
    F1 = 2*TP / (2*TP + FP + FN)
    print("acc: ", acc)
    print("pre: ", pre)
    print("recall: ", recall)
    print("F1: ", F1)
