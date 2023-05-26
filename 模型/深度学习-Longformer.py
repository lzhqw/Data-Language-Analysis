from transformers import LongformerModel, T5Tokenizer, LongformerConfig, AutoTokenizer
import pandas as pd
import re
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder, StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from Longformer import BertCls, train


def drop(data):
    print(f'drop前 数据集数量{len(data)}')
    data = data.drop(data[data['duration'] < 0].index)
    data = data.drop(data[data['condition'] == 'in progress'].index)
    encoder = LabelEncoder()
    data['category'] = encoder.fit_transform(data['category'])
    data['location'] = encoder.fit_transform(data['location'])
    data['type_'] = encoder.fit_transform(data['type_'])
    data = data.reset_index(drop=True)
    print(f'drop后 数据集数量{len(data)}')
    return data


def get_y(data):
    y = data['percent']
    y = y.to_numpy() / 100
    y[y >= 1] = 1
    y[y < 1] = 0
    return y


def load_data(data_path, test):
    data = pd.read_csv(data_path, encoding=encoding)
    print(data.columns)
    data = drop(data)
    data['summary_text'] = data['summary_text'].map(lambda x: '' if pd.isna(x) else x)
    data = data.sample(frac=1, random_state=12, replace=False).reset_index(drop=True)
    if test:
        data = data[:1000]
    train_num = int(len(data) * 0.8)
    Xtrain = data[:train_num]
    Xtest = data[train_num:].reset_index(drop=True)
    ytrain = get_y(Xtrain)
    ytest = get_y(Xtest)

    return Xtrain, Xtest, ytrain, ytest


def get_text(data, item):
    title = re.sub('[\xa0\u3000]', '', data.loc[item, 'title'])
    summary_text = re.sub('[\xa0\u3000]', '', data.loc[item, 'summary_text'])
    main_text = re.sub('[\xa0\u3000]', '', data.loc[item, 'main_text'])

    title = re.sub('[。\n！!?？…・=【】■\-/◆*「」＝()（）『』<>〈〉｛｝［］{}«»‹›＜＞《》“”‘’〝〞\"\'＂＇´～＆&＠@#＃]', ' ', title)
    summary_text = re.sub('[。\n！!?？…・=【】■\-/◆*「」＝()（）『』<>〈〉｛｝［］{}«»‹›＜＞《》“”‘’〝〞\"\'＂＇´～＆&＠@#＃]', ' ', summary_text)
    main_text = re.sub('[。\n！!?？…・=【】■\-/◆*「」＝()（）『』<>〈〉｛｝［］{}«»‹›＜＞《》“”‘’〝〞\"\'＂＇´～＆&＠@#＃]', ' ', main_text)

    text = title + summary_text + main_text
    text = " ".join(sentence for sentence in text)
    text = re.sub(' +', '', text)
    # print(tokenizer.tokenize('[CLS]'+text))
    # print(tokenizer.decode(tokenizer(text, padding="max_length", truncation=True, max_length=512)['input_ids']))
    text = tokenizer('[CLS]' + text,
                     padding="max_length",
                     truncation=True,
                     max_length=4032,
                     return_tensors='pt')

    return text


def get_numeric(data):
    data = data[['title_length', 'is_store_opening', 'has_target_money',
                 'has_expiration', 'is_accepting_support', 'hide_collected_money',
                 'is_new_store_opening', 'summary_text_length', 'main_text_length',
                 'target_amount', 'activity_num', 'start_at', 'end_at', 'is_end',
                 'remain_time', 'duration', 'tag_num', 'category', 'location',
                 'type_', 'choice_num', 'min_price', 'max_price', 'avg_price', 'img_num']]
    for column in ['target_amount', 'activity_num', 'min_price', 'max_price', 'avg_price']:
        data[column] = data[column].map(lambda x: np.log(x + 1))

    return data


class get_dataset(Dataset):
    def __init__(self, data, numeric, y):
        self.data = data
        self.numeric = numeric
        self.y = y

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        text = get_text(self.data, item)
        ids = text['input_ids'][0, :]
        mask = text['attention_mask'][0, :]
        # token_type_ids = text["token_type_ids"][0, :]
        return {
            'input_ids': torch.tensor(ids, dtype=torch.long),
            'attention_mask': torch.tensor(mask, dtype=torch.long),
            # 'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'numeric': self.numeric[item, :],
            'targets': torch.tensor(self.y[item], dtype=torch.float)
        }


if __name__ == '__main__':
    # SETTINGS -------------------------------- #
    model_path = "roberta-small-japanese"
    data_path = "../数据预处理/data.csv"
    weight_path = "model_weight"
    encoding = 'utf-8-sig'
    device = torch.cuda.current_device()
    test = True
    # ----------------------------------------- #
    # SETTINGS -------------------------------- #
    word_per_sentence = 50
    sentence_per_article = 120
    batch_size = 2
    lr = 1e-3
    epochs = 11
    # ----------------------------------------- #

    # model = BertModel.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    Xtrain, Xtest, ytrain, ytest = load_data(data_path, test=test)

    numeric_train = get_numeric(Xtrain)
    scaler = StandardScaler()
    numeric_train = scaler.fit_transform(numeric_train)

    numeric_test = get_numeric(Xtest)
    numeric_test = scaler.transform(numeric_test)

    train_dataset = get_dataset(Xtrain, numeric_train, ytrain)
    test_dataset = get_dataset(Xtest, numeric_test, ytest)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size)

    model = BertCls(model_path, numeric_train.shape[1])

    history = train(net=model,
                    train_loader=train_loader,
                    test_loader=test_loader,
                    lr=lr,
                    epochs=epochs,
                    device=device,
                    weight_path=weight_path)
