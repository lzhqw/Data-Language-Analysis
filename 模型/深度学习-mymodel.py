import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder, StandardScaler
import re
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.preprocessing import StandardScaler
from transformers import BertJapaneseTokenizer
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.metrics import accuracy_score, classification_report
from MyModel import MyModel, train


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


def get_y(data):
    y = data['percent']
    y = y.to_numpy() / 100
    y[y >= 1] = 1
    y[y < 1] = 0
    return y


def drop(data):
    print(f'drop前 数据集数量{len(data)}')
    data = data.drop(data[data['duration'] < 0].index)
    data = data.drop(data[data['condition'] == 'in progress'].index)
    data = data.drop(data[data['condition'] == 'unknown'].index)
    encoder = LabelEncoder()
    data['category'] = encoder.fit_transform(data['category'])
    data['location'] = encoder.fit_transform(data['location'])
    data['type_'] = encoder.fit_transform(data['type_'])
    data = data.reset_index(drop=True)
    print(f'drop后 数据集数量{len(data)}')
    return data


def get_text(data):
    text_data = []
    for item in tqdm(range(len(data))):
        title = re.sub('[\xa0\u3000]', '', data.loc[item, 'title'])
        summary_text = re.sub('[\xa0\u3000]', '', data.loc[item, 'summary_text'])
        main_text = re.sub('[\xa0\u3000]', '', data.loc[item, 'main_text'])

        title = re.sub('[ ・=【】■\-/◆*「」＝()（）『』<>〈〉｛｝［］{}«»‹›＜＞《》“”‘’〝〞\"\'＂＇´～＆&＠@#＃\r]', '', title)
        summary_text = re.sub('[・=【】■\-/◆*「」＝()（）『』<>〈〉｛｝［］{}«»‹›＜＞《》“”‘’〝〞\"\'＂＇´～＆&＠@#＃\r]', '', summary_text)
        main_text = re.sub('[ ・=【】■\-/◆*「」＝()（）『』<>〈〉｛｝［］{}«»‹›＜＞《》“”‘’〝〞\"\'＂＇´～＆&＠@#＃\r]', '', main_text)

        title = re.split('[。\n！!?？…]', title)
        summary_text = re.split('[。\n！!?？…]', summary_text)
        main_text = re.split('[。\n！!?？…]', main_text)

        texts = title + summary_text + main_text
        texts = ''.join(sentence for sentence in texts)
        # print(tokenizer(texts, return_tensors='pt',add_special_tokens=False)['input_ids'].shape)
        text_data.append(texts)
    text_data = tokenizer(text_data,
                          padding="max_length",
                          truncation=True,
                          max_length=2500,
                          add_special_tokens=False,
                          return_tensors='pt')
    print(text_data['input_ids'].shape)

    return text_data['input_ids']


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


class dataset(Dataset):
    def __init__(self, data, numeric, y):
        self.text = get_text(data)
        self.numeric = torch.tensor(numeric, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, item):
        return self.text[item, :], self.numeric[item, :], self.y[item]


def logistic_cls(Xtrain, Xtest, ytrain, ytest):
    lg = LogisticRegression()
    lg.fit(Xtrain, ytrain)
    predict = lg.predict(Xtrain)
    print(accuracy_score(ytrain, predict))
    print(classification_report(ytrain, predict))
    predict = lg.predict(Xtest)
    print(accuracy_score(ytest, predict))
    print(classification_report(ytest, predict))


if __name__ == '__main__':
    # ------------------------------------------ #
    # SETTINGS readscv
    data_path = 'autodl-tmp/data.csv'
    encoding = 'utf-8-sig'
    model_path = 'autodl-tmp/bert-small-japanese'
    weight_path = 'autodl-tmp/model_weight'
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', 20)
    # ------------------------------------------ #
    # ------------------------------------------ #
    # SETTINGS HAN
    epochs = 70
    batch_size = 256
    embed_size = 16
    gru_hidden_size = 16
    fc_hidden_size = 16
    num_layers = 2
    lr = 1e-3

    test = False

    device = torch.cuda.current_device()
    tokenizer = BertJapaneseTokenizer.from_pretrained(model_path)
    # ------------------------------------------ #

    Xtrain, Xtest, ytrain, ytest = load_data(data_path, test)

    numeric_train = get_numeric(Xtrain)
    scaler = StandardScaler()
    numeric_train = scaler.fit_transform(numeric_train)

    numeric_test = get_numeric(Xtest)
    numeric_test = scaler.transform(numeric_test)

    train_dataset = dataset(Xtrain, numeric_train, ytrain)
    test_dataset = dataset(Xtest, numeric_test, ytest)

    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, drop_last=False)

    # 采样
    pos_num = np.sum(ytrain)
    print(pos_num)
    pos_weight = 1 / pos_num
    neg_weight = 1 / (len(ytrain) - pos_num)
    train_weight = np.zeros(len(ytrain))
    for i in range(len(ytrain)):
        if ytrain[i] == 1:
            train_weight[i] = pos_weight
        else:
            train_weight[i] = neg_weight
    sampler = WeightedRandomSampler(weights=train_weight, num_samples=8000 * 2, replacement=False)

    model = MyModel(vocab_size=32768,
                    embed_size=embed_size,
                    gru_hidden_size=gru_hidden_size,
                    num_layers=num_layers,
                    numeric_size=numeric_train.shape[1],
                    fc_hidden_size=fc_hidden_size)
    train(net=model,
          train_loader_params={'dataset': train_dataset, 'batch_size': batch_size, 'sampler': sampler},
          test_loader=test_loader,
          lr=lr,
          epochs=epochs,
          device=device,
          weight_path=weight_path)
