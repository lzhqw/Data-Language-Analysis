from transformers import BertJapaneseTokenizer, BertForPreTraining, BertModel
import pandas as pd
import re
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder, StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from BERT import BertCls, train
from sklearn.linear_model import LogisticRegression, Lasso, LinearRegression
from sklearn.metrics import accuracy_score, classification_report

def drop(data):
    print(f'drop前 数据集数量{len(data)}')
    data = data.drop(data[data['duration'] < 0].index)
    data = data.drop(data[data['condition'] == 'in progress'].index)
    data['type_'] = data['type_'].map(lambda x: 'All in' if x == '販売中' else x)
    data = data.drop(data[data['type_'] == 'このプロジェクトは目標金額の達成に関わらず、応援購入を申し込んだ時点で決済が確定します。'].index)
    encoder = LabelEncoder()
    # data['category'] = encoder.fit_transform(data['category'])
    # data['location'] = encoder.fit_transform(data['location'])
    # data['type_'] = encoder.fit_transform(data['type_'])
    for col in ['category', 'location', 'type_']:
        print(pd.get_dummies(data[col], prefix=col).columns[0])
        dummy_variable = pd.get_dummies(data[col], prefix=col, drop_first=True)
        data.drop(columns=col, inplace=True)
        data = pd.concat([data, dummy_variable], axis=1)
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


def get_text(data):
    text_data = []
    for i in tqdm(range(len(data))):
        title = re.sub('[\xa0\u3000 ]', '', data.loc[i, 'title'])
        summary_text = re.sub('[\xa0\u3000 ]', '', data.loc[i, 'summary_text'])
        main_text = re.sub('[\xa0\u3000 ]', '', data.loc[i, 'main_text'])

        title = re.split('([。\n！!?？…])', title)
        summary_text = re.split('([。\n！!?？…])', summary_text)
        main_text = re.split('([。\n！!?？…])', main_text)

        title = ["".join(i) for i in zip(title[0::2], title[1::2])]
        summary_text = ["".join(i) for i in zip(summary_text[0::2], summary_text[1::2])]
        main_text = ["".join(i) for i in zip(main_text[0::2], main_text[1::2])]

        text = title + summary_text + main_text
        # if len(text) >= sentence_per_article:
        #     text = text[:sentence_per_article]
        # else:
        #     text.extend(['' for i in range(sentence_per_article-len(text))])
        tokens = tokenizer(text,
                           padding="max_length",
                           truncation=True,
                           max_length=word_per_sentence,
                           return_tensors='pt'
                           )
        text_data.append(tokens)
    return text_data
    #         if len(tokens['input_ids']) < 100:
    #             word_per_sentence.append(len(tokens['input_ids']))
    # sns.kdeplot(x=word_per_sentence, fill=True)
    # plt.savefig('word_per_sentence.pdf', dpi=300)


def get_numeric(data):
    # data = data[['title_length', 'is_store_opening', 'has_target_money',
    #              'has_expiration', 'is_accepting_support', 'hide_collected_money',
    #              'is_new_store_opening', 'summary_text_length', 'main_text_length',
    #              'target_amount', 'activity_num', 'start_at', 'end_at', 'is_end',
    #              'remain_time', 'duration', 'tag_num', 'category', 'location',
    #              'type_', 'choice_num', 'min_price', 'max_price', 'avg_price', 'img_num']]
    data = data.drop(labels=['id_', 'collected_money', 'collected_supporter', 'percent',
                             'title', 'summary_text', 'main_text', 'thumb_ups', 'is_end', 'remain_time',
                             'condition', 'curr_time', 'end_at', 'comment_num', 'is_store_opening', 'is_new',
                            'is_new_store_opening'], axis=1)
    for column in ['target_amount', 'activity_num', 'min_price', 'max_price', 'avg_price']:
        data[column] = data[column].map(lambda x: np.log(x + 1))

    return data


class get_dataset(Dataset):
    def __init__(self, text, numeric, y):
        self.text = text
        self.numeric = numeric
        self.y = y
        assert len(text) == self.numeric.shape[0] == self.y.shape[0]

    def __len__(self):
        return len(self.text)

    def __getitem__(self, item):
        return self.text[item], self.numeric[item, :], torch.tensor(self.y[item], dtype=torch.float32)


def plot_train_val_loss_acc(history):
    train_history = history[0]
    train_history = [(i[0], i[1]) for i in train_history]
    test_history = history[1]
    test_history = [(i[0], i[1]) for i in test_history]
    train_history = np.array(train_history)
    test_history = np.array(test_history)
    ax1 = plt.subplot(1,2,1)
    plt.plot(range(len(train_history)), train_history[:, 0])
    plt.plot(range(0, len(train_history), 5), test_history[:,0])

    ax2 = plt.subplot(1,2,2)
    plt.plot(range(len(train_history)), train_history[:,1])
    plt.plot(range(0, len(train_history), 5), test_history[:,1])
    plt.savefig('loss_acc.png')

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
    # SETTINGS -------------------------------- #
    model_path = "autodl-tmp/bert-small-japanese"
    data_path = "autodl-tmp/data.csv"
    weight_path = "autodl-tmp/model_weight"
    logpath = "autodl-tmp/"
    encoding = 'utf-8-sig'
    device = torch.cuda.current_device()
    test = False
    # ----------------------------------------- #
    # SETTINGS -------------------------------- #
    word_per_sentence = 50
    sentence_per_article = 120
    batch_size = 1
    lr = 1e-3
    epochs = 30
    # ----------------------------------------- #

    # model = BertModel.from_pretrained(model_path)
    tokenizer = BertJapaneseTokenizer.from_pretrained(model_path)

    Xtrain, Xtest, ytrain, ytest = load_data(data_path, test=test)



    numeric_train = get_numeric(Xtrain)
    scaler = StandardScaler()
    numeric_train = scaler.fit_transform(numeric_train)

    numeric_test = get_numeric(Xtest)
    numeric_test = scaler.transform(numeric_test)
    
    logistic_cls(numeric_train, numeric_test, ytrain, ytest)
    
    text_train = get_text(Xtrain)
    text_test = get_text(Xtest)
    
    train_dataset = get_dataset(text_train, numeric_train, ytrain)
    test_dataset = get_dataset(text_test, numeric_test, ytest)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, drop_last=False)

    model = BertCls(model_path, numeric_train.shape[1])

    history = train(net=model,
                    train_loader=train_loader,
                    test_loader=test_loader,
                    lr=lr,
                    epochs=epochs,
                    device=device,
                    weight_path=weight_path,
                    logpath=logpath
                   )

    plot_train_val_loss_acc(history)

    # res = model(torch.tensor([[2, 2442, 2886, 40, 17, 953, 5, 7591, 2586, 2992, 8, 3]]))
    # print(res)
    # print(res.last_hidden_state.shape)
