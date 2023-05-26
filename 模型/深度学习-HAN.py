import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder, StandardScaler
import re
import MeCab
import torch
import torch.nn as nn
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import joblib
from HAN import HierarchialAttentionNetwork, train, visualize_attention
from transformers import BertJapaneseTokenizer
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.metrics import accuracy_score, classification_report


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
    sentence_per_document_data = []
    word_per_sentence_data = []
    for item in tqdm(range(len(data))):
        title = re.sub('[\xa0\u3000]', '', data.loc[item, 'title'])
        summary_text = re.sub('[\xa0\u3000]', '', data.loc[item, 'summary_text'])
        main_text = re.sub('[\xa0\u3000]', '', data.loc[item, 'main_text'])

        title = re.sub('[ ・=【】■\-/◆*「」＝()（）『』<>〈〉｛｝［］{}«»‹›＜＞《》“”‘’〝〞\"\'＂＇´～＆&＠@#＃]', '', title)
        summary_text = re.sub('[・=【】■\-/◆*「」＝()（）『』<>〈〉｛｝［］{}«»‹›＜＞《》“”‘’〝〞\"\'＂＇´～＆&＠@#＃]', '', summary_text)
        main_text = re.sub('[ ・=【】■\-/◆*「」＝()（）『』<>〈〉｛｝［］{}«»‹›＜＞《》“”‘’〝〞\"\'＂＇´～＆&＠@#＃]', '', main_text)

        title = re.split('[。\n！!?？…]', title)
        summary_text = re.split('[。\n！!?？…]', summary_text)
        main_text = re.split('[。\n！!?？…]', main_text)

        texts = title + summary_text + main_text
        texts = [sentence for sentence in texts if sentence != '']

        if len(texts) >= sentence_per_document:
            texts = texts[:sentence_per_document]
            document_len = sentence_per_document
            sentence_len = [min(word_per_sentence, len(sentence)) for sentence in texts]
        else:
            document_len = len(texts)
            sentence_len = [min(word_per_sentence, len(sentence)) for sentence in texts]
            sentence_len.extend([0 for i in range(sentence_per_document - len(texts))])
            texts.extend(
                ['' for i in range(sentence_per_document - len(texts))])
        text = tokenizer(texts,
                         padding="max_length",
                         truncation=True,
                         max_length=word_per_sentence,
                         add_special_tokens=False,
                         # return_tensors='pt'
                         )
        sentence_per_document_data.append(document_len)
        word_per_sentence_data.append(sentence_len)
        text_data.append(text['input_ids'])

    return torch.tensor(text_data, dtype=torch.long), \
           torch.tensor(sentence_per_document_data, dtype=torch.long), \
           torch.tensor(word_per_sentence_data, dtype=torch.long)

# 去停用词版本

# def get_text(data):
#     text_data = []
#     sentence_per_document_data = []
#     word_per_sentence_data = []
#     for item in tqdm(range(len(data))):
#         texts = data.loc[item, 'title'] + '\n' + data.loc[item, 'summary_text'] + '\n' + data.loc[item, 'main_text']
#         texts = re.sub('[\xa0\u3000]', '', texts)
#
#         texts = re.sub('[ ・=【】■\-/◆*「」＝()（）『』<>〈〉｛｝［］{}«»‹›＜＞《》“”‘’〝〞\"\'＂＇´～＆&＠@#＃]', '', texts)
#
#         with open('stopword.txt', encoding='utf-8') as f:
#             stopwords = f.readlines()
#         # print(len(texts), end=' ')
#         for stopword in stopwords:
#             stopword = re.sub('\n', '', stopword)
#             texts = re.sub(stopword, '', texts)
#         # print(len(texts))
#
#         texts = re.split('[。\n！!?？…]', texts)
#         texts = [sentence for sentence in texts if sentence != '']
#
#         if len(texts) >= sentence_per_document:
#             texts = texts[:sentence_per_document]
#             document_len = sentence_per_document
#             sentence_len = [min(word_per_sentence, len(sentence)) for sentence in texts]
#         else:
#             document_len = len(texts)
#             sentence_len = [min(word_per_sentence, len(sentence)) for sentence in texts]
#             sentence_len.extend([0 for i in range(sentence_per_document - len(texts))])
#             texts.extend(
#                 ['' for i in range(sentence_per_document - len(texts))])
#         text = tokenizer(texts,
#                          padding="max_length",
#                          truncation=True,
#                          max_length=word_per_sentence,
#                          add_special_tokens=False,
#                          # return_tensors='pt'
#                          )
#         sentence_per_document_data.append(document_len)
#         word_per_sentence_data.append(sentence_len)
#         text_data.append(text['input_ids'])
#
#     return torch.tensor(text_data, dtype=torch.long), \
#            torch.tensor(sentence_per_document_data, dtype=torch.long), \
#            torch.tensor(word_per_sentence_data, dtype=torch.long)


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
        self.documents, self.sentences_per_document, self.words_per_sentence = get_text(data)
        print(self.documents.shape, self.sentences_per_document.shape, self.words_per_sentence.shape)
        self.numeric = torch.tensor(numeric, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return self.numeric.shape[0]

    def __getitem__(self, item):
        return self.documents[item, :], \
               self.sentences_per_document[item], \
               self.words_per_sentence[item, :], \
               self.numeric[item, :], \
               self.y[item]


def plot_train_val_loss_acc(history):
    train_history = history[0]
    train_history = [(i[0], i[1]) for i in train_history]
    test_history = history[1]
    test_history = [(i[0], i[1]) for i in test_history]
    train_history = np.array(train_history)
    test_history = np.array(test_history)
    ax1 = plt.subplot(1, 2, 1)
    plt.plot(range(len(train_history)), train_history[:, 0])
    plt.plot(range(0, len(train_history), 1), test_history[:, 0])

    ax2 = plt.subplot(1, 2, 2)
    plt.plot(range(len(train_history)), train_history[:, 1])
    plt.plot(range(0, len(train_history), 1), test_history[:, 1])
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


def lasso_rg(Xtrain, Xtest, ytrain, ytest):
    lasso = Lasso(alpha=0.1)
    lasso.fit(Xtrain, ytrain)
    predict = lasso.predict(Xtrain)
    predict[predict >= 1] = 1
    predict[predict < 1] = 0
    ytest[ytest >= 1] = 1
    ytest[ytest < 1] = 0
    ytrain[ytrain >= 1] = 1
    ytrain[ytrain < 1] = 0
    print(accuracy_score(ytrain, predict))
    print(classification_report(ytrain, predict))
    predict = lasso.predict(Xtest)
    predict[predict >= 1] = 1
    predict[predict < 1] = 0
    print(accuracy_score(ytest, predict))
    print(classification_report(ytest, predict))


if __name__ == '__main__':
    # ------------------------------------------ #
    # SETTINGS readscv
    data_path = '../数据预处理/data.csv'
    encoding = 'utf-8-sig'
    model_path = 'bert-small-japanese'
    weight_path = 'model_weight'
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', 20)
    # ------------------------------------------ #
    # ------------------------------------------ #
    # SETTINGS HAN
    epochs = 31
    batch_size = 128
    lr = 1e-3
    embed_size = 16
    dense_hidden_size = 32

    word_gru_size = 16
    word_gru_layers = 2
    word_att_size = 32

    sentence_gru_size = 16
    sentence_gru_layers = 2
    sentence_att_size = 32

    word_per_sentence = 50
    sentence_per_document = 150

    dropout = 0.5
    fine_tune_word_embeddings = False  # fine-tune word embeddings?
    test = False

    device = torch.cuda.current_device()
    tokenizer = BertJapaneseTokenizer.from_pretrained(model_path)
    # ------------------------------------------ #

    Xtrain, Xtest, ytrain, ytest = load_data(data_path, test)

    numeric_train = get_numeric(Xtrain)
    # scaler = StandardScaler()
    scaler = joblib.load('scaler.pkl')
    numeric_train = scaler.fit_transform(numeric_train)
    # joblib.dump(scaler, 'scaler.pkl')

    numeric_test = get_numeric(Xtest)
    numeric_test = scaler.transform(numeric_test)

    logistic_cls(numeric_train, numeric_test, ytrain, ytest)

    train_dataset = dataset(Xtrain, numeric_train, ytrain)
    test_dataset = dataset(Xtest, numeric_test, ytest)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, drop_last=False)

    logistic_cls(Xtrain, Xtest, ytrain, ytest)
    lasso_rg(Xtrain, Xtest, ytrain, ytest)

    model = HierarchialAttentionNetwork(vocab_size=32768,
                                        emb_size=embed_size,
                                        word_rnn_size=word_gru_size,
                                        sentence_rnn_size=sentence_gru_size,
                                        word_rnn_layers=word_gru_layers,
                                        sentence_rnn_layers=sentence_gru_layers,
                                        word_att_size=word_att_size,
                                        sentence_att_size=sentence_att_size,
                                        numeric_size=numeric_train.shape[1],
                                        dense_hidden_size=dense_hidden_size,
                                        dropout=dropout)

    optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    criterion = nn.CrossEntropyLoss()

    model = model.to(device)
    criterion = criterion.to(device)

    # model.load_state_dict(torch.load('HAN 调参/eopch_19_loss_0.3499_acc0.8663185378590078.pth'))

    # model.eval()
    # true = 0
    # cnt = 0
    # for batch in tqdm(test_loader):
    #     documents, sentences_per_document, words_per_sentence, numeric, y = [x.to(device) for x in batch]
    #     with torch.no_grad():
    #         scores, word_alphas, sentence_alphas = model(documents, sentences_per_document, words_per_sentence, numeric)
    #     for j in range(documents.shape[0]):
    #         doc = []
    #         for i in range(documents[j].shape[0]):
    #             sentence = [i for i in tokenizer.decode(documents[j, i, :]).split() if i != '[PAD]']
    #             if len(sentence) == 0:
    #                 break
    #             doc.append(sentence)
    #         if scores[j].item() > 0.8 or scores[j].item() < 0.2:
    #             cnt += 1
    #             if scores[j].item() > 0.8 and y[j] == 1:
    #                 true += 1
    #             elif scores[j].item() < 0.2 and y[j] == 0:
    #                 true += 1
    #         if cnt < 100:
    #             visualize_attention(doc=doc, scores=scores[j], word_alphas=word_alphas[j],
    #                                 sentence_alphas=sentence_alphas[j], words_in_each_sentence=words_per_sentence[j],
    #                                 y=ytest[j])
    #         # print(ytest[j], Xtest.loc[j, 'percent'])
    # print(true / cnt)

    history = train(net=model,
                    train_loader=train_loader,
                    test_loader=test_loader,
                    lr=lr,
                    epochs=epochs,
                    device=device,
                    weight_path=weight_path)

    plot_train_val_loss_acc(history)
