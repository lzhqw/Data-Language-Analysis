import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder, StandardScaler
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from HANmulticls import HierarchialAttentionNetwork, train, visualize_attention
from transformers import BertJapaneseTokenizer


def load_data(data_path, test):
    data = pd.read_csv(data_path, encoding=encoding)
    print(data.columns)
    data = drop(data)
    data['summary_text'] = data['summary_text'].map(lambda x: '' if pd.isna(x) else x)
    data = data.sample(frac=1, random_state=42, replace=False).reset_index(drop=True)
    if test:
        data = data[:1000]

    category_count = dict(data['category'].value_counts())
    print(category_count)

    encoder = LabelEncoder()
    data['category'] = encoder.fit_transform(data['category'])
    classes = encoder.classes_
    num_classes = len(encoder.classes_)
    print(num_classes)

    category_weight = [0 for i in range(num_classes)]

    max_category = max(category_count.values())
    for i, class_ in enumerate(classes):
        category_weight[i] = 1 / category_count[class_]
    print(category_weight)
    category_weight = torch.tensor(category_weight, dtype=torch.float32)

    y = get_y(data, num_classes)
    train_num = int(len(data) * 0.8)
    Xtrain = data[:train_num]
    Xtest = data[train_num:].reset_index(drop=True)
    ytrain = y[:train_num]
    ytest = y[train_num:]

    # print(Xtrain['category'])
    # print(torch.argmax(ytrain, dim=1))

    return Xtrain, Xtest, ytrain, ytest, num_classes, category_weight, encoder.classes_


def get_y(data, num_classes):
    y = data['category']
    y = y.to_numpy()
    y = torch.tensor(y, dtype=torch.long)
    # y = F.one_hot(torch.tensor(y, dtype=torch.long), num_classes=num_classes).float()
    return y


def drop(data):
    print(f'drop前 数据集数量{len(data)}')
    data = data.drop(data[data['duration'] < 0].index)
    data = data.drop(data[data['condition'] == 'in progress'].index)
    data = data.drop(data[data['condition'] == 'unknown'].index)
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


class dataset(Dataset):
    def __init__(self, data, y):
        self.documents, self.sentences_per_document, self.words_per_sentence = get_text(data)
        print(self.documents.shape, self.sentences_per_document.shape, self.words_per_sentence.shape)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, item):
        return self.documents[item, :], \
               self.sentences_per_document[item], \
               self.words_per_sentence[item, :], \
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


def draw_attention_fig(model, test_loader, device):
    model.eval()
    cnt = 0
    true = 0
    for batch in tqdm(test_loader):
        documents, sentences_per_document, words_per_sentence, y = [x.to(device) for x in batch]
        with torch.no_grad():
            scores, word_alphas, sentence_alphas = model(documents, sentences_per_document, words_per_sentence)
        for j in range(documents.shape[0]):
            doc = []
            for i in range(documents[j].shape[0]):
                sentence = [i for i in tokenizer.decode(documents[j, i, :]).split() if i != '[PAD]']
                if len(sentence) == 0:
                    break
                doc.append(sentence)
            cnt += 1
            scores = scores.cpu()
            score = F.softmax(scores[j])
            max_idx = np.argmax(score)
            category = class_list[max_idx]
            if y[j].cpu() == max_idx:
                true += 1
            if cnt < 100:
                visualize_attention(doc=doc, scores=score[max_idx], word_alphas=word_alphas[j],
                                    sentence_alphas=sentence_alphas[j], words_in_each_sentence=words_per_sentence[j],
                                    y=y[j], category=category, idx=max_idx)
    print(true/cnt)
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
    # print(true / cnt)


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
    epochs = 100
    batch_size = 64
    lr = 1e-3
    embed_size = 64

    word_gru_size = 50
    word_gru_layers = 2
    word_att_size = 100

    sentence_gru_size = 50
    sentence_gru_layers = 2
    sentence_att_size = 100

    word_per_sentence = 40
    sentence_per_document = 120

    dropout = 0.5
    fine_tune_word_embeddings = False  # fine-tune word embeddings?
    test = False

    device = torch.cuda.current_device()
    tokenizer = BertJapaneseTokenizer.from_pretrained(model_path)
    # ------------------------------------------ #

    Xtrain, Xtest, ytrain, ytest, num_classes, category_weight, class_list = load_data(data_path, test)
    print(class_list)
    # train_dataset = dataset(Xtrain, ytrain)
    test_dataset = dataset(Xtest, ytest)

    train_weight = np.zeros(len(ytrain))
    for i in range(len(train_weight)):
        train_weight[i] = category_weight[ytrain[i]]
    train_weight = torch.FloatTensor(train_weight)
    print(train_weight.shape)
    print(train_weight)
    sampler = WeightedRandomSampler(weights=train_weight, num_samples=500 * num_classes, replacement=False)

    # train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, sampler=sampler)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, drop_last=False)

    model = HierarchialAttentionNetwork(n_classes=num_classes,
                                        vocab_size=32768,
                                        emb_size=embed_size,
                                        word_rnn_size=word_gru_size,
                                        sentence_rnn_size=sentence_gru_size,
                                        word_rnn_layers=word_gru_layers,
                                        sentence_rnn_layers=sentence_gru_layers,
                                        word_att_size=word_att_size,
                                        sentence_att_size=sentence_att_size,
                                        dropout=dropout)
    model.load_state_dict(torch.load('model_weight/[HAN_multi]eopch_53_loss_0.9744_acc0.767449956483899.pth'))
    model = model.to(device)
    draw_attention_fig(model, test_loader, device)

    # history = train(net=model,
    #                 train_loader_params={'dataset': train_dataset, 'batch_size': batch_size, 'sampler': sampler},
    #                 test_loader=test_loader,
    #                 num_classes=num_classes,
    #                 lr=lr,
    #                 epochs=epochs,
    #                 device=device,
    #                 weight_path=weight_path,
    #                 weight=category_weight)
    #
    # plot_train_val_loss_acc(history)
