# -*- coding: utf-8 -*-
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


def load_data(train_path, test_path):
    train = pd.read_csv(train_path, header=None)
    test = pd.read_csv(test_path, header=None)
    train.columns = ['epoch', 'loss', 'acc']
    test.columns = ['epoch', 'loss', 'acc']
    train['type'] = 'train'
    test['type'] = 'val'
    data = pd.concat([train, test])
    data.reset_index(inplace=True, drop=True)
    print(data)
    plt.figure(figsize=(8, 6))
    plt.subplots_adjust(left=0.09, bottom=0.08, right=0.96, top=0.95)
    sns.set_palette(['#9F3636', '#0063CA'])
    ax1 = plt.subplot(1, 2, 1)
    sns.lineplot(x='epoch', y='loss', hue='type', data=data, ax=ax1)
    ax2 = plt.subplot(1, 2, 2)
    sns.lineplot(x='epoch', y='acc', hue='type', data=data, ax=ax2)
    plt.savefig(train_path[:-4] + '.pdf', dpi=300)


def plot_category():
    plt.rcParams['font.sans-serif'] = ['Yu Gothic']
    count = {'プロダクト': 15064, 'ファッション': 4312, 'フード': 3308, 'レストラン・バー': 1067, 'コスメ・ビューティー': 825, '地域活性化': 785,
             'スポーツ': 559, '社会貢献': 517, 'テクノロジー': 474, 'アート・写真': 281, '音楽': 273, 'スタートアップ': 251, '映画・映像': 218, '教育': 201,
             'ゲーム': 186, '出版・ジャーナリズム': 132, 'アニメ・マンガ': 120, '演劇・パフォーマンス': 81, 'お笑い・エンタメ': 55, '世界一周': 12}
    data = []
    for i in count.items():
        data.append([i[0], i[1]])

    data = pd.DataFrame(columns=['category', 'count'], data=data)
    print(data)
    plt.figure(figsize=(8, 5))
    plt.subplots_adjust(left=0.1, bottom=0.26, top=0.95, right=0.95)
    ax = sns.barplot(x='category', y='count', data=data)
    ax.tick_params(axis='x', labelsize=7, rotation=-60)
    # plt.show()
    plt.savefig('category.pdf', dpi=300)


if __name__ == '__main__':
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)

    train_path = 'han multi train.txt'
    test_path = 'han multi test.txt'

    # load_data(train_path, test_path)
    plot_category()
