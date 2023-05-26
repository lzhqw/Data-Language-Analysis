import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import re


def displot(data):
    fig, axes = plt.subplots(4, 5, figsize=(25, 12))
    # sns.color_palette("Spectral", as_cmap=True)
    plt.subplots_adjust(left=0.040, bottom=0.04, right=0.99, top=0.98, wspace=0.2, hspace=0.3)
    for i, column in enumerate(['collected_money', 'collected_supporter', 'percent', 'title_length',
                                'summary_text_length', 'main_text_length', 'target_amount', 'thumb_ups',
                                'activity_num', 'comment_num', 'start_at', 'end_at', 'curr_time',
                                'duration', 'tag_num', 'choice_num', 'min_price', 'max_price',
                                'avg_price', 'img_num']):
        if i != 19:
            sns.kdeplot(data=data, x=column, hue='condition', fill=True, ax=axes[i // 5][i - (i // 5) * 5],
                        legend=False)
        else:
            ax = sns.kdeplot(data=data, x=column, hue='condition', fill=True, ax=axes[i // 5][i - (i // 5) * 5])
            sns.move_legend(loc='lower right', obj=ax, bbox_to_anchor=(1, 0))

    plt.savefig('hist.pdf', dpi=300)
    plt.show()


def boxplot(data):
    log_column = ['percent', 'collected_money', 'collected_supporter', 'min_price', 'max_price', 'avg_price',
                  'comment_num',
                  'target_amount', 'thumb_ups', 'activity_num', 'choice_num']
    for column in log_column:
        data[column] = data[column].map(lambda x: np.log(x + 1))
        data = data.rename(columns={column: 'log_' + column})
    fig, axes = plt.subplots(4, 5, figsize=(25, 12))
    plt.subplots_adjust(left=0.040, bottom=0.04, right=0.99, top=0.98, wspace=0.2, hspace=0.3)
    for i, column in enumerate(['collected_money', 'collected_supporter', 'percent', 'title_length',
                                'summary_text_length', 'main_text_length', 'target_amount', 'thumb_ups',
                                'activity_num', 'comment_num', 'start_at', 'end_at', 'curr_time',
                                'duration', 'tag_num', 'choice_num', 'min_price', 'max_price',
                                'avg_price', 'img_num']):
        if column in log_column:
            column = 'log_' + column
        if i != 19:
            sns.boxenplot(data=data, y=column, x='condition', ax=axes[i // 5][i - (i // 5) * 5])
        else:
            ax = sns.boxenplot(data=data, y=column, x='condition', ax=axes[i // 5][i - (i // 5) * 5])
            # sns.move_legend(loc='lower right', obj=ax, bbox_to_anchor=(1, 0))
    plt.savefig('box.pdf', dpi=300)
    plt.show()


def freq_label(data):
    plt.rcParams['font.sans-serif'] = ['Yu Gothic']
    fig, axes = plt.subplots(3, 1, figsize=(15, 15))
    sns.color_palette("Set2")
    plt.subplots_adjust(left=0.04, bottom=0.08, right=0.98, top=0.98, wspace=None, hspace=0.5)
    for i, column in enumerate(['category', 'location', 'type_']):
        categorys = {}
        for j in range(len(data)):
            conditoin = data.loc[j, 'condition']
            category = data.loc[j, column]
            categorys.setdefault(category, [0, 0, 0])
            if conditoin == 'in progress':
                categorys[category][0] += 1
            elif conditoin == 'success':
                categorys[category][1] += 1
            else:
                categorys[category][2] += 1
        for key in categorys.keys():
            categorys[key] = np.array(categorys[key]) / sum(categorys[key])
        print(categorys)
        if 'このプロジェクトは目標金額の達成に関わらず、応援購入を申し込んだ時点で決済が確定します。' in categorys:
            categorys.pop('このプロジェクトは目標金額の達成に関わらず、応援購入を申し込んだ時点で決済が確定します。')
        df = pd.concat({k: pd.Series(v) for k, v in categorys.items()}).reset_index()
        df.columns = [column, 'condition', 'value']
        d = {0: 'in progress', 1: 'success', 2: 'fail'}
        df['condition'] = df['condition'].map(lambda x: d[x])
        ax = sns.barplot(data=df, x=column, y='value', hue='condition', ax=axes[i], palette='Set2')
        ax.tick_params(axis='x', labelsize=7, rotation=-90)
    plt.savefig('label.pdf', dpi=300)


def heatmap(data):
    from sklearn.preprocessing import LabelEncoder
    data = data.drop(data[data['condition'] == 'in progress'].index)
    data = data.reset_index(drop=True)
    encoder = LabelEncoder()
    data['category'] = encoder.fit_transform(data['category'])
    data['location'] = encoder.fit_transform(data['location'])
    data['type_'] = encoder.fit_transform(data['type_'])
    data = data[['collected_money', 'collected_supporter', 'percent', 'thumb_ups', 'comment_num',
                 'title_length', 'is_store_opening', 'has_target_money',
                 'has_expiration', 'is_accepting_support', 'hide_collected_money',
                 'is_new_store_opening', 'summary_text_length', 'main_text_length',
                 'target_amount', 'activity_num', 'start_at', 'end_at', 'duration',
                 'tag_num', 'category', 'location',
                 'type_', 'choice_num', 'min_price', 'max_price', 'avg_price', 'img_num']]
    corr = data.corr()
    plt.figure(figsize=(10, 8))
    plt.subplots_adjust(left=0.2, right=0.95, top=0.95, bottom=0.23)
    sns.heatmap(corr, cmap='YlGnBu')  # YlGnBu, GnBu
    plt.savefig('heatmap.pdf', dpi=300)


if __name__ == '__main__':
    pd.set_option('display.max_columns', None)
    data = pd.read_csv('data.csv', encoding='utf-8-sig')
    print(data.columns)
    # data.describe().round(2).T.to_csv('describe.csv',encoding='utf-8')
    data = data.drop(data[data['duration'] < 0].index)
    data = data.reset_index()
    # displot(data)
    # boxplot(data)
    # freq_label(data)
    heatmap(data)
