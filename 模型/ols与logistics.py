import pandas as pd
import numpy as np
import statsmodels.api as sm

def load_data(data_path, encoding):
    data = pd.read_csv(data_path, encoding=encoding)
    print(data.columns)
    data = drop(data)
    # data['type_'][data['type_']=='このプロジェクトは目標金額の達成に関わらず、応援購入を申し込んだ時点で決済が確定します。'] = 'All in'
    y = data['percent']/100

    data = data[['collected_money', 'collected_supporter', 'thumb_ups', 'comment_num',
                 'title_length', 'is_store_opening', 'has_target_money',
                 'has_expiration', 'is_accepting_support', 'hide_collected_money',
                 'is_new_store_opening', 'summary_text_length', 'main_text_length',
                 'target_amount', 'activity_num', 'start_at',
                 'duration', 'tag_num', 'category', 'location',
                 'type_', 'choice_num', 'min_price', 'max_price', 'avg_price', 'img_num']]

    for col in ['category', 'location', 'type_']:
        print(pd.get_dummies(data[col], prefix=col).columns[0])
        dummy_variable = pd.get_dummies(data[col], prefix=col, drop_first=True)
        data.drop(columns=col, inplace=True)
        data = pd.concat([data, dummy_variable], axis=1)

    for column in ['collected_money', 'target_amount', 'activity_num', 'min_price', 'max_price', 'avg_price']:
        data[column] = data[column].map(lambda x: np.log(x + 1))

    return data, y


def drop(data):
    print(f'drop前 数据集数量{len(data)}')
    data = data.drop(data[data['duration'] < 0].index)
    data = data.drop(data[data['condition'] == 'in progress'].index)
    data = data.reset_index(drop=True)
    print(f'drop后 数据集数量{len(data)}')
    return data


def OLS(X, y):
    X = sm.add_constant(X)
    est = sm.OLS(y, X).fit()
    print(est.summary())
    print(est.params)


if __name__ == '__main__':
    # SETTINGS --------------------- #
    data_path = '../数据预处理/data.csv'
    encodeing = 'utf-8-sig'
    # ------------------------------ #
    X, y = load_data(data_path, encodeing)
    OLS(X, y)
