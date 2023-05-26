import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Lasso, LinearRegression
from sklearn.metrics import accuracy_score, classification_report


def load_data(data_path, test, rg=False):
    data = pd.read_csv(data_path, encoding=encoding)
    # print(data.columns)
    data = drop(data)
    data['summary_text'] = data['summary_text'].map(lambda x: '' if pd.isna(x) else x)
    data = data.sample(frac=1, random_state=12, replace=False).reset_index(drop=True)
    if test:
        data = data[:1000]
    train_num = int(len(data) * 0.8)
    Xtrain = data[:train_num]
    Xtest = data[train_num:].reset_index(drop=True)
    if rg:
        ytrain = Xtrain['percent'].to_numpy() / 100
        ytest = Xtest['percent'].to_numpy() / 100
    else:
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
    # data = data.drop(data[data['type_'] == '販売中'].index)
    data['type_'] = data['type_'].map(lambda x: 'All in' if x == '販売中' else x)
    data = data.drop(data[data['type_'] == 'このプロジェクトは目標金額の達成に関わらず、応援購入を申し込んだ時点で決済が確定します。'].index)
    encoder = LabelEncoder()
    # data['category'] = encoder.fit_transform(data['category'])
    # data['location'] = encoder.fit_transform(data['location'])
    # data['type_'] = encoder.fit_transform(data['type_'])
    for col in ['category', 'location', 'type_']:
        # print(pd.get_dummies(data[col], prefix=col).columns[0])
        dummy_variable = pd.get_dummies(data[col], prefix=col, drop_first=True)
        data.drop(columns=col, inplace=True)
        data = pd.concat([data, dummy_variable], axis=1)
    data = data.reset_index(drop=True)

    print(f'drop后 数据集数量{len(data)}')
    return data


def get_numeric(data):
    data = data.drop(labels=['id_', 'collected_money', 'collected_supporter', 'percent',
                             'title', 'summary_text', 'main_text', 'thumb_ups', 'is_end', 'remain_time',
                             'condition', 'curr_time', 'end_at', 'comment_num', 'is_store_opening', 'is_new',
                             'is_new_store_opening'], axis=1)
    for column in ['target_amount', 'activity_num', 'min_price', 'max_price', 'avg_price']:
        data[column] = data[column].map(lambda x: np.log(x + 1))

    return data


def logistic_cls(Xtrain, Xtest, ytrain, ytest):
    print(Xtrain.shape)
    lg = LogisticRegression()
    lg.fit(Xtrain, ytrain)
    print(len(lg.coef_[0]))
    print(np.where(abs(lg.coef_[0]) > 0.1))
    print(lg.intercept_)
    predict = lg.predict(Xtrain)
    print(accuracy_score(ytrain, predict))
    print(classification_report(ytrain, predict))
    predict = lg.predict(Xtest)
    print(accuracy_score(ytest, predict))
    print(classification_report(ytest, predict))
    plot_heat(lg.coef_, 21, 'rg.pdf')


def plot_lasso(Xtrain, ytrain):
    plt.cla()
    plt.figure()
    accs = []
    coefs = []
    crange = np.arange(0, 1.5, 1e-2)
    for i in tqdm(crange):
        y = ytrain.copy()
        lasso = Lasso(alpha=i, random_state=42, copy_X=True)
        lasso.fit(Xtrain, y)
        predict = lasso.predict(Xtrain)
        y[y >= 1] = 1
        y[y < 1] = 0
        predict[predict >= 1] = 1
        predict[predict < 1] = 0
        acc = accuracy_score(y, predict)
        coefs.append(lasso.coef_)
        accs.append(acc)
        print(acc)
        # print(classification_report(y,predict))
    coefs = np.array(coefs)
    print(coefs.shape)
    with plt.style.context(['seaborn']):
        for i in range(coefs.shape[1]):
            plt.plot(crange, coefs[:, i])
        plt.xlabel('lambda')
        plt.ylabel('coef')
        plt.savefig('解的路径.pdf', dpi=300)
        plt.close()
        plt.figure()
        # plt.show()
        plt.plot(crange, accs)
        plt.xlabel('lambda')
        plt.ylabel('acc')
        plt.savefig('lasoo_acc.pdf', dpi=300)
        # plt.show()


def lasso_rg(Xtrain, Xtest, ytrain, ytest):
    lasso = Lasso(alpha=0.6)
    # lasso = LinearRegression()
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
    print(len(np.where(lasso.coef_ != 0)[0]))
    print(np.where(lasso.coef_ != 0)[0])
    plot_heat(lasso.coef_, 21, 'lasso.pdf')


def plot_heat(array, col, name):
    array = array.reshape(-1)
    row = int(np.ceil(len(array) / col))
    append_num = row * col - len(array)
    array = np.concatenate([array, np.array([0 for i in range(append_num)])])
    array = array.reshape((len(array) + append_num) // col, col)
    plt.figure(figsize=(8, 2.5))
    plt.subplots_adjust(left=0.02, right=1.05)
    sns.heatmap(array, vmax=1, vmin=-1,
                cmap=sns.diverging_palette(h_neg=250, h_pos=12, s=50, l=50, as_cmap=True),
                square=True, cbar=True, xticklabels=False, yticklabels=False)
    plt.savefig(name, dpi=300)


if __name__ == '__main__':
    # ------------------------------------------ #
    # SETTINGS readscv
    data_path = '../数据预处理/data.csv'
    encoding = 'utf-8-sig'
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', 20)
    test = False
    # ------------------------------------------ #

    Xtrain, Xtest, ytrain, ytest = load_data(data_path, test)

    numeric_train = get_numeric(Xtrain)
    for step, col in enumerate(numeric_train.columns):
        print(step, col)
    print(numeric_train.columns)
    scaler = StandardScaler()
    numeric_train = scaler.fit_transform(numeric_train)

    numeric_test = get_numeric(Xtest)
    numeric_test = scaler.transform(numeric_test)

    logistic_cls(numeric_train, numeric_test, ytrain, ytest)

    Xtrain, Xtest, ytrain, ytest = load_data(data_path, test, rg=True)

    lasso_rg(numeric_train, numeric_test, ytrain.copy(), ytest)


    plot_lasso(numeric_train, ytrain)
