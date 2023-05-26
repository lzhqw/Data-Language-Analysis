from pymongo import MongoClient
import pandas as pd
import re
import time
from tqdm import tqdm
import MeCab
from sklearn.decomposition import PCA
import numpy as np
from sklearn.linear_model import Lasso


def string2numeric(string):
    string = ''.join(i for i in re.findall('\d+', string))
    if string == '':
        string = 0
    return int(string)


def bool2numeric(bool):
    return 1 if bool else 0


def time2numeric(datetime):
    datetime = datetime.replace(' ', 'T')
    datetime = time.strptime(datetime[:19], '%Y-%m-%dT%H:%M:%S')
    timestamp = int(time.mktime(datetime) - 3600)
    return timestamp


def get_price(choice_list):
    price_list = []
    for choice in choice_list:
        price_list.append(string2numeric(choice['price']))
    return min(price_list), max(price_list), sum(price_list) / len(price_list)


def Conditon(percent, is_end):
    if is_end:
        if percent >= 100:
            return 'success'
        else:
            return 'failed'
    else:
        return 'in progress'


def get_type(string):
    if re.search('All in', string):
        string = 'All in'
    return string


def parse(item):
    id_ = item['id']
    collected_money = string2numeric(item['collected_money'])
    collected_supporter = string2numeric(item['collected_supporter'])
    percent = item['percent']
    title = item['title']
    title_length = len(title)
    is_new = bool2numeric(item['is_new'])
    is_store_opening = bool2numeric(item['is_store_opening'])
    has_target_money = bool2numeric(item['has_target_money'])
    has_expiration = bool2numeric(item['has_expiration'])
    is_accepting_support = bool2numeric(item['is_accepting_support'])
    hide_collected_money = bool2numeric(item['hide_collected_money'])
    is_new_store_opening = bool2numeric(item['is_new_store_opening'])
    summary_text = item['summary_text']
    if summary_text is None:
        summary_text = ''
    summary_text = summary_text.replace('\n', '')
    summary_text_length = len(summary_text)
    main_text = item['main_text'].replace('\n', '')
    main_text_length = len(main_text)
    target_amount = string2numeric(item['target_amount'])
    thumb_ups = string2numeric(item['thumb_ups'])
    activity_num = item['activity']['threads_count']['published']
    comment_num = len(item['comment_list'])
    start_at = time2numeric(item['start_at'])
    end_at = time2numeric(item['end_at'])
    curr_time = time2numeric(item['curr_time'])
    is_end = bool2numeric(curr_time - end_at > 0)
    remain_time = 0 if is_end else end_at - curr_time
    conditon = Conditon(percent, is_end)
    duration = end_at - start_at
    tag_num = len(item['tag'])
    category = item['category']
    location = item['location']
    type_ = get_type(item['type_'])
    choice_num = len(item['choice_list'])
    min_price, max_price, avg_price = get_price(item['choice_list'])
    img_num = len(item['img_left_href'])

    return {
        'id_': id_,
        'collected_money': collected_money,
        'collected_supporter': collected_supporter,
        'percent': percent,
        'title': title,
        'title_length': title_length,
        'is_new': is_new,
        'is_store_opening': is_store_opening,
        'has_target_money': has_target_money,
        'has_expiration': has_expiration,
        'is_accepting_support': is_accepting_support,
        'hide_collected_money': hide_collected_money,
        'is_new_store_opening': is_new_store_opening,
        'summary_text': summary_text,
        'summary_text_length': summary_text_length,
        'main_text': main_text,
        'main_text_length': main_text_length,
        'target_amount': target_amount,
        'thumb_ups': thumb_ups,
        'activity_num': activity_num,
        'comment_num': comment_num,
        'start_at': start_at,
        'end_at': end_at,
        'curr_time': curr_time,
        'is_end': is_end,
        'remain_time': remain_time,
        'condition': conditon,
        'duration': duration,
        'tag_num': tag_num,
        'category': category,
        'location': location,
        'type_': type_,
        'choice_num': choice_num,
        'min_price': min_price,
        'max_price': max_price,
        'avg_price': avg_price,
        'img_num': img_num
    }


if __name__ == '__main__':
    # ------------------------------------------ #
    # SETTINGS SAVE
    db_path = 'mongodb://localhost:27017'
    db_name = 'crowdFunding'
    tb_name = 'makuake_basic_info'
    tb_detail_name = 'makuake_detail_info'
    data_path = 'data.csv'
    encodeing = 'utf-8-sig'
    # ------------------------------------------ #

    # ------------------------------------------ #
    # 第一次预处理：从mongodb中提取有用的信息，并进行格式转换
    # ------------------------------------------ #
    client = MongoClient(db_path)
    db = client[db_name]
    tb = db[tb_detail_name]

    data = pd.DataFrame()
    for item in tqdm(tb.find()):
        data = pd.concat([data, pd.DataFrame([parse(item)])])
        data.reset_index(inplace=True, drop=True)
    data.to_csv(data_path, encoding=encodeing, index=False)
