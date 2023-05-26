import requests
from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError
import pandas as pd
import time


def connect(pg):
    url = 'https://api.makuake.com/v2/projects?page={}&per_page=100'.format(pg)
    headers = {
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/107.0.0.0 Safari/537.36 Edg/107.0.1418.56',
        'cookie': '_gid=GA1.2.68241266.1670556064; UA-42337127-11=GA1.2.1859704258.1670556064; UA-42337127-11_gid=GA1.2.797236667.1670556064; _ts_yjad=1670556064827; _fbp=fb.1.1670556065045.1459711310; FPLC=DWQLhtzSjhkMptko48PDbxie7SK+Qjo5t0HwnEJHTfathh0bRbxsjl7ArUjCcPRXv22Hl/KIfkUwKASe2LtJpWPaJY80TdzaEbG/DT0rDbXNJ3P6swJWDkLLlnPeXA==; FPID=FPID2.2.G5anOU3h9zdySlqV6UrNlgv/0WtpsIVIuHleEomeFbg=.1670556064; FPAU=1.2.1656490350.1670556065; _gcl_au=1.1.928055942.1670556067; _tt_enable_cookie=1; _ttp=7yQEIZs68li9RoSbYHqi86Puqjf; _ga_YP1P8FF63J=GS1.1.1670574438.2.1.1670574440.0.0.0; cto_bundle=Oi_R-19XNDZieGpSNEoyRjc5ZU5ucGNUUmtLSk1MVXBmb2xCSnBPczUwaDQwWnhoaGxpaFRpNDdoQ3JvdVZXRnRDWFVoY3ZZbkV4bDZhckglMkZERU9FWVZkb1hITmVmRUVKdktLYUZnSGtEOW5hTDRLUzU4QTA5Wlk5VDNwUE9XamRnbCUyQkRuSHclMkJqdUt5ckZrMUklMkJ6dmVmdjgzZyUzRCUzRA; _ga=GA1.2.1859704258.1670556064; _ga_J9KZGB7P74=GS1.1.1670574438.2.1.1670574445.53.0.0; _ga_SFR1WSTL43=GS1.1.1670574438.2.1.1670574445.53.0.0; _td=7517fa25-9223-44ce-8a30-38a2c57d2f72'
    }
    response = requests.get(url, headers)
    return response.json()


def saveMongoDB(data):
    client = MongoClient(db_path)
    db = client[db_name]
    tb = db[tb_name]
    tb.insert_one(data)
    print('save to mongo suc')


def parse(data):
    data = data['projects']
    for item in data:
        item = {**item, **{'_id': item['id']}}
        print(item)
        try:
            saveMongoDB(item)
        except DuplicateKeyError:
            print(Exception)


if __name__ == '__main__':
    # ------------------------------------------ #
    # SETTINGS SAVE
    json_path = '主播id_dy.csv'
    db_path = 'mongodb://localhost:27017'
    db_name = 'crowdFunding'
    tb_name = 'makuake_basic_info'
    # ------------------------------------------ #
    for i in range(1, 298):
        response = connect(i)
        parse(data=response)
        time.sleep(10)
