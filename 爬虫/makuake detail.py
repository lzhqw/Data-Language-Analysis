import requests
from bs4 import BeautifulSoup
from selenium import webdriver
import time
import re
from pymongo import MongoClient
import numpy as np

def connect(url,key):
    proxyMeta = 'http://'+proxy['data']['list'][key]
    # proxyHost = proxy['data'][key]['ip']
    # proxyPort = proxy['data'][key]['port']
    # # 非账号密码验证
    # proxyMeta = "http://%(host)s:%(port)s" % {
    #
    #     "host": proxyHost,
    #     "port": proxyPort,
    # }

    proxies = {
        "http": proxyMeta,
        "https": proxyMeta
    }
    headers = {
        'user-agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/107.0.0.0 Safari/537.36 Edg/107.0.1418.56',
        'cookie':'_gid=GA1.2.68241266.1670556064; UA-42337127-11=GA1.2.1859704258.1670556064; UA-42337127-11_gid=GA1.2.797236667.1670556064; _ts_yjad=1670556064827; _fbp=fb.1.1670556065045.1459711310; FPLC=DWQLhtzSjhkMptko48PDbxie7SK+Qjo5t0HwnEJHTfathh0bRbxsjl7ArUjCcPRXv22Hl/KIfkUwKASe2LtJpWPaJY80TdzaEbG/DT0rDbXNJ3P6swJWDkLLlnPeXA==; FPID=FPID2.2.G5anOU3h9zdySlqV6UrNlgv/0WtpsIVIuHleEomeFbg=.1670556064; FPAU=1.2.1656490350.1670556065; _gcl_au=1.1.928055942.1670556067; _tt_enable_cookie=1; _ttp=7yQEIZs68li9RoSbYHqi86Puqjf; _ga_YP1P8FF63J=GS1.1.1670574438.2.1.1670574440.0.0.0; cto_bundle=Oi_R-19XNDZieGpSNEoyRjc5ZU5ucGNUUmtLSk1MVXBmb2xCSnBPczUwaDQwWnhoaGxpaFRpNDdoQ3JvdVZXRnRDWFVoY3ZZbkV4bDZhckglMkZERU9FWVZkb1hITmVmRUVKdktLYUZnSGtEOW5hTDRLUzU4QTA5Wlk5VDNwUE9XamRnbCUyQkRuSHclMkJqdUt5ckZrMUklMkJ6dmVmdjgzZyUzRCUzRA; _ga=GA1.2.1859704258.1670556064; _ga_J9KZGB7P74=GS1.1.1670574438.2.1.1670574445.53.0.0; _ga_SFR1WSTL43=GS1.1.1670574438.2.1.1670574445.53.0.0; _td=7517fa25-9223-44ce-8a30-38a2c57d2f72',
        'Connection': 'close'
    }
    # print(proxies)
    response = requests.get(url,headers,proxies=proxies)
    if response.status_code==200:
        return response
    else:
        print(response.status_code,response.text)

def connect2(url,key):
    proxyMeta = 'http://'+proxy['data']['list'][key]

    proxies = {
        "http": proxyMeta,
        "https": proxyMeta
    }
    headers = {
        'user-agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/107.0.0.0 Safari/537.36 Edg/107.0.1418.56',
        'cookie': 'sb=tLsLYv0akVPt6ftavHV_xxUo; datr=tLsLYqwli-vz8LILfE3GabWt; c_user=100078298060296; xs=4:cv8QGEReNh0Kig:2:1644936264:-1:-1::AcW55HlhwCwc131Z1agk4TP1hUxRQdXx3fKFAFqHKQ; fr=0661H4Zno6AV0wEtB.AWUIBjXxd_mWbaCSmiw2p5hIiyg.BjThvu.Eo.AAA.0.0.BjThvu.AWXjq1Uvy7U; locale=zh_CN; dpr=1.149999976158142',
        'Connection': 'close'
    }
    # print(proxies)
    response = requests.get(url,headers,proxies=proxies)
    if response.status_code==200:
        return response
    else:
        print(response.status_code,response.text)

def support_parse(data):
    data = data['threads']
    comment_list = []
    for item in data:
        thread_url_no_encode = item['thread_url_no_encode']
        user_name = item['user_name']
        heart_cnt = item['heart_cnt']
        comment_cnt = item['comment_cnt']
        fuzzy_time = item['fuzzy_time']
        message = (item['message'].encode('utf_8_sig')).decode('utf_8_sig')
        comment_list.append({'thread_url_no_encode':thread_url_no_encode,
                             'user_name':user_name,
                             'heart_cnt':heart_cnt,
                             'comment_cnt':comment_cnt,
                             'fuzzy_time':fuzzy_time,
                             'message':message})
    return comment_list

def parse(Url,id_):
    # -------------------------------------------------- #
    # step 1. 获取facebook 点赞数
    # -------------------------------------------------- #
    item_name = re.search('t/.+/', Url).group()[2:-1]
    url_facebook = 'https://www.facebook.com/v2.9/plugins/like.php?app_id=1407034392847669&channel=https%3A%2F%2Fstaticxx.facebook.com%2Fx%2Fconnect%2Fxd_arbiter%2F%3Fversion%3D46%23cb%3Df2de8671ff926c4%26domain%3Dwww.makuake.com%26is_canvas%3Dfalse%26origin%3Dhttps%253A%252F%252Fwww.makuake.com%252Ff39041ffa9d25b4%26relation%3Dparent.parent&container_width=0&href=https%3A%2F%2Fwww.makuake.com%2Fproject%2F{}%2F&layout=button_count&locale=ja_JP&sdk=joey&send=false&show_faces=false&width=111'.format(
        item_name)

    print(url_facebook)
    response_facebook = connect2(url_facebook, np.random.randint(0,199))
    # time.sleep(1)
    soup_facebook = BeautifulSoup(response_facebook.text, 'html.parser')
    thumb_ups = soup_facebook.select('button[type="submit"] > div > span')[2].text
    print('\033[0;32m-- facebook done ! --\033[0m')
    # -------------------------------------------------- #
    # step 2. 评论内容及数量
    # -------------------------------------------------- #
    comment_list = []
    cnt = 1
    url_comment = 'https://www.makuake.com/api/com/supporter/{}/{}/'.format(item_name, cnt)
    ccnt = 0
    while True:
        ccnt+=1
        try:
            response_comment = connect(url_comment, np.random.randint(0,199)).json()
            break
        except Exception as e:
            if ccnt>20: raise Exception(e)
        print(ccnt, end=',')
    # print(response_comment)
    # time.sleep(1)
    comment_list.extend(support_parse(response_comment))
    while response_comment['exists_next']:
        # time.sleep(1)
        cnt += 1
        print(cnt)
        url_comment = 'https://www.makuake.com/api/com/supporter/{}/{}/'.format(item_name, cnt)
        ccnt = 0
        while True:
            ccnt+=1
            # time.sleep(0.5)
            try:
                response_comment = connect(url_comment, np.random.randint(0, 199)).json()
                break
            except Exception as e:
                if ccnt > 50:raise Exception(e)
            print(ccnt, end=',')
        # print(response_comment)
        comment_list.extend(support_parse(response_comment))
    # print(comment_list)
    comment_num = len(comment_list)
    print('\033[0;32m-- comment done ! --\033[0m')

    # -------------------------------------------------- #
    # step 3. 获取location 点赞人数 活动数
    # -------------------------------------------------- #
    activity_url = 'https://www.makuake.com/api/com/owner/{}/1/?draft=false'.format(item_name)
    cnt = 0
    while True:
        cnt +=1
        try:
            response_activity = connect(activity_url,np.random.randint(0, 199)).json()
            break
        except Exception as e:
            if cnt>20: raise Exception(e)
        print(cnt, end=',')
    # print(response_activity)
    print('\033[0;32m-- activity done ! --\033[0m')
    # -------------------------------------------------- #
    # step 5. 重新获取金额与支持者人数
    # -------------------------------------------------- #
    url_investment = 'https://www.makuake.com/api/projects/{}/investment-info?_='.format(id_)
    cnt = 0
    while True:
        cnt+=1
        try:
            response_invest = connect(url_investment, np.random.randint(0, 199)).json()
            break
        except Exception as e:
            if cnt>20: raise Exception(e)
        print(cnt, end=',')
    # print(response_invest)
    current_amount = response_invest['collected_money']
    supporters = response_invest['collected_supporter']
    percent = response_invest['percent']
    projects_returns = response_invest['projects_returns']
    print('\033[0;32m-- investment done! --\033[0m ')
    # -------------------------------------------------- #
    # step 3. 解析主页中的信息
    # -------------------------------------------------- #
    print(Url)
    cnt = 0
    while True:
        cnt+=1
        try:
            response = connect(Url, np.random.randint(0, 199))
            break
        except Exception as e:
            if cnt > 20: raise Exception(e)
        print(cnt,end=',')
    soup = BeautifulSoup(response.text,'html.parser')
    leftContent = soup.select('div#leftContent')[0]
    rightContent = soup.select('div#rightContent')[0]
    leftContentMain = soup.select('div#leftContentMain')[0]

    # 摘要中的文本
    try:
        summary_text = leftContent.select('div.summaryContainer')[0].get_text()
    except:
        summary_text = None
    # 主体中的文本
    main_text = leftContentMain.get_text()
    # 目标金额
    target_amount = soup.select('meta[property="note:target_amount"]')[0].get('content')
    # 开始与截止日期与当前日期
    end_at = soup.select('meta[property="note:end_at"]')[0].get('content')
    start_at = soup.select('meta[property="note:start_at"]')[0].get('content')
    curr_time = time.mktime(time.gmtime()) + 9 * 3600
    curr_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(curr_time))
    # tag
    tag = soup.select('meta[name="keywords"]')[0].get('content').split(',')
    # category
    category = soup.select('meta[property="note:category"]')[0].get('content')
    # location
    try:
        location = soup.select('span.location-name')[0].text
    except:
        location = None
    # type all in   all or nothing
    try:
        anoinfo = rightContent.select('section > div.anoInfo > p')[0].get_text()
        try:
            type_ = re.search('トは.+型',anoinfo).group()[2:-1]
        except:
            type_ = anoinfo
    except:
        anoinfo = rightContent.select('section > div.anoInfo')[0].get_text()
        if '販売中' in anoinfo:
            type_ = '販売中'
        else:
            raise Exception('type未知')
    # 图片数量
    img_left = leftContentMain.select('img')
    img_href = []
    for img in img_left:
        img_href.append(img.get('src'))

    # -------------------------------------------------- #
    # step 9. 不同选择的付费
    # -------------------------------------------------- #
    choice_list = []
    buy_choice = soup.select('div#return > section')
    for i,choice in enumerate(buy_choice):
        price = choice.select('h4.lefth4Right')[0].text
        choice_id = choice.select('a')[0].get('href').split('/')[-1]
        choice_describe = choice.select('a > p.detailTextRight')[0].text
        buy_num = projects_returns[i]['supported_amount']
        buy_left = projects_returns[i]['remain']
        choice_list.append({'price': price, 'choice_describe':choice_describe,
                            'buy_num': buy_num, 'buy_left': buy_left})
    print('\033[0;32m-- main done ! --\033[0m')




    return {
        'summary_text': summary_text,
        'main_text': main_text,
        'target_amount': target_amount,
        'collected_money': current_amount,
        'collected_supporter': supporters,
        'percent': percent,
        'thumb_ups': thumb_ups,
        'activity': response_activity,
        'start_at': start_at,
        'end_at': end_at,
        'curr_time': curr_time,
        'tag': tag,
        'category': category,
        'location': location,
        'type_': type_,
        'choice_list': choice_list,
        'img_left_href': img_href,
        'comment_list': comment_list
    }

def save_MongoDB(project, data):
    project.pop('time_left_label')
    new_data = {**project,**data}
    client = MongoClient(db_path)
    db = client[db_name]
    tb_detail = db[tb_detail_name]
    tb_detail.insert_one(new_data)
    print('\033[0;32msave to mongo suc\033[0m')



if __name__ == '__main__':
    # ------------------------------------------ #
    # SETTINGS SAVE
    db_path = 'mongodb://localhost:27017'
    db_name = 'crowdFunding'
    tb_name = 'makuake_basic_info'
    tb_detail_name = 'makuake_detail_info'
    # ------------------------------------------ #
    client = MongoClient(db_path)
    db = client[db_name]
    tb_base = db[tb_name]
    tb_detail = db[tb_detail_name]


    for project in tb_base.find():
        id_ = project['_id']
        url = project['url']
        # print(f'# --{id_}------------------------------------------')
        if tb_detail.find_one({'_id':id_}) or id_ == 29560:
            pass
            # print('已经爬取')
        else:
            print(f'# --{id_}------------------------------------------')
            try:
                # api = r'http://api.ipipgo.com/ip?cty=00&c=300&pt=1&ft=json&pat=\n&rep=1&key=4807a967&ts=3'
                api = r'https://api.smartproxy.cn/web_v1/ip/get-ip?app_key=50d241a32d2423789a6a9c0af3c5b0d3&pt=9&num=200&cc=&protocol=1&format=json&nr=%5Cr%5Cn'
                proxy = requests.get(api).json()
                print(len(proxy['data']['list']))
                # time.sleep(1)
                data = parse(url, id_)
                save_MongoDB(project,data)
            except Exception as e:
                print('\033[0;31m'+str(e)+'\033[0m')
            time.sleep(1)
    # url = 'https://www.makuake.com/project/nwojp6/'
    # response= connect(url)
    # data = parse(response,30409)
    # print(data)
