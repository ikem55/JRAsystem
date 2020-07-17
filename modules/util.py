import os
import shutil
import urllib.request
import zipfile
import glob
import numpy as np
import pandas as pd
import pickle
import math

from sklearn.preprocessing import MinMaxScaler, StandardScaler
import category_encoders as ce
import bisect
from sklearn.preprocessing import LabelEncoder

def scale_df_for_fa(df, mmsc_columns, mmsc_dict_name, stdsc_columns, stdsc_dict_name, dict_folder):

    mmsc_filename = dict_folder + mmsc_dict_name + '.pkl'
    if os.path.exists(mmsc_filename):
        mmsc = load_dict(mmsc_dict_name, dict_folder)
        stdsc = load_dict(stdsc_dict_name, dict_folder)
    else:
        mmsc = MinMaxScaler()
        stdsc = StandardScaler()
        mmsc.fit(df[mmsc_columns])
        stdsc.fit(df[stdsc_columns])
        save_dict(mmsc, mmsc_dict_name, dict_folder)
        save_dict(stdsc, stdsc_dict_name, dict_folder)
    mmsc_norm = pd.DataFrame(mmsc.transform(df[mmsc_columns]), columns=mmsc_columns)
    stdsc_norm = pd.DataFrame(stdsc.transform(df[stdsc_columns]), columns=stdsc_columns)
    other_df = df.drop(mmsc_columns, axis=1)
    other_df = other_df.drop(stdsc_columns, axis=1)
    norm_df = pd.concat([mmsc_norm, stdsc_norm, other_df], axis=1)
    return norm_df

def save_dict(dict, dict_name, dict_folder):
    """ エンコードした辞書を保存する

    :param dict dict: エンコード辞書
    :param str dict_name: 辞書名
    """
    if not os.path.exists(dict_folder):
        os.makedirs(dict_folder)
    with open(dict_folder + dict_name + '.pkl', 'wb') as f:
        print("save dict:" + dict_folder + dict_name)
        pickle.dump(dict, f)

def load_dict(dict_name, dict_folder):
    """ エンコードした辞書を呼び出す

    :param str dict_name: 辞書名
    :return: encodier
    """
    with open(dict_folder + dict_name + '.pkl', 'rb') as f:
        return pickle.load(f)

def onehot_eoncoding(df, oh_columns, dict_name, dict_folder):
    """ dfを指定したEncoderでdataframeに置き換える

    :param dataframe df_list_oh: エンコードしたいデータフレーム
    :param str dict_name: 辞書の名前
    :param str encode_type: エンコードのタイプ。OneHotEncoder or HashingEncoder
    :param int num: HashingEncoder時のコンポーネント数
    :return: dataframe
    """
    encoder = ce.OneHotEncoder(cols=oh_columns, handle_unknown='impute')
    filename = dict_folder + dict_name + '.pkl'
    oh_df = df[oh_columns].astype('str')
    if os.path.exists(filename):
        ce_fit = load_dict(dict_name, dict_folder)
    else:
        ce_fit = encoder.fit(oh_df)
        save_dict(ce_fit, dict_name, dict_folder)
    df_ce = ce_fit.transform(oh_df)
    other_df = df.drop(oh_columns, axis=1)
    return_df = pd.concat([other_df, df_ce], axis=1)
    return return_df

def hash_eoncoding(df, oh_columns, num, dict_name, dict_folder):
    """ dfを指定したEncoderでdataframeに置き換える

    :param dataframe df_list_oh: エンコードしたいデータフレーム
    :param str dict_name: 辞書の名前
    :param str encode_type: エンコードのタイプ。OneHotEncoder or HashingEncoder
    :param int num: HashingEncoder時のコンポーネント数
    :return: dataframe
    """
    encoder = ce.HashingEncoder(cols=oh_columns, n_components=num)
    filename = dict_folder + dict_name + '.pkl'
    oh_df = df[oh_columns].astype('str')
    if os.path.exists(filename):
        ce_fit = load_dict(dict_name, dict_folder)
    else:
        ce_fit = encoder.fit(oh_df)
        save_dict(ce_fit, dict_name, dict_folder)
    #print(dir(ce_fit))
    df_ce = ce_fit.transform(oh_df)
    df_ce.columns = [dict_name + '_' + str(x) for x in list(range(num))]
    other_df = df.drop(oh_columns, axis=1)
    return_df = pd.concat([other_df, df_ce.astype(str)], axis=1)
    return return_df



def setup_basic_auth(url, id, pw):
    """ ベーシック認証のあるURLにアクセス

    :param str url:
    """
    password_mgr = urllib.request.HTTPPasswordMgrWithDefaultRealm()
    password_mgr.add_password(
        realm=None, uri = url, user = id, passwd = pw
    )
    auth_handler = urllib.request.HTTPBasicAuthHandler(password_mgr)
    opener = urllib.request.build_opener(auth_handler)
    urllib.request.install_opener(opener)


def unzip_file(filename, download_path, archive_path):
    """  ZIPファイルを解凍する。解凍後のZIPファイルはarvhice_pathに移動

    :param str filename:
    :param str download_path:
    :param str archive_path:
    """
    with zipfile.ZipFile(download_path + filename) as existing_zip:
        existing_zip.extractall(download_path)
    shutil.move(download_path + filename, archive_path + filename)


def get_downloaded_list(type, archive_path):
    """ ARCHIVE_FOLDERに存在するTYPEに対してのダウンロード済みリストを取得する

    :param str type:
    :param str archive_path:
    :return:
    """
    os.chdir(archive_path)
    filelist = glob.glob(type + "*")
    return filelist


def get_file_list(filetype, folder_path):
    """ 該当フォルダに存在するTYPE毎のファイル一覧を取得する

    :param str filetype:
    :param str folder_path:
    :return:
    """
    os.chdir(folder_path)
    filelist = glob.glob(filetype + "*")
    return filelist


def get_latest_file(finish_path):
    """ 最新のダウンロードファイルを取得する（KYIで判断）

    :param str finish_path:
    :return:
    """
    os.chdir(finish_path)
    latest_file = glob.glob("KYI*")
    return sorted(latest_file)[-1]


def int_null(str):
    """ 文字列をintにし、空白の場合はNoneに変換する

    :param str str:
    :return:
    """
    cnt = len(str)
    empty_val = ''
    for i in range(cnt):
        empty_val += ' '
    if str == empty_val:
        return None
    else:
        return int(str)


def float_null(str):
    """ 文字列をfloatにし、空白の場合はNoneに変換する

    :param str str:
    :return:
    """
    cnt = len(str)
    empty_val = ''
    for i in range(cnt):
        empty_val += ' '
    if str == empty_val:
        return None
    else:
        return float(str)


def int_bataiju_zogen(str):
    """ 文字列の馬体重増減を数字に変換する

    :param str str:
    :return:
    """
    fugo = str[0:1]
    if fugo == "+":
        return int(str[1:3])
    elif fugo == "-":
        return int(str[1:3])*(-1)
    else:
        return 0


def convert_time(str):
    """ 文字列のタイム(ex.1578)を秒数(ex.1178)に変換する

    :param str str:
    :return:
    """
    min = str[0:1]
    if min != ' ':
        return int(min) * 600 + int(str[1:4])
    else:
        return int_null(str[1:4])


def int_haito(str):
    """ 文字列の配当を数字に変換する。空白の場合は0にする

    :param str str:
    :return:
    """
    cnt = len(str)
    empty_val = ''
    for i in range(cnt):
        empty_val += ' '
    if str == empty_val:
        return 0
    else:
        return int(str)


def get_kaisai_date(filename):
    """ ファイル名から開催年月日を取得する(ex.20181118)

    :param str filename:
    :return:
    """
    return '20' + filename[3:9]


def move_file(file, folder_path):
    """ ファイルをフォルダに移動する

    :param str file:
    :param str folder_path:
    :return:
    """
    shutil.move(file, folder_path)


def escape_create_text(text):
    """ CREATE SQL文の生成時に含まれる記号をエスケープする

    :param str text:
    :return:
    """
    new_text = text.replace('%', '')
    return new_text


def convert_python_to_sql_type(dtype):
    """ Pythonのデータ型(ex.float)からSQL serverのデータ型(ex.real)に変換する

    :param str dtype:
    :return:
    """
    if dtype == 'float64':
        return 'real'
    elif dtype == 'object':
        return 'nvarchar(10)'
    elif dtype == 'int64':
        return 'real'


def convert_date_to_str(date):
    """ yyyy/MM/ddの文字列をyyyyMMddの文字型に変換する

    :param str date: date yyyy/MM/dd
    :return: str yyyyMMdd
    """
    return date.replace('/', '')


def convert_str_to_date(date):
    """ yyyyMMddの文字列をyyyy/MM/ddの文字型に変換する

    :param str date: date yyyyMMdd
    :return: str yyyy/MM/dd
    """
    return date[0:4] + '/' + date[4:6] + '/' + date[6:8]


def check_df(df):
    """ 与えられたdfの値チェック

    :param dataframe df:
    """
    pd.set_option('display.max_columns', 3000)
    pd.set_option('display.max_rows', 3000)

    print("------------ データサンプル ----------------------------")
    print(df.iloc[0])
    print(df.shape)

    print("----------- データ統計量確認 ---------------")
    print(df.describe())

    print("----------- Null個数確認 ---------------")
    df_null = df.isnull().sum()
    print(df_null[df_null != 0])

    print("----------- infinity存在確認 ---------------")
    temp_df_inf = df.replace([np.inf, -np.inf], np.nan).isnull().sum()
    df_inf = temp_df_inf - df_null
    print(df_inf[df_inf != 0])

    print("----------- 重複データ確認 ---------------")
    # print(df[df[["RACE_KEY","UMABAN"]].duplicated()].shape)

    print("----------- データ型確認 ---------------")
    print("object型")
    print(df.select_dtypes(include=object).columns)
    # print(df.select_dtypes(include=object).columns.tolist())
    print("int型")
    print(df.select_dtypes(include=int).columns)
    print("float型")
    print(df.select_dtypes(include=float).columns)
    print("datetime型")
    print(df.select_dtypes(include='datetime').columns)


def trans_baken_type(type):
    if type == 1:
        return '単勝　'
    elif type == 2:
        return '複勝　'
    elif type == 3:
        return '枠連　'
    elif type == 4:
        return '枠単　'
    elif type == 5:
        return '馬連　'
    elif type == 6:
        return '馬単　'
    elif type == 7:
        return 'ワイド'
    elif type == 8:
        return '三連複'
    elif type == 9:
        return '三連単'
    elif type == 0:
        return '合計　'

def label_encoding(sr, dict_name, dict_folder):
    """ srに対してlabel encodingした結果を返す

    :param series sr: エンコードしたいSeries
    :param str dict_name: 辞書の名前
    :param str dict_folder: 辞書のフォルダ
    :return: Series
    """
    le = LabelEncoder()
    dict_name = "le_" + dict_name
    filename = dict_folder + dict_name + '.pkl'
    if os.path.exists(filename):
        le = load_dict(dict_name, dict_folder)
    else:
        le = le.fit(sr.astype('str'))
        save_dict(le, dict_name, dict_folder)
    sr = sr.map(lambda s: 'other' if s not in le.classes_ else s)
    le_classes = le.classes_.tolist()
    bisect.insort_left(le_classes, 'other')
    le.classes_ = le_classes
    sr_ce = le.transform(sr.astype('str'))
    return sr_ce

def create_folder(folder_path):
    try:
        os.makedirs(folder_path)
    except FileExistsError:
        pass

def encode_rap_type(type):
    if type == "一貫":
        return "0"
    elif type == "L4加速":
        return "1"
    elif type == "L3加速":
        return "2"
    elif type == "L2加速":
        return "3"
    elif type == "L1加速":
        return "4"
    elif type == "L4失速":
        return "5"
    elif type == "L3失速":
        return "6"
    elif type == "L2失速":
        return "7"
    elif type == "L1失速":
        return "8"
    else:
        return "9"

def encode_zengo_bias(num):
    if num < -3:
        return "0" #"超前有利"
    elif num < -1.2:
        return "1" #"前有利"
    elif num > 3:
        return "4" ##"超後有利"
    elif num > 1.2:
        return "3" #"後有利"
    else:
        return "2" #"フラット"

def encode_uchisoto_bias(num):
    if num < -1.8:
        return "0" #"超内有利"
    elif num < -0.6:
        return "1" #"内有利"
    elif num > 1.8:
        return "4" #"超外有利"
    elif num > 0.6:
        return "3" #"外有利"
    else:
        return "2" #"フラット"

def encode_race_pace(val):
    if val == "11": return "1"
    elif val == "12": return "2"
    elif val == "13": return "3"
    elif val == "21": return "4"
    elif val == "22": return "5"
    elif val == "23": return "6"
    elif val == "31": return "7"
    elif val == "32": return "8"
    elif val == "33": return "9"
    else: return "0"

def convert_ls_hyouka(cd):
    dict = {"A":3, "B":2, "C":1,  "0": 0}
    return dict.get(cd, 0)

def check_df_value(df, method_name):
    print(f"----- {method_name} -------")
    print(df.iloc[0])
    print(df.dtypes)
    check_columns = df.dtypes[df.dtypes == "object"].index.tolist()
    for col in check_columns:
        print(col)
        print(df[col].unique())