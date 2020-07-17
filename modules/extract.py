import pandas as pd
import numpy as np
import scipy
import math
import os
import datetime as dt
from collections import deque
import unicodedata
import shutil
import glob
import my_config as mc

class Extract(object):
    """
    データ抽出に関する共通モデル。データ取得については下位モデルで定義する。
    """
    mock_flag = False
    """ mockデータを使うかの判断に使用するフラグ。Trueの場合はMockデータを使う """
    mock_path = '../mock_data/'
    """ mockファイルが格納されているフォルダのパス """
    dict_path = mc.return_jrdb_path()
    jrdb_folder_path = dict_path + 'jrdb_data/'
    pred_folder_path = dict_path + 'pred/'
    race_df = pd.DataFrame()
    race_before_df = pd.DataFrame()
    raceuma_df = pd.DataFrame()
    raceuma_before_df = pd.DataFrame()
    haraimodoshi_df = pd.DataFrame()

    def __init__(self, start_date, end_date, mock_flag):
        self.start_date = start_date
        self.end_date = end_date
        if mock_flag:
            self.set_mock_path()
            self.mock_flag = mock_flag

    def set_mock_path(self):
        """ mock_flagをTrueにしてmockのパスを設定する。  """
        self.mock_path_race = self.mock_path + 'race.pkl'
        self.mock_path_race_before = self.mock_path + 'race_before.pkl'
        self.mock_path_raceuma = self.mock_path + 'raceuma.pkl'
        self.mock_path_raceuma_before = self.mock_path + 'raceuma_before.pkl'
        self.mock_path_chokuzen = self.mock_path + 'chokuzen.pkl'
        self.mock_path_zenso = self.mock_path + 'zenso.pkl'
        self.mock_path_horse = self.mock_path + 'horse.pkl'
        self.mock_path_kishu = self.mock_path + 'kishu.pkl'
        self.mock_path_chokyoshi = self.mock_path + 'chokyoshi.pkl'

    def create_mock_data(self):
        """ mock dataを作成する  """
        self.mock_flag = False
        race_df = self.get_race_table_base()
        race_before_df = self.get_race_before_table_base()
        raceuma_df = self.get_raceuma_table_base()
        raceuma_before_df = self.get_raceuma_before_table_base()
        chokuzen_df = self.get_raceuma_chokuzen_table_base()
        zenso_df = self.get_raceuma_zenso_table_base()
        horse_df = self.get_horse_table_base()
        # kishu_df = self.get_kishu_table_base()
        # chokyoshi_df = self.get_chokyoshi_table_base()

        self.set_mock_path()
        race_df.to_pickle(self.mock_path_race)
        race_before_df.to_pickle(self.mock_path_race_before)
        raceuma_df.to_pickle(self.mock_path_raceuma)
        raceuma_before_df.to_pickle(self.mock_path_raceuma_before)
        chokuzen_df.to_pickle(self.mock_path_chokuzen)
        zenso_df.to_pickle(self.mock_path_zenso)
        horse_df.to_pickle(self.mock_path_horse)
        horse_df.to_pickle(self.mock_path_horse)
        # kishu_df.to_pickle(self.mock_path_kishu)
        # chokyoshi_df.to_pickle(self.mock_path_chokyoshi)

    def get_race_table_base(self):
        """ レースに関するデータを取得する。SRファイルを読み込む """
        if self.mock_flag:
            self.race_df = pd.read_pickle(self.mock_path_race)
        else:
            if len(self.race_df.index) == 0:
                srb_df = self._get_type_df("SRB")
                srb_df = srb_df.drop(["KAISAI_KEY", "連続何日目", "芝種類", "草丈", "転圧", "凍結防止剤", "中間降水量", "１着算入賞金"], axis=1)
                if len(self.race_before_df.index) == 0:
                    self.race_before_df = self.get_race_before_table_base().copy()
                self.race_df = pd.merge(self.race_before_df, srb_df, on=["RACE_KEY", "target_date"])
        return self.race_df

    def get_race_before_table_base(self):
        """ レースに関するデータを取得する。結果データは含まない。BA, KAファイルを読み込む """
        if self.mock_flag:
            self.race_before_df = pd.read_pickle(self.mock_path_race_before)
        else:
            if len(self.race_before_df.index) == 0:
                bac_df = self._get_type_df("BAC")
                if bac_df.empty:
                    return pd.DataFrame()
                kab_df = self._get_type_df("KAB")
                kab_df = kab_df.drop(["NENGAPPI", "開催区分", "データ区分"], axis=1)
                race_before_df = pd.merge(bac_df, kab_df, on=["KAISAI_KEY", "target_date"])
                self.race_before_df = race_before_df.drop_duplicates(subset='RACE_KEY', keep='last')
        return self.race_before_df

    def get_raceuma_table_base(self):
        """ レースに関するデータを取得する。SE,SKファイルを読み込む"""
        if self.mock_flag:
            self.raceuma_df = pd.read_pickle(self.mock_path_raceuma)
        else:
            if len(self.raceuma_df.index) == 0:
                sed_df = self._get_type_df("SED")
                if sed_df.empty:
                    return pd.DataFrame()
                sed_df = sed_df.drop(["血統登録番号", "NENGAPPI", "クラスコード", "騎手名", "騎手コード", "調教師名", "調教師コード", "収得賞金", "馬名"],
                                     axis=1)
                skb_df = self._get_type_df("SKB")
                skb_df = skb_df.drop(["血統登録番号", "NENGAPPI"], axis=1)
                if len(self.raceuma_before_df.index) == 0:
                    self.raceuma_before_df = self.get_raceuma_before_table_base()
                raceuma_df = pd.merge(self.raceuma_before_df, sed_df, on=["RACE_KEY", "UMABAN", "target_date"])
                self.raceuma_df = pd.merge(raceuma_df, skb_df,
                                           on=["RACE_KEY", "UMABAN", "target_date", "KYOSO_RESULT_KEY"])
        return self.raceuma_df

    def get_raceuma_before_table_base(self):
        """ レースに関するデータを取得する。結果データは含まない。KY、JO、CH、CY,KKファイルを読み込む """
        if self.mock_flag:
            self.raceuma_before_df = pd.read_pickle(self.mock_path_raceuma_before)
        else:
            if len(self.raceuma_before_df.index) == 0:
                kyi_df = self._get_type_df("KYI")
                if kyi_df.empty:
                    return pd.DataFrame()
                joa_df = self._get_type_df("JOA")
                joa_df = joa_df.drop(["血統登録番号", "基準オッズ", "基準複勝オッズ"], axis=1)
                cha_df = self._get_type_df("CHA")
                cyb_df = self._get_type_df("CYB")
                cyb_df = cyb_df.drop(["追切指数"], axis=1)
                kka_df = self._get_type_df("KKA")
                raceuma_df = pd.merge(kyi_df, joa_df, on=["RACE_KEY", "UMABAN", "target_date"])
                raceuma_df = pd.merge(raceuma_df, cha_df, on=["RACE_KEY", "UMABAN", "target_date"])
                raceuma_df = pd.merge(raceuma_df, cyb_df, on=["RACE_KEY", "UMABAN", "target_date"])
                self.raceuma_before_df = pd.merge(raceuma_df, kka_df, on=["RACE_KEY", "UMABAN", "target_date"])
        return self.raceuma_before_df

    def get_raceuma_chokuzen_table_base(self):
        """ レースに関する直前データを取得する。TYファイルを読み込む """
        if self.mock_flag:
            df = pd.read_pickle(self.mock_path_chokuzen)
        else:
            tyb_df = self._get_type_df("TYB")
            if tyb_df.empty:
                return pd.DataFrame()
            df = tyb_df.copy()
        return df

    def get_odds_df(self, type):
        """ オッズファイルを読み込む """
        if type == "単勝":
            df = self._get_type_df("OZ")
            df = df[["RACE_KEY", "UMABAN", "単勝オッズ", "target_date"]].copy()
        elif type == "複勝":
            df = self._get_type_df("OZ")
            df = df[["RACE_KEY", "UMABAN", "複勝オッズ", "target_date"]].copy()
        elif type == "馬連":
            df = self._get_type_df("OZ")
            df.drop(["単勝オッズ", "複勝オッズ"], axis=1, inplace=True)
        elif type == "三連複":
            df = self._get_type_df("OT")
        elif type == "ワイド":
            df = self._get_type_df("OW")
        elif type == "馬単":
            df = self._get_type_df("OU")
        return df

    def get_raceuma_zenso_table_base(self):
        """ レースに関する前走データを取得する。TYファイルを読み込む """
        if self.mock_flag:
            df = pd.read_pickle(self.mock_path_zenso)
        else:
            zed_df = self._get_type_df("ZED")
            if zed_df.empty:
                return pd.DataFrame()
            zkb_df = self._get_type_df("ZKB")
            zkb_df = zkb_df.drop(["RACE_KEY", "UMABAN", "血統登録番号", "NENGAPPI"], axis=1)
            df = pd.merge(zed_df, zkb_df, on=["KYOSO_RESULT_KEY", "target_date"])
            target_date_list = zed_df["NENGAPPI"].drop_duplicates()
            all_srb_df = pd.DataFrame()
            for date in target_date_list:
                srb_file_name = f"SRB{date[2:8]}.txt"
                if os.path.exists(self.jrdb_folder_path + "SRB/" + srb_file_name + ".pkl"):
                    srb_file_name = srb_file_name + ".pkl"
                srb_df = self.get_srb_df(srb_file_name)
                all_srb_df = all_srb_df.append(srb_df)
            df = pd.merge(df, all_srb_df.drop(["馬場状態", "target_date"], axis=1), on="RACE_KEY", how="left")
        return df

    def get_horse_table_base(self):
        """ 競走馬に関するデータを取得する。UKファイルを読み込む """
        if self.mock_flag:
            df = pd.read_pickle(self.mock_path_horse)
        else:
            ukc_df = self._get_type_df("UKC")
            df = ukc_df
        return df

    def get_kishu_table_base(self):
        """ 騎手に関するデータを取得する。KZファイルを読み込む """
        if self.mock_flag:
            df = pd.read_pickle(self.mock_path_kishu)
        else:
            filename = self.jrdb_jolder_path + "master/kz.txt"
        return df

    def get_chokyoshi_table_base(self):
        """ 調教師に関するデータを取得する。CZファイルを読み込む """
        if self.mock_flag:
            df = pd.read_pickle(self.mock_path_chokyoshi)
        else:
            filename = self.jrdb_jolder_path + "master/cz.txt"
        return df

    def get_haraimodoshi_table_base(self):
        """ 払い戻しに関するデータを取得する。HJファイルを読み込む """
        if self.mock_flag:
            self.haraimodoshi_df = pd.read_pickle(self.mock_path_haraimodoshi)
        else:
            # if len(self.haraimodoshi_df.index) == 0:
            self.haraimodoshi_df = self._get_type_df("HJC")
        return self.haraimodoshi_df

    def get_pred_df(self, model_version, target):
        """ 予測したtargetのデータを取得する """
        target_filelist = self._get_file_list_for_pred(model_version, target)
        df = pd.DataFrame()
        for filename in target_filelist:
            temp_df = pd.read_pickle(filename)
            df = pd.concat([df, temp_df])
        return df

    def _get_file_list_for_pred(self, folder, type):
        """ predで予測したファイルの対象リストを取得する"""
        folder_path = self.pred_folder_path + folder + "/" + type + "*.pickle"
        filelist = glob.glob(folder_path)
        file_df = pd.DataFrame({"filename": filelist})
        file_df.loc[:, "date"] = file_df["filename"].apply(lambda x: dt.datetime.strptime(x[-15:-7], '%Y%m%d'))
        target_filelist = file_df[(file_df["date"] >= self.start_date) & (file_df["date"] <= self.end_date)][
            "filename"].tolist()
        return sorted(target_filelist)

    def _get_type_df(self, type):
        """ タイプで指定したデータを取得する """
        print(f"--------{type}----------")
        target_filelist = self._get_file_list(type)
        df = pd.DataFrame()
        for filename in target_filelist:
            # print(filename)
            if type == "BAC":
                temp_df = self.get_bac_df(filename)
            elif type == "CHA":
                temp_df = self.get_cha_df(filename)
            elif type == "CYB":
                temp_df = self.get_cyb_df(filename)
            elif type == "HJC":
                temp_df = self.get_hjc_df(filename)
            elif type == "JOA":
                temp_df = self.get_joa_df(filename)
            elif type == "KAB":
                temp_df = self.get_kab_df(filename)
            elif type == "KKA":
                temp_df = self.get_kka_df(filename)
            elif type == "KYI":
                temp_df = self.get_kyi_df(filename)
            elif type == "OT":
                temp_df = self.get_ot_df(filename)
            elif type == "OZ":
                temp_df = self.get_oz_df(filename)
            elif type == "OU":
                temp_df = self.get_ou_df(filename)
            elif type == "OW":
                temp_df = self.get_ow_df(filename)
            elif type == "SED":
                temp_df = self.get_sed_df(filename, "SED")
            elif type == "SKB":
                temp_df = self.get_skb_df(filename, "SKB")
            elif type == "SRB":
                temp_df = self.get_srb_df(filename)
            elif type == "TYB":
                temp_df = self.get_tyb_df(filename)
            elif type == "UKC":
                temp_df = self.get_ukc_df(filename)
            elif type == "ZED":
                temp_df = self.get_sed_df(filename, "ZED")
            elif type == "ZKB":
                temp_df = self.get_skb_df(filename, "ZKB")
            else:
                temp_df = pd.DataFrame()
            df = pd.concat([df, temp_df])
        return df

    def _get_file_list(self, type):
        """ type(ex.SRA)で指定したファイルのリストを取得する """
        folder_path = self.jrdb_folder_path + type + "/"
        filelist = os.listdir(folder_path)
        file_df = pd.DataFrame({"filename": filelist})
#        start_date = dt.datetime.strptime(self.start_date, '%Y/%m/%d')
#        end_date = dt.datetime.strptime(self.end_date, '%Y/%m/%d')
        if type in ("OZ", "OT", "OW", "OU"):
            file_df.loc[:, "date"] = file_df["filename"].apply(lambda x: dt.datetime.strptime('20' + x[2:8], '%Y%m%d'))
        else:
            file_df.loc[:, "date"] = file_df["filename"].apply(lambda x: dt.datetime.strptime('20' + x[3:9], '%Y%m%d'))
        target_filelist = file_df[(file_df["date"] >= self.start_date) & (file_df["date"] <= self.end_date)][
            "filename"].tolist()
        return sorted(target_filelist)

    def get_bac_df(self, filename):
        names = ['RACE_KEY', 'KAISAI_KEY', 'NENGAPPI', '発走時間', '距離', '芝ダ障害コード', '右左', '内外', '種別', '条件', '記号', '重量',
                 'グレード', 'レース名', '回数', '頭数',
                 'コース', '開催区分', 'レース名短縮', 'レース名９文字', 'データ区分', '１着賞金', '２着賞金', '３着賞金', '４着賞金', '５着賞金', '１着算入賞金',
                 '２着算入賞金',
                 '馬券発売フラグ', 'WIN5フラグ', 'target_date']
        if filename[-3:] == "pkl":
            df = pd.read_pickle(self.jrdb_folder_path + 'BAC/' + filename)
        else:
            print(filename)
            with open(self.jrdb_folder_path + 'BAC/' + filename, 'r', encoding="ms932") as fh:
                df = pd.DataFrame(index=[], columns=names)
                for line in fh:
                    new_line = self.replace_line(line)
                    sr = pd.Series([
                        new_line[0:8]  # RACE_KEY
                        , self.str_null(new_line[0:6])  # KAISAI_KEY
                        , self.str_null(new_line[8:16])  # NENGAPPI
                        , self.str_null(new_line[16:20])  # HASSO_JIKOKU
                        , self.int_0(new_line[20:24])  # KYORI
                        , self.str_null(new_line[24:25])  # SHIBA_DART
                        , self.str_null(new_line[25:26])  # MIGIHIDARI
                        , self.str_null(new_line[26:27])  # UCHISOTO
                        , self.str_null(new_line[27:29])  # SHUBETSU
                        , self.str_null(new_line[29:31])  # JOKEN
                        , self.str_null(new_line[31:34])  # KIGO
                        , self.str_null(new_line[34:35])  # JURYO
                        , self.str_null(new_line[35:36])  # GRADE
                        , self.str_null(new_line[36:86])  # .replace(" ", "")  # RACE_NAME
                        , self.str_null(new_line[86:94])  # .replace(" ", "")  # KAISU
                        , self.int_0(new_line[94:96])  # TOSU
                        , self.str_null(new_line[96:97])  # COURSE
                        , self.str_null(new_line[97:98])  # KAISAI_KUBUN
                        , self.str_null(new_line[98:106])  # .replace(" ", "")  # RACE_TANSHUKU
                        , self.str_null(new_line[106:124])  # .replace(" ", "")  # RACE_TANSHUKU_9
                        , self.str_null(new_line[124:125])  # DATA_KUBUN
                        , self.int_0(new_line[125:130])  # SHOKIN_1CK
                        , self.int_0(new_line[130:135])  # SHOKIN_2CK
                        , self.int_0(new_line[135:140])  # SHOKIN_3CK
                        , self.int_0(new_line[140:145])  # SHOKIN_4CK
                        , self.int_0(new_line[145:150])  # SHOKIN_5CK
                        , self.int_0(new_line[150:155])  # SANNYU_SHOKIN_1CK
                        , self.int_0(new_line[155:160])  # SANNYU_SHOKIN_2CK
                        , self.str_null(new_line[160:176])  # BAKEN_HATSUBAI_FLAG
                        , self.str_null(new_line[176:177])  # WIN5
                        , self.get_kaisai_date(filename)  # target_date
                    ], index=names)
                    df = df.append(sr, ignore_index=True)
            df = df.astype(
                {'頭数': 'int8', '１着賞金': 'int32', '２着賞金': 'int16', '３着賞金': 'int16', '４着賞金': 'int16', '５着賞金': 'int16',
                 '１着算入賞金': 'int16', '２着算入賞金': 'int16'})
            if len(df.query("データ区分 == '4'")) >= 10:
                df.to_pickle(self.jrdb_folder_path + 'BAC/' + filename + ".pkl")
            shutil.move(self.jrdb_folder_path + 'BAC/' + filename, self.jrdb_folder_path + 'backup/' + filename)
        return df

    def get_cha_df(self, filename):
        names = ['RACE_KEY', 'UMABAN', '調教曜日', '調教年月日', '調教回数', '調教コースコード', '追切種類', '追い状態', '乗り役', '調教Ｆ', 'テンＦ', '中間Ｆ',
                 '終いＦ', 'テンＦ指数', '中間Ｆ指数', '終いＦ指数', '追切指数', '併せ結果', '併せ追切種類', '併せ年齢', '併せクラス', 'target_date']
        if filename[-3:] == "pkl":
            df = pd.read_pickle(self.jrdb_folder_path + 'CHA/' + filename)
        else:
            print(filename)
            with open(self.jrdb_folder_path + 'CHA/' + filename, 'r', encoding="ms932") as fh:
                df = pd.DataFrame(index=[], columns=names)
                for line in fh:
                    new_line = self.replace_line(line)
                    sr = pd.Series([
                        new_line[0:8]  # RACE_KEY
                        , self.str_null(new_line[8:10])  # UMABAN
                        , self.str_null(new_line[10:12])  # YOBI
                        , self.str_null(new_line[12:20])  # CHOKYO_NENGAPPI
                        , self.str_null(new_line[20:21])  # KAISU
                        , self.str_null(new_line[21:23])  # CHOKYO_COURSE
                        , self.str_null(new_line[23:24])  # OIKIRI_SHURUI
                        , self.str_null(new_line[24:26])  # OI_JOTAI
                        , self.str_null(new_line[26:27])  # NORIYAKU
                        , self.int_0(new_line[27:28])  # CHOKYO_F
                        , self.int_0(new_line[28:31])  # TEN_F
                        , self.int_0(new_line[31:34])  # CHUKAN_F
                        , self.int_0(new_line[34:37])  # SHIMAI_F
                        , self.int_0(new_line[37:40])  # TEN_F_RECORD
                        , self.int_0(new_line[40:43])  # CHUKAN_F_RECORD
                        , self.int_0(new_line[43:46])  # SHIMAI_F_RECORD
                        , self.int_0(new_line[46:49])  # OIKIRI_RECORD
                        , self.str_null(new_line[49:50])  # AWASE_KEKKA
                        , self.str_null(new_line[50:51])  # AWASE_OIKIRI_SHURUI
                        , self.int_0(new_line[51:53])  # NENREI
                        , self.str_null(new_line[53:55])  # CLASS
                        , self.get_kaisai_date(filename)  # target_date
                    ], index=names)
                    df = df.append(sr, ignore_index=True)
            df = df.astype(
                {'調教Ｆ': 'int16', 'テンＦ': 'int16', '中間Ｆ': 'int16', '終いＦ': 'int16', 'テンＦ指数': 'int16', '中間Ｆ指数': 'int16',
                 '終いＦ指数': 'int16', '追切指数': 'int8'})
            if len(df.index) >= 60:
                df.to_pickle(self.jrdb_folder_path + 'CHA/' + filename + ".pkl")
            shutil.move(self.jrdb_folder_path + 'CHA/' + filename, self.jrdb_folder_path + 'backup/' + filename)
        return df

    def get_cyb_df(self, filename):
        names = ['RACE_KEY', 'UMABAN', '調教タイプ', '調教コース種別', '調教コース坂', '調教コースW', '調教コースダ', '調教コース芝', '調教コースプール',
                 '調教コース障害',
                 '調教コースポリ', '調教距離', '調教重点', '追切指数', '仕上指数', '調教量評価', '仕上指数変化', '調教コメント', 'コメント年月日', '調教評価',
                 'target_date']
        if filename[-3:] == "pkl":
            df = pd.read_pickle(self.jrdb_folder_path + 'CYB/' + filename)
        else:
            print(filename)
            with open(self.jrdb_folder_path + 'CYB/' + filename, 'r', encoding="ms932") as fh:
                df = pd.DataFrame(index=[], columns=names)
                for line in fh:
                    new_line = self.replace_line(line)
                    sr = pd.Series([
                        new_line[0:8]  # RACE_KEY
                        , self.str_null(new_line[8:10])  # UMABAN
                        , self.str_null(new_line[10:12])  # CHOKYO_TYPE
                        , self.str_null(new_line[12:13])  # CHOKYO_COURSE_SHUBETSU
                        , self.int_0(new_line[13:15])  # HANRO
                        , self.int_0(new_line[15:17])  # WOOD
                        , self.int_0(new_line[17:19])  # DART
                        , self.int_0(new_line[19:21])  # SHIBA
                        , self.int_0(new_line[21:23])  # POOL
                        , self.int_0(new_line[23:25])  # SHOBAI
                        , self.int_0(new_line[25:27])  # POLYTRACK
                        , self.str_null(new_line[27:28])  # CHOKYO_KYORI
                        , self.str_null(new_line[28:29])  # CHOKYO_JUTEN
                        , self.int_0(new_line[29:32])  # OIKIRI_RECORD
                        , self.int_0(new_line[32:35])  # SHIAGE_RECORD
                        , self.str_null(new_line[35:36])  # CHOKYORYO_HYOUKA
                        , self.str_null(new_line[36:37])  # SHIAGE_RECORD_HENKA
                        , self.str_null(new_line[37:77])  # CHOKYO_COMMENT
                        , self.str_null(new_line[77:85])  # COMENT_NENGAPPI
                        , self.str_null(new_line[85:86])  # CHOKYO_HYOKA
                        , self.get_kaisai_date(filename)  # target_date
                    ], index=names)
                    df = df.append(sr, ignore_index=True)
            df = df.astype({'調教コース坂': 'int8', '調教コースW': 'int8', '調教コースダ': 'int8', '調教コース芝': 'int8', '調教コースプール': 'int8',
                            '調教コース障害': 'int8',
                            '調教コースポリ': 'int8', '追切指数': 'int8', '仕上指数': 'int8'})
            if len(df.index) >= 60:
                df.to_pickle(self.jrdb_folder_path + 'CYB/' + filename + ".pkl")
            shutil.move(self.jrdb_folder_path + 'CYB/' + filename, self.jrdb_folder_path + 'backup/' + filename)
        return df

    def get_joa_df(self, filename):
        names = ['RACE_KEY', 'UMABAN', '血統登録番号', '馬名', '基準オッズ', '基準複勝オッズ', 'CID調教素点', 'CID厩舎素点', 'CID素点', 'CID', 'LS指数',
                 'LS評価', 'EM', '厩舎ＢＢ印', '厩舎ＢＢ◎単勝回収率', '厩舎ＢＢ◎連対率', '騎手ＢＢ印', '騎手ＢＢ◎単勝回収率', '騎手ＢＢ◎連対率', 'target_date']
        if filename[-3:] == "pkl":
            df = pd.read_pickle(self.jrdb_folder_path + 'JOA/' + filename)
        else:
            print(filename)
            with open(self.jrdb_folder_path + 'JOA/' + filename, 'r', encoding="ms932") as fh:
                df = pd.DataFrame(index=[], columns=names)
                for line in fh:
                    new_line = self.replace_line(line)
                    sr = pd.Series([
                        new_line[0:8]  # RACE_KEY
                        , self.str_null(new_line[8:10])  # UMABAN
                        , self.str_null(new_line[10:18])  # KETTO_TOROUK_BANGO
                        , self.str_null(new_line[18:54])  # .replace(" ", "")  # UMA_NAME
                        , self.float_null(new_line[54:59])  # KIJUN_ODDS
                        , self.float_null(new_line[59:64])  # KIJUN_FUKUSHO_ODDS
                        , self.float_null(new_line[64:69])  # CID_CHOKYO_SOTEN
                        , self.float_null(new_line[69:74])  # CID_KYUSHA_SOTEN
                        , self.float_null(new_line[74:79])  # CID_SOTEN
                        , self.int_0(new_line[79:82])  # CID
                        , self.float_null(new_line[82:87])  # LS_RECORD
                        , self.str_null(new_line[87:88])  # LS_HYOKA
                        , self.str_null(new_line[88:89])  # EM
                        , self.str_null(new_line[89:90])  # KYUSHA_BB_MARK
                        , self.int_0(new_line[90:95])  # KYUSHA_BB_TANSHO_KAISHU
                        , self.int_0(new_line[95:100])  # KYUSHA_BB_RENTAIRITSU
                        , self.str_null(new_line[100:101])  # KISHU_BB_MARK
                        , self.int_0(new_line[101:106])  # KISHU_BB_TANSHO_KAISHU
                        , self.int_0(new_line[106:111])  # TANSHO_BB_RENTAIRITSU
                        , self.get_kaisai_date(filename)  # target_date
                    ], index=names)
                    df = df.append(sr, ignore_index=True)
            df = df.astype({'厩舎ＢＢ◎単勝回収率': 'int16', '厩舎ＢＢ◎連対率': 'int16', '騎手ＢＢ◎単勝回収率': 'int16', '騎手ＢＢ◎連対率': 'int16'})
            if len(df.index) >= 60:
                df.to_pickle(self.jrdb_folder_path + 'JOA/' + filename + ".pkl")
            shutil.move(self.jrdb_folder_path + 'JOA/' + filename, self.jrdb_folder_path + 'backup/' + filename)
        return df

    def get_kab_df(self, filename):
        names = ['KAISAI_KEY', 'NENGAPPI', '開催区分', '曜日', '場名', '天候コード', '芝馬場状態コード', '芝馬場状態内', '芝馬場状態中', '芝馬場状態外',
                 '芝馬場差', '直線馬場差最内',
                 '直線馬場差内', '直線馬場差中', '直線馬場差外', '直線馬場差大外', 'ダ馬場状態コード', 'ダ馬場状態内', 'ダ馬場状態中', 'ダ馬場状態外', 'ダ馬場差', 'データ区分',
                 '連続何日目',
                 '芝種類', '草丈', '転圧', '凍結防止剤', '中間降水量', 'target_date']
        if filename[-3:] == "pkl":
            df = pd.read_pickle(self.jrdb_folder_path + 'KAB/' + filename)
        else:
            print(filename)
            with open(self.jrdb_folder_path + 'KAB/' + filename, 'r', encoding="ms932") as fh:
                df = pd.DataFrame(index=[], columns=names)
                for line in fh:
                    new_line = self.replace_line(line)
                    sr = pd.Series([
                        new_line[0:6]  # KAISAI_KEY
                        , self.str_null(new_line[6:14])  # NENGAPPI
                        , self.str_null(new_line[14:15])  # KAISAI_KUBUN
                        , self.str_null(new_line[15:17])  # YOUBI
                        , self.str_null(new_line[17:21])  # .replace(" ", "")  # BASHO_NAME
                        , self.str_null(new_line[21:22])  # TENKO
                        , self.str_null(new_line[22:24])  # SHIBA_JOTAI
                        , self.str_null(new_line[24:25])  # SHIBA_JOTAI_UCHI
                        , self.str_null(new_line[25:26])  # SHIBA_JOTAI_NAKA
                        , self.str_null(new_line[26:27])  # SHIBA_JOTAI_SOTO
                        , self.str_null(new_line[27:30])  # SHIBA_BABASA
                        , self.str_null(new_line[30:32])  # CHOKUSEN_BABASA_SAIUCHI
                        , self.str_null(new_line[32:34])  # CHOKUSEN_BABASA_UCHI
                        , self.str_null(new_line[34:36])  # CHOKUSEN_BABASA_NAKA
                        , self.str_null(new_line[36:38])  # CHOKUSEN_BABASA_SOTO
                        , self.str_null(new_line[38:40])  # CHOKUSEN_BABASA_OOSOTO
                        , self.str_null(new_line[40:42])  # DART_BABA_JOTAI
                        , self.str_null(new_line[42:43])  # DART_JOTAI_UCHI
                        , self.str_null(new_line[43:44])  # DART_JOTAI_NAKA
                        , self.str_null(new_line[44:45])  # DART_JOTAI_SOTO
                        , self.str_null(new_line[45:48])  # DART_BABASA
                        , self.str_null(new_line[48:49])  # DATA_KUBUN
                        , self.int_0(new_line[49:51])  # RENZOKU
                        , self.str_null(new_line[51:52])  # SHIBA_SHURUI
                        , self.float_null(new_line[52:56])  # KUSATAKE
                        , self.str_null(new_line[56:57])  # TENATSU
                        , self.str_null(new_line[57:58])  # TOKETSU_BOSHI
                        , self.float_null(new_line[58:63])  # CHUKAN_KOUSUI
                        , self.get_kaisai_date(filename)  # target_date
                    ], index=names)
                    df = df.append(sr, ignore_index=True)
            df = df.astype({"芝馬場差": 'int8', "ダ馬場差": 'int8', "連続何日目": 'int8', "中間降水量": 'float16', "草丈": 'float16'})
            if len(df.query("データ区分 == '4'")) >= 1:
                df.to_pickle(self.jrdb_folder_path + 'KAB/' + filename + ".pkl")
            shutil.move(self.jrdb_folder_path + 'KAB/' + filename, self.jrdb_folder_path + 'backup/' + filename)
        return df

    def get_kka_df(self, filename):
        names = ['RACE_KEY', 'UMABAN', 'ＪＲＡ成績１着', 'ＪＲＡ成績２着', 'ＪＲＡ成績３着', 'ＪＲＡ成績着外', '交流成績１着', '交流成績２着', '交流成績３着',
                 '交流成績着外', '他成績１着', '他成績２着', '他成績３着', '他成績着外',
                 '芝ダ障害別成績１着', '芝ダ障害別成績２着', '芝ダ障害別成績３着', '芝ダ障害別成績着外',
                 '芝ダ障害別距離成績１着', '芝ダ障害別距離成績２着', '芝ダ障害別距離成績３着', '芝ダ障害別距離成績着外', 'トラック距離成績１着', 'トラック距離成績２着', 'トラック距離成績３着',
                 'トラック距離成績着外',
                 'ローテ成績１着', 'ローテ成績２着', 'ローテ成績３着', 'ローテ成績着外', '回り成績１着', '回り成績２着', '回り成績３着', '回り成績着外',
                 '騎手成績１着', '騎手成績２着', '騎手成績３着', '騎手成績着外', '良成績１着', '良成績２着', '良成績３着', '良成績着外', '稍成績１着', '稍成績２着', '稍成績３着',
                 '稍成績着外', '重成績１着',
                 '重成績２着', '重成績３着', '重成績着外', 'Ｓペース成績１着', 'Ｓペース成績２着', 'Ｓペース成績３着', 'Ｓペース成績着外',
                 'Ｍペース成績１着', 'Ｍペース成績２着', 'Ｍペース成績３着', 'Ｍペース成績着外', 'Ｈペース成績１着', 'Ｈペース成績２着', 'Ｈペース成績３着', 'Ｈペース成績着外',
                 '季節成績１着', '季節成績２着',
                 '季節成績３着', '季節成績着外', '枠成績１着', '枠成績２着', '枠成績３着', '枠成績着外', '騎手距離成績１着', '騎手距離成績２着', '騎手距離成績３着', '騎手距離成績着外',
                 '騎手トラック距離成績１着', '騎手トラック距離成績２着', '騎手トラック距離成績３着', '騎手トラック距離成績着外', '騎手調教師別成績１着', '騎手調教師別成績２着',
                 '騎手調教師別成績３着',
                 '騎手調教師別成績着外', '騎手馬主別成績１着', '騎手馬主別成績２着', '騎手馬主別成績３着', '騎手馬主別成績着外',
                 '騎手ブリンカ成績１着', '騎手ブリンカ成績２着', '騎手ブリンカ成績３着', '騎手ブリンカ成績着外', '調教師馬主別成績１着', '調教師馬主別成績２着', '調教師馬主別成績３着',
                 '調教師馬主別成績着外', '父馬産駒芝連対率', '父馬産駒ダ連対率', '父馬産駒連対平均距離', '母父馬産駒芝連対率', '母父馬産駒ダ連対率', '母父馬産駒連対平均距離',
                 'target_date']
        if filename[-3:] == "pkl":
            df = pd.read_pickle(self.jrdb_folder_path + 'KKA/' + filename)
        else:
            print(filename)
            with open(self.jrdb_folder_path + 'KKA/' + filename, 'r', encoding="ms932") as fh:
                df = pd.DataFrame(index=[], columns=names)
                for line in fh:
                    new_line = self.replace_line(line)
                    sr = pd.Series([
                        new_line[0:8]  # RACE_KEY
                        , self.str_null(new_line[8:10])  # UMABAN
                        , self.int_0(new_line[10:13])  # JRA_1CK
                        , self.int_0(new_line[13:16])  # JRA_2CK
                        , self.int_0(new_line[16:19])  # JRA_3CK
                        , self.int_0(new_line[19:22])  # JRA_GAI
                        , self.int_0(new_line[22:25])  # NRA_1CK
                        , self.int_0(new_line[25:28])  # NRA_2CK
                        , self.int_0(new_line[28:31])  # NRA_3CK
                        , self.int_0(new_line[31:34])  # NRA_GAI
                        , self.int_0(new_line[34:37])  # OTHER_1CK
                        , self.int_0(new_line[37:40])  # OTHER_2CK
                        , self.int_0(new_line[40:43])  # OTHER_3CK
                        , self.int_0(new_line[43:46])  # OTHER_GAI
                        , self.int_0(new_line[46:49])  # SHIDA_1CK
                        , self.int_0(new_line[49:52])  # SHIDA_2CK
                        , self.int_0(new_line[52:55])  # SHIDA_3CK
                        , self.int_0(new_line[55:58])  # SHIDA_GAI
                        , self.int_0(new_line[58:61])  # SHIDA_KYORI_1CK
                        , self.int_0(new_line[61:64])  # SHIDA_KYORI_2CK
                        , self.int_0(new_line[64:67])  # SHIDA_KYORI_3CK
                        , self.int_0(new_line[67:70])  # SHIDA_KYORI_GAI
                        , self.int_0(new_line[70:73])  # TRACK_1CK
                        , self.int_0(new_line[73:76])  # TRACK_2CK
                        , self.int_0(new_line[76:79])  # TRACK_3CK
                        , self.int_0(new_line[79:82])  # TRACK_GAI
                        , self.int_0(new_line[82:85])  # ROTE_1CK
                        , self.int_0(new_line[85:88])  # ROTE_2CK
                        , self.int_0(new_line[88:91])  # ROTE_3CK
                        , self.int_0(new_line[91:94])  # ROTE_GAI
                        , self.int_0(new_line[94:97])  # MAWARI_1CK
                        , self.int_0(new_line[97:100])  # MAWARI_2CK
                        , self.int_0(new_line[100:103])  # MAWARI_3CK
                        , self.int_0(new_line[103:106])  # MAWARI_GAI
                        , self.int_0(new_line[106:109])  # KISHU_1CK
                        , self.int_0(new_line[109:112])  # KISHU_2CK
                        , self.int_0(new_line[112:115])  # KISHU_3CK
                        , self.int_0(new_line[115:118])  # KISHU_GAI
                        , self.int_0(new_line[118:121])  # RYO_1CK
                        , self.int_0(new_line[121:124])  # RYO_2CK
                        , self.int_0(new_line[124:127])  # RYO_3CK
                        , self.int_0(new_line[127:130])  # RYO_GAI
                        , self.int_0(new_line[130:133])  # YAYA_1CK
                        , self.int_0(new_line[133:136])  # YAYA_2CK
                        , self.int_0(new_line[136:139])  # YAYA_3CK
                        , self.int_0(new_line[139:142])  # YAYA_GAI
                        , self.int_0(new_line[142:145])  # OMO_1CK
                        , self.int_0(new_line[145:148])  # OMO_2CK
                        , self.int_0(new_line[148:151])  # OMO_3CK
                        , self.int_0(new_line[151:154])  # OMO_GAI
                        , self.int_0(new_line[154:157])  # S_1CK
                        , self.int_0(new_line[157:160])  # S_2CK
                        , self.int_0(new_line[160:163])  # S_3CK
                        , self.int_0(new_line[163:166])  # S_GAI
                        , self.int_0(new_line[166:169])  # M_1CK
                        , self.int_0(new_line[169:172])  # M_2CK
                        , self.int_0(new_line[172:175])  # M_3CK
                        , self.int_0(new_line[175:178])  # M_GAI
                        , self.int_0(new_line[178:181])  # H_1CK
                        , self.int_0(new_line[181:184])  # H_2CK
                        , self.int_0(new_line[184:187])  # H_3CK
                        , self.int_0(new_line[187:190])  # H_GAI
                        , self.int_0(new_line[190:193])  # SEASON_1CK
                        , self.int_0(new_line[193:196])  # SEASON_2CK
                        , self.int_0(new_line[196:199])  # SEASON_3CK
                        , self.int_0(new_line[199:202])  # SEASON_GAI
                        , self.int_0(new_line[202:205])  # WAKU_1CK
                        , self.int_0(new_line[205:208])  # WAKU_2CK
                        , self.int_0(new_line[208:211])  # WAKU_3CK
                        , self.int_0(new_line[211:214])  # WAKU_GAI
                        , self.int_0(new_line[214:217])  # KISHU_KYORI_1CK
                        , self.int_0(new_line[217:220])  # KISHU_KYORI_2CK
                        , self.int_0(new_line[220:223])  # KISHU_KYORI_3CK
                        , self.int_0(new_line[223:226])  # KISHU_KYORI_GAI
                        , self.int_0(new_line[226:229])  # KISHU_TRACK_1CK
                        , self.int_0(new_line[229:232])  # KISHU_TRACK_2CK
                        , self.int_0(new_line[232:235])  # KISHU_TRACK_3CK
                        , self.int_0(new_line[235:238])  # KISHU_TRACK_GAI
                        , self.int_0(new_line[238:241])  # KISHU_CHOKYOSHI_1CK
                        , self.int_0(new_line[241:244])  # KISHU_CHOKYOSHI_2CK
                        , self.int_0(new_line[244:247])  # KISHU_CHOKYOSHI_3CK
                        , self.int_0(new_line[247:250])  # KISHU_CHOKYOSHI_GAI
                        , self.int_0(new_line[250:253])  # KISHU_BANUSHI_1CK
                        , self.int_0(new_line[253:256])  # KISHU_BANUSHI_2CK
                        , self.int_0(new_line[256:259])  # KISHU_BANUSHI_3CK
                        , self.int_0(new_line[259:262])  # KISHU_BANUSHI_GAI
                        , self.int_0(new_line[262:265])  # KISHU_BURINKA_1CK
                        , self.int_0(new_line[265:268])  # KISHU_BURINKA_2CK
                        , self.int_0(new_line[268:271])  # KISHU_BURINKA_3CK
                        , self.int_0(new_line[271:274])  # KISHU_BURINKA_GAI
                        , self.int_0(new_line[274:277])  # CHOKYOSHI_BANUSHI_1CK
                        , self.int_0(new_line[277:280])  # CHOKYOSHI_BANUSHI_2CK
                        , self.int_0(new_line[280:283])  # CHOKYOSHI_BANUSHI_3CK
                        , self.int_0(new_line[283:286])  # CHOKYOSHI_BANUSHI_GAI
                        , self.int_0(new_line[286:289])  # CHICHI_SHIBA_RENTAI
                        , self.int_0(new_line[289:292])  # CHICHI_DART_RENTAI
                        , self.int_0(new_line[292:296])  # CHICHI_RENTAI_KYORI
                        # HAHACHICHI_SHIBA_RENTAI
                        # HAHACHICHI_DART_RENTAI
                        # HAHACHICHI_RENTAI_KYORI
                        , self.int_0(new_line[296:298]), self.int_0(new_line[299:302]), self.int_0(new_line[302:306])
                        , self.get_kaisai_date(filename)  # target_date
                    ], index=names)
                    df = df.append(sr, ignore_index=True)
            df = df.astype(
                {'ＪＲＡ成績１着': 'int8', 'ＪＲＡ成績２着': 'int8', 'ＪＲＡ成績３着': 'int8', 'ＪＲＡ成績着外': 'int8', '交流成績１着': 'int8',
                 '交流成績２着': 'int8', '交流成績３着': 'int8', '交流成績着外': 'int8', '他成績１着': 'int8', '他成績２着': 'int8',
                 '他成績３着': 'int8', '他成績着外': 'int8',
                 '芝ダ障害別成績１着': 'int8', '芝ダ障害別成績２着': 'int8', '芝ダ障害別成績３着': 'int8', '芝ダ障害別成績着外': 'int8',
                 '芝ダ障害別距離成績１着': 'int8', '芝ダ障害別距離成績２着': 'int8', '芝ダ障害別距離成績３着': 'int8', '芝ダ障害別距離成績着外': 'int8',
                 'トラック距離成績１着': 'int8', 'トラック距離成績２着': 'int8', 'トラック距離成績３着': 'int8', 'トラック距離成績着外': 'int8',
                 'ローテ成績１着': 'int8', 'ローテ成績２着': 'int8', 'ローテ成績３着': 'int8', 'ローテ成績着外': 'int8', '回り成績１着': 'int8',
                 '回り成績２着': 'int8', '回り成績３着': 'int8', '回り成績着外': 'int8',
                 '騎手成績１着': 'int8', '騎手成績２着': 'int8', '騎手成績３着': 'int8', '騎手成績着外': 'int8', '良成績１着': 'int8',
                 '良成績２着': 'int8', '良成績３着': 'int8', '良成績着外': 'int8', '稍成績１着': 'int8', '稍成績２着': 'int8', '稍成績３着': 'int8',
                 '稍成績着外': 'int8', '重成績１着': 'int8',
                 '重成績２着': 'int8', '重成績３着': 'int8', '重成績着外': 'int8', 'Ｓペース成績１着': 'int8', 'Ｓペース成績２着': 'int8',
                 'Ｓペース成績３着': 'int8', 'Ｓペース成績着外': 'int8',
                 'Ｍペース成績１着': 'int8', 'Ｍペース成績２着': 'int8', 'Ｍペース成績３着': 'int8', 'Ｍペース成績着外': 'int8', 'Ｈペース成績１着': 'int8',
                 'Ｈペース成績２着': 'int8', 'Ｈペース成績３着': 'int8', 'Ｈペース成績着外': 'int8', '季節成績１着': 'int8', '季節成績２着': 'int8',
                 '季節成績３着': 'int8', '季節成績着外': 'int8', '枠成績１着': 'int8', '枠成績２着': 'int8', '枠成績３着': 'int8', '枠成績着外': 'int8',
                 '騎手距離成績１着': 'int16', '騎手距離成績２着': 'int16', '騎手距離成績３着': 'int16', '騎手距離成績着外': 'int16',
                 '騎手トラック距離成績１着': 'int16', '騎手トラック距離成績２着': 'int16', '騎手トラック距離成績３着': 'int16', '騎手トラック距離成績着外': 'int16',
                 '騎手調教師別成績１着': 'int16', '騎手調教師別成績２着': 'int16', '騎手調教師別成績３着': 'int16',
                 '騎手調教師別成績着外': 'int16', '騎手馬主別成績１着': 'int16', '騎手馬主別成績２着': 'int16', '騎手馬主別成績３着': 'int16',
                 '騎手馬主別成績着外': 'int16',
                 '騎手ブリンカ成績１着': 'int16', '騎手ブリンカ成績２着': 'int16', '騎手ブリンカ成績３着': 'int16', '騎手ブリンカ成績着外': 'int16',
                 '調教師馬主別成績１着': 'int16', '調教師馬主別成績２着': 'int16', '調教師馬主別成績３着': 'int16',
                 '調教師馬主別成績着外': 'int16', '父馬産駒芝連対率': 'int8', '父馬産駒ダ連対率': 'int8', '父馬産駒連対平均距離': 'int16',
                 '母父馬産駒芝連対率': 'int8', '母父馬産駒ダ連対率': 'int8', '母父馬産駒連対平均距離': 'int16'})
            if len(df.index) >= 60:
                df.to_pickle(self.jrdb_folder_path + 'KKA/' + filename + ".pkl")
            shutil.move(self.jrdb_folder_path + 'KKA/' + filename, self.jrdb_folder_path + 'backup/' + filename)
        return df

    def get_kyi_df(self, filename):
        names = ['RACE_KEY', 'UMABAN', 'NENGAPPI', '血統登録番号', 'IDM', '騎手指数', '情報指数', '総合指数', '脚質', '距離適性', '上昇度',
                 'ローテーション', '基準オッズ', '基準人気順位', '基準複勝オッズ', '基準複勝人気順位', '特定情報◎', '特定情報○', '特定情報▲', '特定情報△', '特定情報×',
                 '総合情報◎', '総合情報○', '総合情報▲', '総合情報△', '総合情報×', '人気指数', '調教指数', '厩舎指数', '調教矢印コード', '厩舎評価コード', '騎手期待連対率',
                 '激走指数', '蹄コード', '重適正コード', 'クラスコード', 'ブリンカー', '騎手名', '負担重量', '見習い区分', '調教師名',
                 '調教師所属', 'ZENSO1_KYOSO_RESULT', 'ZENSO2_KYOSO_RESULT', 'ZENSO3_KYOSO_RESULT', 'ZENSO4_KYOSO_RESULT',
                 'ZENSO5_KYOSO_RESULT', 'ZENSO1_RACE_KEY', 'ZENSO2_RACE_KEY', 'ZENSO3_RACE_KEY', 'ZENSO4_RACE_KEY',
                 'ZENSO5_RACE_KEY', '枠番', '総合印', 'ＩＤＭ印', '情報印', '騎手印', '厩舎印', '調教印', '激走印', '芝適性コード',
                 'ダ適性コード', '騎手コード', '調教師コード', '獲得賞金', '収得賞金', '条件クラス', 'テン指数', 'ペース指数', '上がり指数', '位置指数', 'ペース予想',
                 '道中順位', '道中差', '道中内外', '後３Ｆ順位', '後３Ｆ差', '後３Ｆ内外', 'ゴール順位', 'ゴール差', 'ゴール内外',
                 '展開記号', '距離適性２', '枠確定馬体重', '枠確定馬体重増減', '取消フラグ', '性別コード', '馬主名', '馬主会コード', '馬記号コード', '激走順位', 'LS指数順位',
                 'テン指数順位', 'ペース指数順位', '上がり指数順位', '位置指数順位', '騎手期待単勝率', '騎手期待３着内率', '輸送区分', '走法',
                 '体型０１', '体型０２', '体型０３', '体型０４', '体型０５', '体型０６', '体型０７', '体型０８', '体型０９', '体型１０', '体型１１', '体型１２', '体型１３',
                 '体型１４', '体型１５', '体型１６', '体型１７', '体型１８',
                 '体型総合１', '体型総合２', '体型総合３', '馬特記１', '馬特記２', '馬特記３', '馬スタート指数', '馬出遅率', '参考前走', '参考前走騎手コード', '万券指数',
                 '万券印', '降級フラグ', '激走タイプ', '休養理由分類コード', '芝ダ障害フラグ', '距離フラグ', 'クラスフラグ', '転厩フラグ', '去勢フラグ', '乗替フラグ', '入厩何走目',
                 '入厩年月日', '入厩何日前', '放牧先',
                 '放牧先ランク', '厩舎ランク', 'target_date']
        if filename[-3:] == "pkl":
            df = pd.read_pickle(self.jrdb_folder_path + 'KYI/' + filename)
        else:
            if os.path.exists(self.jrdb_folder_path + 'KYI/' + filename):
                with open(self.jrdb_folder_path + 'KYI/' + filename, 'r', encoding="ms932") as fh:
                    df = pd.DataFrame(index=[], columns=names)
                    for line in fh:
                        new_line = self.replace_line(line)
                        sr = pd.Series([
                            new_line[0:8]  # RACE_KEY
                            , self.str_null(new_line[8:10])  # UMABAN
                            , self.get_kaisai_date(filename)  # NENGAPPI
                            , self.str_null(new_line[10:18])  # KETTO_TOROKU_BANGO
                            , self.float_null(new_line[54:59])  # IDM
                            , self.float_null(new_line[59:64])  # KISHU_RECORD
                            , self.float_null(new_line[64:69])  # JOHO_RECORD
                            , self.float_null(new_line[84:89])  # SOGO_RECORD
                            , self.str_null(new_line[89:90])  # KYAKUSHITSU
                            , self.str_null(new_line[90:91])  # KYORI_TEKISEI
                            , self.str_null(new_line[91:92])  # JOSHODO
                            , self.int_0(new_line[92:95])  # ROTE
                            , self.float_null(new_line[95:100])  # BASE_ODDS
                            , self.int_0(new_line[100:102])  # BASE_NINKI_JUNI
                            , self.float_null(new_line[102:107])  # KIJUN_FUKUSHO_ODDS
                            , self.int_0(new_line[107:109])  # KIJUN_FUKUSHO_NINKIJUN
                            , self.int_0(new_line[109:112])  # TOKUTEI_HONMEI
                            , self.int_0(new_line[112:115])  # TOKUTEI_TAIKO
                            , self.int_0(new_line[115:118])  # TOKUTEI_TANANA
                            , self.int_0(new_line[118:121])  # TOKUTEI_OSAE
                            , self.int_0(new_line[121:124])  # TOKUTEI_HIMO
                            , self.int_0(new_line[124:127])  # SOGO_HONMEI
                            , self.int_0(new_line[127:130])  # SOGO_TAIKO
                            , self.int_0(new_line[130:133])  # SOGO_TANANA
                            , self.int_0(new_line[133:136])  # SOGO_OSAE
                            , self.int_0(new_line[136:139])  # SOGO_HIMO
                            , self.float_null(new_line[139:144])  # NINKI_RECORD
                            , self.float_null(new_line[144:149])  # CHOKYO_RECORD
                            , self.float_null(new_line[149:154])  # KYUSHA_RECOCRD
                            , self.str_null(new_line[154:155])  # CHOKYO_YAJIRUSHI
                            , self.str_null(new_line[155:156])  # KYUSHA_HYOUKA
                            , self.float_null(new_line[156:160])  # KISHU_KITAI_RENRITSU
                            , self.int_0(new_line[160:163])  # GEKISO_RECORD
                            , self.str_null(new_line[163:165])  # HIDUME
                            , self.str_null(new_line[165:166])  # OMOTEKISEI
                            , self.str_null(new_line[166:168])  # CLASS_CODE
                            , self.str_null(new_line[170:171])  # BURINKA
                            , self.str_null(new_line[171:183])  # .replace(" ", "")  # KISHU_NAME
                            , self.int_0(new_line[183:186])  # FUTAN_JURYO
                            , self.str_null(new_line[186:187])  # MINARAIA_KUBUN
                            , self.str_null(new_line[187:199])  # .replace(" ", "")  # CHOKYOSHI_NAME
                            , self.str_null(new_line[199:203])  # .replace(" ", "")  # CHOKYOSHO_SHOZOKU
                            , self.str_null(new_line[203:219])  # ZENSO1_KYOSO_RESULT
                            , self.str_null(new_line[219:235])  # ZENSO2_KYOSO_RESULT
                            , self.str_null(new_line[235:251])  # ZENSO3_KYOSO_RESULT
                            , self.str_null(new_line[251:267])  # ZENSO4_KYOSO_RESULT
                            , self.str_null(new_line[267:283])  # ZENSO5_KYOSO_RESULT
                            , self.str_null(new_line[283:291])  # ZENSO1_RACE_KEY
                            , self.str_null(new_line[291:299])  # ZENSO2_RACE_KEY
                            , self.str_null(new_line[299:307])  # ZENSO3_RACE_KEY
                            , self.str_null(new_line[307:315])  # ZENSO4_RACE_KEY
                            , self.str_null(new_line[315:323])  # ZENSO5_RACE_KEY
                            , self.str_null(new_line[323:324])  # WAKUBAN
                            , self.str_null(new_line[326:327])  # TOTAL_MARK
                            , self.str_null(new_line[327:328])  # IDM_MARK
                            , self.str_null(new_line[328:329])  # JOHO_MARK
                            , self.str_null(new_line[329:330])  # KISHU_MARK
                            , self.str_null(new_line[330:331])  # KYUSHA_MARK
                            , self.str_null(new_line[331:332])  # CHOKYO_MARK
                            , self.str_null(new_line[332:333])  # GEKISO_MARK
                            , self.str_null(new_line[333:334])  # SHIBA_TEKISEI
                            , self.str_null(new_line[334:335])  # DART_TEKISEI
                            , self.str_null(new_line[335:340])  # KISHU_CODE
                            , self.str_null(new_line[340:345])  # CHOKYKOSHI_CODE
                            , self.int_0(new_line[346:352])  # KAKUTOKU_SHOKIN
                            , self.int_0(new_line[352:357])  # SHUTOK_SHOKIN
                            , self.str_null(new_line[357:358])  # JOKEN_CLASS
                            , self.float_null(new_line[358:363])  # TEN_RECORD
                            , self.float_null(new_line[363:368])  # PACE_RECORD
                            , self.float_null(new_line[368:373])  # AGARI_RECORD
                            , self.float_null(new_line[373:378])  # ICHI_RECORD
                            , self.str_null(new_line[378:379])  # PACE_YOSO
                            , self.int_0(new_line[379:381])  # DOCHU_JUNI
                            , self.int_0(new_line[381:383])  # DOCHU_SA
                            , self.str_null(new_line[383:384])  # DOCHU_UCHISOTO
                            , self.int_0(new_line[384:386])  # ATO3F_JUNI
                            , self.int_0(new_line[386:388])  # ATO3F_SA
                            , self.str_null(new_line[388:389])  # ATO3F_UCHISOTO
                            , self.int_0(new_line[389:391])  # GOAL_JUNI
                            , self.int_0(new_line[391:393])  # GOAL_SA
                            , self.str_null(new_line[393:394])  # GOAL_UCHISOTO
                            , self.str_null(new_line[394:395])  # TENAI_MARK
                            , self.str_null(new_line[395:396])  # KYORI_TEKISEI2
                            , self.int_0(new_line[396:399])  # WAKU_KAKUTEI_BATAIJU
                            , self.int_0(new_line[399:402])  # WAKU_KAKUTEI_ZOGEN
                            , self.str_null(new_line[402:403])  # TORIKESHI
                            , self.str_null(new_line[403:404])  # SEX
                            , self.str_null(new_line[404:444])  ##.replace(" ", "")  # BANUSHI_NAME
                            , self.str_null(new_line[444:446])  # BANUSHI_CODE
                            , self.str_null(new_line[446:448])  # UMA_KIGO
                            , self.int_0(new_line[448:450])  # GEKISO_JUNI
                            , self.int_0(new_line[450:452])  # LS_RECORD_JUNI
                            , self.int_0(new_line[452:454])  # TEN_RECORD_JUNI
                            , self.int_0(new_line[454:456])  # PACE_RECORD_JUNI
                            , self.int_0(new_line[456:458])  # AGARI_RECORD_JUNI
                            , self.int_0(new_line[458:460])  # ICHI_RECORD_JUNI
                            , self.float_null(new_line[460:464])  # KISHU_KITAI_1CK
                            , self.float_null(new_line[464:468])  # KISHU_KITAI_3CK
                            , self.str_null(new_line[468:469])  # YUSO_KUBUN
                            , self.str_null(new_line[469:477])  # SOHO
                            , self.str_null(new_line[477:478])  # TAIEKI_01
                            , self.str_null(new_line[478:479])  # TAIEKI_02
                            , self.str_null(new_line[479:480])  # TAIEKI_03
                            , self.str_null(new_line[480:481])  # TAIEKI_04
                            , self.str_null(new_line[481:482])  # TAIEKI_05
                            , self.str_null(new_line[482:483])  # TAIEKI_06
                            , self.str_null(new_line[483:484])  # TAIEKI_07
                            , self.str_null(new_line[484:485])  # TAIEKI_08
                            , self.str_null(new_line[485:486])  # TAIEKI_09
                            , self.str_null(new_line[486:487])  # TAIEKI_10
                            , self.str_null(new_line[487:488])  # TAIEKI_11
                            , self.str_null(new_line[488:489])  # TAIEKI_12
                            , self.str_null(new_line[489:490])  # TAIEKI_13
                            , self.str_null(new_line[490:491])  # TAIEKI_14
                            , self.str_null(new_line[491:492])  # TAIEKI_15
                            , self.str_null(new_line[492:493])  # TAIEKI_16
                            , self.str_null(new_line[493:494])  # TAIEKI_17
                            , self.str_null(new_line[494:495])  # TAIEKI_18
                            , self.str_null(new_line[501:504])  # TAIKEI1
                            , self.str_null(new_line[504:507])  # TAIKEI2
                            , self.str_null(new_line[507:510])  # TAIKEI3
                            , self.str_null(new_line[510:513])  # UMA_TOKKI1
                            , self.str_null(new_line[513:516])  # UMA_TOKKI2
                            , self.str_null(new_line[516:519])  # UMA_TOKKI3
                            , self.float_null(new_line[519:523])  # UMA_START_RECORD
                            , self.float_null(new_line[523:527])  # UMA_DEOKURE_RITSU
                            , self.str_null(new_line[527:529])  # SANKO_ZENSO
                            , self.str_null(new_line[529:534])  # SANKO_ZENSO_KISHU_CODE
                            , self.int_0(new_line[534:537])  # MANKEN_RECORD
                            , self.str_null(new_line[537:538])  # MANKEN_MARK
                            , self.str_null(new_line[538:539])  # KOKYU_FLAG
                            , self.str_null(new_line[539:541])  # GEKISO_TYPE
                            , self.str_null(new_line[541:543])  # KYUYOU_RIYU_CODE
                            , self.str_null(new_line[543:544])  # FLAG_1
                            , self.str_null(new_line[544:545])  # FLAG_2
                            , self.str_null(new_line[545:546])  # FLAG_3
                            , self.str_null(new_line[546:547])  # FLAG_4
                            , self.str_null(new_line[547:548])  # FLAG_5
                            , self.str_null(new_line[548:549])  # FLAG_6
                            , self.int_0(new_line[559:561])  # NYUKYO_SOUME
                            , self.str_null(new_line[561:569])  # NYUKYO_NENGAPPI
                            , self.int_0(new_line[569:572])  # NYUKYO_NICHIMAE
                            , self.str_null(new_line[572:622])  # HOUBOKUSAKI
                            , self.str_null(new_line[622:623])  # HOUBOKUSAKI_RANK
                            , self.str_null(new_line[623:624])  # KYUSHA_RANK
                            , self.get_kaisai_date(filename)  # target_date
                        ], index=names)
                        df = df.append(sr, ignore_index=True)
                df = df.astype(
                    {'ローテーション': 'int8', '基準人気順位': 'int8', '基準複勝人気順位': 'int8', '特定情報◎': 'int8', '特定情報○': 'int8',
                     '特定情報▲': 'int8',
                     '特定情報△': 'int8', '特定情報×': 'int8', '総合情報◎': 'int8', '総合情報○': 'int8', '総合情報▲': 'int8',
                     '総合情報△': 'int8', '総合情報×': 'int8',
                     '獲得賞金': 'int32', '収得賞金': 'int32', '道中順位': 'int8', '道中差': 'int8', '後３Ｆ順位': 'int8',
                     '後３Ｆ差': 'int8', 'ゴール順位': 'int8',
                     'ゴール差': 'int8', '激走順位': 'int8', 'LS指数順位': 'int8', 'テン指数順位': 'int8', 'ペース指数順位': 'int8',
                     '上がり指数順位': 'int8', '位置指数順位': 'int8',
                     '万券指数': 'int8', '厩舎ランク': 'int8'})
                if len(df.index) <= 60:
                    print("--- 予測データ不足!!")
                    # archiveフォルダとpickle化したファイルを削除
                    archive_file = self.jrdb_folder_path + 'archive/PACI' + filename[3:9] + ".zip"
                    if os.path.exists(archive_file):
                        os.remove(archive_file)
                else:
                    df.to_pickle(self.jrdb_folder_path + 'KYI/' + filename + ".pkl")
                shutil.move(self.jrdb_folder_path + 'KYI/' + filename, self.jrdb_folder_path + 'backup/' + filename)
            else:
                df = pd.DataFrame()
        return df

    def get_ukc_df(self, filename):
        names = ['血統登録番号', '馬名', '性別コード', '毛色コード', '馬記号コード', '父馬名', '母馬名', '母父馬名', '生年月日', '父馬生年', '母馬生年',
                 '母父馬生年', '馬主名', '馬主会コード', '生産者名', '産地名', '登録抹消フラグ', 'NENGAPPI', '父系統コード', '母父系統コード', 'target_date']
        if filename[-3:] == "pkl":
            df = pd.read_pickle(self.jrdb_folder_path + 'UKC/' + filename)
        else:
            print(filename)
            with open(self.jrdb_folder_path + 'UKC/' + filename, 'r', encoding="ms932") as fh:
                df = pd.DataFrame(index=[], columns=names)
                for line in fh:
                    new_line = self.replace_line(line)
                    sr = pd.Series([
                        new_line[0:8]  # KETTO_TOROKU_BANGO
                        , self.str_null(new_line[8:44])  # UMA_NAME
                        , self.str_null(new_line[44:45])  # SEX
                        , self.str_null(new_line[45:47])  # KEIRO
                        , self.str_null(new_line[47:49])  # UMA_KIGO
                        , self.str_null(new_line[49:85])  # .replace(" ", "")  # CHICHI_NAME
                        , self.str_null(new_line[85:121])  # .replace(" ", "")  # HAHA_NAME
                        , self.str_null(new_line[121:157])  # .replace(" ", "")  # HAHA_CHICHI_NAME
                        , self.str_null(new_line[157:165])  # SEINENGAPPI
                        , self.str_null(new_line[165:169])  # CHICHI_UMARE_YEAR
                        , self.str_null(new_line[169:173])  # HAHA_UMARE_YEAR
                        , self.str_null(new_line[173:177])  # HAHA_CHICHI_UMARE_YEAR
                        , self.str_null(new_line[177:217])  # .replace(" ", "")  # BANUSHI_NAME
                        , self.str_null(new_line[217:219])  # BANUSHI_CODE
                        , self.str_null(new_line[219:259])  # .replace(" ", "")  # SEISANSHA
                        , self.str_null(new_line[259:267])  # .replace(" ", "")  # SANCHI
                        , self.str_null(new_line[267:268])  # MASSHO_FLAG
                        , self.str_null(new_line[268:276])  # DATA_NENGAPPI
                        , self.str_null(new_line[276:280])  # CHICHI_KEITO
                        , self.str_null(new_line[280:284])  # HAHA_CHICHI_KEITO
                        , self.get_kaisai_date(filename)  # target_date
                    ], index=names)
                    df = df.append(sr, ignore_index=True)
            if len(df.index) >= 60:
                df.to_pickle(self.jrdb_folder_path + 'UKC/' + filename + ".pkl")
            shutil.move(self.jrdb_folder_path + 'UKC/' + filename, self.jrdb_folder_path + 'backup/' + filename)
        return df

    def get_sed_df(self, filename, type):
        names = ['RACE_KEY', 'UMABAN', '血統登録番号', 'NENGAPPI', 'KYOSO_RESULT_KEY', '馬名', '距離', '芝ダ障害コード', '右左', '内外',
                 '馬場状態', '種別', '条件', '記号', '重量', 'グレード', 'レース名', '頭数', 'レース名略称',
                 '着順', '異常区分', 'タイム', '斤量', '騎手名', '調教師名', '確定単勝オッズ', '確定単勝人気順位', 'ＩＤＭ結果', '素点', '馬場差', 'ペース', '出遅',
                 '位置取', '不利', '前不利', '中不利', '後不利',
                 'レース', 'コース取り', '上昇度コード', 'クラスコード', '馬体コード', '気配コード',
                 'レースペース', '馬ペース', 'テン指数結果', '上がり指数結果', 'ペース指数結果', 'レースＰ指数結果', '1(2)着馬名', '1(2)着タイム差', '前３Ｆタイム',
                 '後３Ｆタイム', '確定複勝オッズ下', '10時単勝オッズ',
                 '10時複勝オッズ', 'コーナー順位１', 'コーナー順位２', 'コーナー順位３', 'コーナー順位４', '前３Ｆ先頭差', '後３Ｆ先頭差', '騎手コード', '調教師コード', '馬体重',
                 '馬体重増減', '天候コード', 'コース', 'レース脚質',
                 '単勝', '複勝', '本賞金', '収得賞金', 'レースペース流れ', '馬ペース流れ', '４角コース取り', 'target_date']
        if filename[-3:] == "pkl":
            df = pd.read_pickle(self.jrdb_folder_path + type + '/' + filename)
        else:
            if os.path.exists(self.jrdb_folder_path + type + '/' + filename):
                with open(self.jrdb_folder_path + type + '/' + filename, 'r', encoding="ms932") as fh:
                    df = pd.DataFrame(index=[], columns=names)
                    for line in fh:
                        new_line = self.replace_line(line)
                        sr = pd.Series([
                            new_line[0:8]  # RACE_KEY
                            , self.str_null(new_line[8:10])  # UMABAN
                            , self.str_null(new_line[10:18])  # KETTO_TOROKU_BANGO
                            , self.str_null(new_line[18:26])  # NENGAPPI
                            , self.str_null(new_line[10:26])  # KYOSO_RESULT_KEY
                            , self.str_null(new_line[26:62])  # .replace(" ", "")  # UMA_NAME
                            , self.int_0(new_line[62:66])  # KYORI
                            , self.str_null(new_line[66:67])  # SHIBA_DART
                            , self.str_null(new_line[67:68])  # MIGIHIDARI
                            , self.str_null(new_line[68:69])  # UCHISOTO
                            , self.str_null(new_line[69:71])  # BABA_JOTAI
                            , self.str_null(new_line[71:73])  # SHUBETSU
                            , self.str_null(new_line[73:75])  # JOKEN
                            , self.str_null(new_line[75:78])  # KIGO
                            , self.str_null(new_line[78:79])  # JURYO
                            , self.str_null(new_line[79:80])  # GRADE
                            , self.str_null(new_line[80:130])  # RACE_NAME
                            , self.int_0(new_line[130:132])  # TOSU
                            , self.str_null(new_line[132:140])  # RACE_NAME_RYAKUSHO
                            , self.int_0(new_line[140:142])  # CHAKUJUN
                            , self.str_null(new_line[142:143])  # IJO_KUBUN
                            , self.convert_time(new_line[143:147])  # TIME
                            , self.int_0(new_line[147:150])  # KINRYO
                            , self.str_null(new_line[150:162])  # .replace(" ", "")  # KISHU_NAME
                            , self.str_null(new_line[162:174])  # .replace(" ", "")  # CHOKYOSHI_NAME
                            , self.float_null(new_line[174:180])  # KAKUTEI_TANSHO_ODDS
                            # KAKUTEI_TANSHO_NINKIJUN
                            # IDM
                            # SOTEN
                            # BABA_SA
                            # PACE
                            # DEOKURE
                            # ICHIDORI
                            , self.int_0(new_line[180:182]), self.int_0(new_line[182:185]),
                            self.int_0(new_line[185:188]), self.int_0(new_line[188:191]), self.int_0(new_line[191:194]),
                            self.int_0(new_line[194:197]), self.int_0(new_line[197:200]), self.int_0(new_line[200:203])
                            # FURI
                            , self.int_0(new_line[203:206])  # MAE_FURI
                            , self.int_0(new_line[206:209])  # NAKA_FURI
                            , self.int_0(new_line[209:212])  # ATO_FURI
                            , self.int_0(new_line[212:215])  # RACE
                            , self.str_null(new_line[215:216])  # COURSE_DORI
                            , self.str_null(new_line[216:217])  # JOSHODO
                            , self.str_null(new_line[217:219])  # CLASS_CODE
                            , self.str_null(new_line[219:220])  # BATAI_CODE
                            , self.str_null(new_line[220:221])  # KEHAI_CODE
                            , self.str_null(new_line[221:222])  # RACE_PACE
                            , self.str_null(new_line[222:223])  # UMA_PACE
                            , self.float_null(new_line[223:228])  # TEN_RECORD
                            , self.float_null(new_line[228:233])  # AGARI_RECORD
                            , self.float_null(new_line[233:238])  # PACE_RECORD
                            , self.float_null(new_line[238:243])  # RACE_PACE_RECORD
                            , self.str_null(new_line[243:255])  # .replace(" ", "")  # KACHIUMA_NAME
                            , self.str_null(new_line[255:258])  # TIME_SA
                            , self.int_0(new_line[258:261])  # MAE_3F_TIME
                            , self.int_0(new_line[261:264])  # ATO_3F_TIME
                            , self.float_null(new_line[290:296])  # KAKUTEI_FUKUSHO_ODDS
                            , self.float_null(new_line[296:302])  # TANSHO_ODDS_10JI
                            , self.float_null(new_line[302:308])  # FUKUSHO_ODDS_10JI
                            , self.int_0(new_line[308:310])  # CORNER_JUNI1
                            , self.int_0(new_line[310:312])  # CORNER_JUNI2
                            , self.int_0(new_line[312:314])  # CORNER_JUNI3
                            , self.int_0(new_line[314:316])  # CORNER_JUNI4
                            , self.int_0(new_line[316:319])  # MAE_3F_SA
                            , self.int_0(new_line[319:322])  # ATO_3F_SA
                            , self.str_null(new_line[322:327])  # KISHU_CODE
                            , self.str_null(new_line[327:332])  # CHOKYOSHI_CODE
                            , self.int_0(new_line[332:335])  # BATAIJU
                            , self.int_bataiju_zogen(new_line[335:338])  # BATAIJU_ZOGEN
                            , self.str_null(new_line[338:339])  # TENKO_CODE
                            , self.str_null(new_line[339:340])  # COURSE
                            , self.str_null(new_line[340:341])  # RACE_KYAKUSHITSU
                            , self.int_haito(new_line[341:348])  # TANSHO
                            , self.int_haito(new_line[348:355])  # FUKUSHO
                            , self.int_shokin(new_line[355:360])  # HON_SHOKIN
                            , self.int_shokin(new_line[360:365])  # SHUTOKU_SHOKIN
                            , self.str_null(new_line[365:367])  # RACE_PACE_NAGARE
                            , self.str_null(new_line[367:369])  # UMA_PACE_NAGARE
                            , self.str_null(new_line[369:370])  # COURSE_4KAKU
                            , self.get_kaisai_date(filename)  # target_date
                        ], index=names)
                        df = df.append(sr, ignore_index=True)
                if type == "SED":
                    df.loc[:, "テン指数結果順位"] = df.groupby(["RACE_KEY"])["テン指数結果"].rank(ascending=False).fillna(0)
                    df.loc[:, "上がり指数結果順位"] = df.groupby(["RACE_KEY"])["上がり指数結果"].rank(ascending=False).fillna(0)
                    df.loc[:, "ペース指数結果順位"] = df.groupby(["RACE_KEY"])["ペース指数結果"].rank(ascending=False).fillna(0)
                elif type == "ZED":
                    target_date_list = df["NENGAPPI"].drop_duplicates()
                    # 成績データを取得して指数の順位を取得
                    all_sed_df = pd.DataFrame()
                    for date in target_date_list:
                        sed_file_name = f"SED{date[2:8]}.txt"
                        if os.path.exists(self.jrdb_folder_path + "SED/" + sed_file_name + ".pkl"):
                            sed_file_name = sed_file_name + ".pkl"
                        sed_df = self.get_sed_df(sed_file_name, "SED")
                        all_sed_df = pd.concat([all_sed_df, sed_df])
                    all_kyi_df = pd.DataFrame()
                    # 競走データを取得して出走時情報(予想ＩＤＭ等）を取得
                    for date in target_date_list:
                        kyi_file_name = f"KYI{date[2:8]}.txt"
                        if os.path.exists(self.jrdb_folder_path + "KYI/" + kyi_file_name + ".pkl"):
                            kyi_file_name = kyi_file_name + ".pkl"
                        kyi_df = self.get_kyi_df(kyi_file_name)
                        all_kyi_df = pd.concat([all_kyi_df, kyi_df])

                    df = pd.merge(df, all_sed_df[["RACE_KEY", "UMABAN", "テン指数結果順位", "上がり指数結果順位", "ペース指数結果順位"]],
                                  on=["RACE_KEY", "UMABAN"], how="left").fillna(0)
                    df = pd.merge(df, all_kyi_df[["RACE_KEY", "UMABAN", "IDM"]],
                                  on=["RACE_KEY", "UMABAN"], how="left").fillna(0)
                df = df.astype(
                    {'頭数': 'int8', '着順': 'int8', 'タイム': 'int16', '斤量': 'int16', '確定単勝人気順位': 'int8', 'ＩＤＭ結果': 'int8',
                     '素点': 'int8', '馬場差': 'int8', 'ペース': 'int8', '出遅': 'int8',
                     '位置取': 'int8', '不利': 'int8', '前不利': 'int8', '中不利': 'int8', '後不利': 'int8', 'レース': 'int8',
                     '1(2)着タイム差': 'int8', '前３Ｆタイム': 'int16',
                     '後３Ｆタイム': 'int16', 'コーナー順位１': 'int8', 'コーナー順位２': 'int8', 'コーナー順位３': 'int8', 'コーナー順位４': 'int8',
                     '前３Ｆ先頭差': 'int16', '後３Ｆ先頭差': 'int16',
                     '馬体重': 'int16', '馬体重増減': 'int8', '単勝': 'int32', '複勝': 'int16', '本賞金': 'int32', '収得賞金': 'int32',
                     "テン指数結果順位": 'int8', "上がり指数結果順位": 'int8', "ペース指数結果順位": 'int8'
                     })
                if df["ＩＤＭ結果"].mean() != 0:
                    df.to_pickle(self.jrdb_folder_path + type + '/' + filename + ".pkl")
                shutil.move(self.jrdb_folder_path + type + '/' + filename, self.jrdb_folder_path + 'backup/' + filename)
            else:
                df = pd.DataFrame()
        return df

    def get_sed_sokuho_df(self, filename):
        names = ['RACE_KEY', 'UMABAN', '着順', '異常区分', '確定単勝オッズ', '確定単勝人気順位']
        if os.path.exists(self.jrdb_folder_path + 'sokuho/' + filename):
            with open(self.jrdb_folder_path + 'sokuho/' + filename, 'r', encoding="ms932") as fh:
                df = pd.DataFrame(index=[], columns=names)
                for line in fh:
                    new_line = self.replace_line(line)
                    sr = pd.Series([
                        new_line[0:8]  # RACE_KEY
                        , self.str_null(new_line[8:10])  # UMABAN
                        , self.int_0(new_line[140:142])  # CHAKUJUN
                        , self.str_null(new_line[142:143])  # IJO_KUBUN
                        , self.float_null(new_line[174:180])  # KAKUTEI_TANSHO_ODDS
                        , self.int_0(new_line[180:182])  # KAKUTEI_TANSHO_NINKIJUN
                    ], index=names)
                    df = df.append(sr, ignore_index=True)
            df = df.astype(
                {'着順': 'int8', '確定単勝人気順位': 'int8'})
        else:
            df = pd.DataFrame()
        return df

    def get_srb_df(self, filename):
        names = ['RACE_KEY', 'KAISAI_KEY', 'ハロンタイム０１', 'ハロンタイム０２', 'ハロンタイム０３', 'ハロンタイム０４', 'ハロンタイム０５', 'ハロンタイム０６',
                 'ハロンタイム０７', 'ハロンタイム０８', 'ハロンタイム０９', 'ハロンタイム１０',
                 'ハロンタイム１１', 'ハロンタイム１２', 'ハロンタイム１３', 'ハロンタイム１４', 'ハロンタイム１５', 'ハロンタイム１６', 'ハロンタイム１７', 'ハロンタイム１８',
                 '１コーナー', '２コーナー', '３コーナー',
                 '４コーナー', 'ペースアップ位置', '１角１', '１角２', '１角３', '２角１', '２角２', '２角３', '向正１', '向正２', '向正３', '３角１', '３角２',
                 '３角３', '４角０', '４角１', '４角２', '４角３',
                 '４角４', '直線０', '直線１', '直線２', '直線３', '直線４', 'レースコメント', 'target_date']
        if filename[-3:] == "pkl":
            df = pd.read_pickle(self.jrdb_folder_path + 'SRB/' + filename)
        else:
            # print(filename)
            if os.path.exists(self.jrdb_folder_path + 'SRB/' + filename):
                with open(self.jrdb_folder_path + 'SRB/' + filename, 'r', encoding="ms932") as fh:
                    df = pd.DataFrame(index=[], columns=names)
                    for line in fh:
                        new_line = self.replace_line(line)
                        sr = pd.Series([
                            new_line[0:8]  # RACE_KEY
                            , new_line[0:6]  # KAISAI_KEY
                            , self.int_0(new_line[8:11])  # HARON_01
                            , self.int_0(new_line[11:14])  # HARON_02
                            , self.int_0(new_line[14:17])  # HARON_03
                            , self.int_0(new_line[17:20])  # HARON_04
                            , self.int_0(new_line[20:23])  # HARON_05
                            , self.int_0(new_line[23:26])  # HARON_06
                            , self.int_0(new_line[26:29])  # HARON_07
                            , self.int_0(new_line[29:32])  # HARON_08
                            , self.int_0(new_line[32:35])  # HARON_09
                            , self.int_0(new_line[35:38])  # HARON_10
                            , self.int_0(new_line[38:41])  # HARON_11
                            , self.int_0(new_line[41:44])  # HARON_12
                            , self.int_0(new_line[44:47])  # HARON_13
                            , self.int_0(new_line[47:50])  # HARON_14
                            , self.int_0(new_line[50:53])  # HARON_15
                            , self.int_0(new_line[53:56])  # HARON_16
                            , self.int_0(new_line[56:59])  # HARON_17
                            , self.int_0(new_line[59:62])  # HARON_18
                            , self.str_null(new_line[62:126])  # CORNER_1
                            , self.str_null(new_line[126:190])  # CORNER_2
                            , self.str_null(new_line[190:254])  # CORNER_3
                            , self.str_null(new_line[254:318])  # CORNER_4
                            , self.int_null(new_line[318:320])  # PACE_UP_POINT
                            , self.str_null(new_line[320:321])  # TRACK_BIAS_1KAKU
                            , self.str_null(new_line[321:322])  # TRACK_BIAS_1KAKU
                            , self.str_null(new_line[322:323])  # TRACK_BIAS_1KAKU
                            , self.str_null(new_line[323:324])  # TRACK_BIAS_2KAKU
                            , self.str_null(new_line[324:325])  # TRACK_BIAS_2KAKU
                            , self.str_null(new_line[325:326])  # TRACK_BIAS_2KAKU
                            , self.str_null(new_line[326:327])  # TRACK_BIAS_MUKAI
                            , self.str_null(new_line[327:328])  # TRACK_BIAS_MUKAI
                            , self.str_null(new_line[328:329])  # TRACK_BIAS_MUKAI
                            , self.str_null(new_line[329:330])  # TRACK_BIAS_3KAKU
                            , self.str_null(new_line[330:331])  # TRACK_BIAS_3KAKU
                            , self.str_null(new_line[331:332])  # TRACK_BIAS_3KAKU
                            , self.str_null(new_line[332:333])  # TRACK_BIAS_4KAKU
                            , self.str_null(new_line[333:334])  # TRACK_BIAS_4KAKU
                            , self.str_null(new_line[334:335])  # TRACK_BIAS_4KAKU
                            , self.str_null(new_line[335:336])  # TRACK_BIAS_4KAKU
                            , self.str_null(new_line[336:337])  # TRACK_BIAS_4KAKU
                            , self.str_null(new_line[337:338])  # TRACK_BIAS_CHOKUSEN
                            , self.str_null(new_line[338:339])  # TRACK_BIAS_CHOKUSEN
                            , self.str_null(new_line[339:340])  # TRACK_BIAS_CHOKUSEN
                            , self.str_null(new_line[340:341])  # TRACK_BIAS_CHOKUSEN
                            , self.str_null(new_line[341:342])  # TRACK_BIAS_CHOKUSEN
                            , self.str_null(new_line[342:842])  # .replace(" ", "")  # RACE_COMMENT
                            , self.get_kaisai_date(filename)  # NENGAPPI
                        ], index=names)
                        df = df.append(sr, ignore_index=True)
                haron_list = ['ハロンタイム０１', 'ハロンタイム０２', 'ハロンタイム０３', 'ハロンタイム０４', 'ハロンタイム０５', 'ハロンタイム０６', 'ハロンタイム０７',
                              'ハロンタイム０８',
                              'ハロンタイム０９', 'ハロンタイム１０',
                              'ハロンタイム１１', 'ハロンタイム１２', 'ハロンタイム１３', 'ハロンタイム１４', 'ハロンタイム１５', 'ハロンタイム１６', 'ハロンタイム１７',
                              'ハロンタイム１８']
                df.loc[:, "ラップリスト"] = df.apply(lambda x: x[haron_list].tolist(), axis=1)
                df.loc[:, "ラップリスト"] = df["ラップリスト"].apply(lambda x: [s for s in x if s != 0])
                df.loc[:, "ラスト５ハロン"] = df["ラップリスト"].apply(lambda x: x[-5] if len(x) >= 5 else 0)
                df.loc[:, "ラスト４ハロン"] = df["ラップリスト"].apply(lambda x: x[-4] if len(x) >= 5 else 0)
                df.loc[:, "ラスト３ハロン"] = df["ラップリスト"].apply(lambda x: x[-3] if len(x) >= 5 else 0)
                df.loc[:, "ラスト２ハロン"] = df["ラップリスト"].apply(lambda x: x[-2] if len(x) >= 5 else 0)
                df.loc[:, "ラスト１ハロン"] = df["ラップリスト"].apply(lambda x: x[-1] if len(x) >= 5 else 0)
                df.loc[:, "ラップ差４ハロン"] = df["ラスト５ハロン"] - df["ラスト４ハロン"]
                df.loc[:, "ラップ差３ハロン"] = df["ラスト４ハロン"] - df["ラスト３ハロン"]
                df.loc[:, "ラップ差２ハロン"] = df["ラスト３ハロン"] - df["ラスト２ハロン"]
                df.loc[:, "ラップ差１ハロン"] = df["ラスト２ハロン"] - df["ラスト１ハロン"]
                df.loc[:, "abs_diff"] = df.apply(
                    lambda x: np.abs([x["ラップ差４ハロン"], x["ラップ差３ハロン"], x["ラップ差２ハロン"], x["ラップ差１ハロン"]]), axis=1)
                df.loc[:, "max_index"] = df["abs_diff"].apply(lambda x: np.argmax(x))
                df.loc[:, "max_rap"] = df.apply(lambda x: x["ラップ差４ハロン"] if x["max_index"] == 0 else (
                    x["ラップ差３ハロン"] if x["max_index"] == 1 else (
                        x["ラップ差２ハロン"] if x["max_index"] == 2 else x["ラップ差１ハロン"])), axis=1)
                df.loc[:, "RAP_TYPE"] = df.apply(lambda x: "一貫" if -5 <= x["max_rap"] <= 5 else (
                    f"L{4 - x['max_index']}加速" if np.sign(x["max_rap"]) > 0 else f"L{4- x['max_index']}失速"), axis=1)
                df.drop(["ラップリスト", "abs_diff", "max_index", "max_rap"], axis=1, inplace=True)

                kab_file_name = f"KAB{filename[3:9]}.txt"
                if os.path.exists(self.jrdb_folder_path + "KAB/" + kab_file_name + ".pkl"):
                    kab_file_name = kab_file_name + ".pkl"
                kab_df = self.get_kab_df(kab_file_name)
                df = pd.merge(df, kab_df[
                    ["KAISAI_KEY", "連続何日目", "芝種類", "草丈", "転圧", "凍結防止剤", "中間降水量", "ダ馬場状態コード", "芝馬場状態コード"]],
                              on="KAISAI_KEY")

                bac_file_name = f"BAC{filename[3:9]}.txt"
                if os.path.exists(self.jrdb_folder_path + "BAC/" + bac_file_name + ".pkl"):
                    bac_file_name = bac_file_name + ".pkl"
                bac_df = self.get_bac_df(bac_file_name)
                df = pd.merge(df, bac_df[["RACE_KEY", "距離", "芝ダ障害コード", "内外", "１着算入賞金"]], on="RACE_KEY")
                # factory analyzer用のデータを作成する
                df.loc[:, "ハロン数"] = df.apply(lambda x: x["距離"] / 200, axis=1)
                df.loc[:, "芝"] = df["芝ダ障害コード"].apply(lambda x: 1 if x == "1" else 0)
                df.loc[:, "外"] = df["内外"].apply(lambda x: 1 if x == "1" else 0)
                df.loc[:, "重"] = df.apply(
                    lambda x: 1 if (x["芝ダ障害コード"] == "1" and x["芝種類"] == "2") or (
                            x["芝ダ障害コード"] == "2" and x["凍結防止剤"] == "1") else 0, axis=1)
                df.loc[:, "軽"] = df.apply(
                    lambda x: 1 if (x["芝ダ障害コード"] == "1" and x["芝種類"] == "1") or (
                            x["芝ダ障害コード"] == "2" and x["転圧"] == "1") else 0, axis=1)
                df.loc[:, "馬場状態"] = df.apply(
                    lambda x: x["ダ馬場状態コード"] if x["芝ダ障害コード"] == "2" else x["芝馬場状態コード"], axis=1)
                df.drop(["ダ馬場状態コード", "芝馬場状態コード", "距離", "芝ダ障害コード", "内外"], axis=1, inplace=True)

                sed_file_name = f"SED{filename[3:9]}.txt"
                if os.path.exists(self.jrdb_folder_path + "SED/" + sed_file_name + ".pkl"):
                    sed_file_name = sed_file_name + ".pkl"
                sed_df = self.get_sed_df(sed_file_name, "SED")
                if sed_df.empty:
                    return pd.DataFrame()
                target_df = sed_df.query("着順 in (1,2,3,4,5)")  # .drop("着順", axis=1)

                # factory analyzer用のデータを作成する
                if target_df["ＩＤＭ結果"].mean() == 0:
                    print("--- 結果データ不足!!")
                    # archiveフォルダとpickle化したファイルを削除
                    archive_file = self.jrdb_folder_path + 'archive/SED' + filename[3:9] + ".zip"
                    if os.path.exists(archive_file):
                        os.remove(archive_file)
                    if os.path.exists(sed_file_name):
                        os.remove(sed_file_name)
                    df = pd.DataFrame()
                    return df

                target_df.loc[:, "追走力"] = target_df.apply(
                    lambda x: x["コーナー順位２"] - x["コーナー順位４"] if x["コーナー順位２"] != 0 else x["コーナー順位３"] - x["コーナー順位４"], axis=1)
                target_df.loc[:, "追上力"] = target_df.apply(lambda x: x["コーナー順位４"] - x["着順"], axis=1)
                target_df.loc[:, "ハロン数"] = target_df.apply(lambda x: x["距離"] / 200, axis=1)
                target_df.loc[:, "１ハロン平均"] = target_df.apply(lambda x: x["タイム"] / x["距離"] * 200, axis=1)
                target_df.loc[:, "後傾指数"] = target_df.apply(
                    lambda x: x["１ハロン平均"] * 3 / x["後３Ｆタイム"] if x["後３Ｆタイム"] != 0 else 1, axis=1)
                gp_mean_raceuma_result_df = target_df[
                    ["RACE_KEY", '１ハロン平均', 'ＩＤＭ結果', 'テン指数結果', '上がり指数結果', 'ペース指数結果', '前３Ｆタイム', '後３Ｆタイム',
                     'コーナー順位１', 'コーナー順位２', 'コーナー順位３', 'コーナー順位４', '前３Ｆ先頭差', '後３Ｆ先頭差', '追走力', '追上力',
                     '後傾指数']].groupby("RACE_KEY").mean() \
                    .reset_index().add_suffix('_mean').rename(columns={"RACE_KEY_mean": "RACE_KEY"})
                gp_std_raceuma_result_df = target_df[
                    ["RACE_KEY", '１ハロン平均', '上がり指数結果', 'ペース指数結果']].groupby("RACE_KEY").std() \
                    .reset_index().add_suffix('_std').rename(columns={"RACE_KEY_std": "RACE_KEY"})
                gp_raceuma_df = pd.merge(gp_mean_raceuma_result_df, gp_std_raceuma_result_df, on="RACE_KEY")[[
                    'RACE_KEY', '１ハロン平均_mean', 'ＩＤＭ結果_mean', 'テン指数結果_mean', '上がり指数結果_mean', 'ペース指数結果_mean',
                    '前３Ｆタイム_mean',
                    '後３Ｆタイム_mean', 'コーナー順位１_mean', 'コーナー順位２_mean', 'コーナー順位３_mean', 'コーナー順位４_mean', '前３Ｆ先頭差_mean',
                    '後３Ｆ先頭差_mean',
                    '追走力_mean', '追上力_mean', '後傾指数_mean', '１ハロン平均_std', '上がり指数結果_std', 'ペース指数結果_std']]

                target_df.loc[:, "馬番"] = target_df["UMABAN"].astype(int) / target_df["頭数"]
                column_list = ["馬番", "レース脚質", "４角コース取り", "コーナー順位３", "コーナー順位４"]
                target_df[column_list] = target_df[column_list].astype(int)
                group_df = target_df.groupby("RACE_KEY")[column_list].mean().reset_index()
                zero_group_df = pd.DataFrame(scipy.stats.zscore(group_df[column_list]), columns=column_list)
                zero_group_df.loc[:, "RACE_KEY"] = group_df["RACE_KEY"]
                zero_group_df.loc[:, "TRACK_BIAS_ZENGO"] = zero_group_df["レース脚質"] + zero_group_df["コーナー順位３"] + \
                                                           zero_group_df["コーナー順位４"]
                zero_group_df.loc[:, "TRACK_BIAS_UCHISOTO"] = zero_group_df["馬番"] + zero_group_df["４角コース取り"]
                df = pd.merge(df, zero_group_df[["RACE_KEY", "TRACK_BIAS_ZENGO", "TRACK_BIAS_UCHISOTO"]], on="RACE_KEY",
                              how="left")
                df = pd.merge(df, gp_raceuma_df, on="RACE_KEY", how="left")
                df = df.astype({'ハロンタイム０１': 'int16', 'ハロンタイム０２': 'int16', 'ハロンタイム０３': 'int16', 'ハロンタイム０４': 'int16',
                                'ハロンタイム０５': 'int16',
                                'ハロンタイム０６': 'int16', 'ハロンタイム０７': 'int16', 'ハロンタイム０８': 'int16', 'ハロンタイム０９': 'int16',
                                'ハロンタイム１０': 'int16',
                                'ハロンタイム１１': 'int16', 'ハロンタイム１２': 'int16', 'ハロンタイム１３': 'int16', 'ハロンタイム１４': 'int16',
                                'ハロンタイム１５': 'int16',
                                'ハロンタイム１６': 'int16', 'ハロンタイム１７': 'int16', 'ハロンタイム１８': 'int16'})
                df.to_pickle(self.jrdb_folder_path + 'SRB/' + filename + ".pkl")
                shutil.move(self.jrdb_folder_path + 'SRB/' + filename, self.jrdb_folder_path + 'backup/' + filename)
            else:
                df = pd.DataFrame()
        return df

    def _calc_raptype(self, arr):
        rap_list = [s for s in arr if s != 0]
        if len(rap_list) >= 5:
            last_rap = rap_list[-5:]
            rap_diff = [last_rap[3] - last_rap[4], last_rap[2] - last_rap[3], last_rap[1] - last_rap[2],
                        last_rap[0] - last_rap[1]]
            abs_diff = np.abs(rap_diff)
            max_index = np.argmax(abs_diff)
            max_rap = rap_diff[max_index]
            if max_rap in range(-5, 5):
                return "0"  # "一貫"
            elif np.sign(max_rap) > 0:
                return f"{max_index + 1}"  # f"L{max_index + 1}加速"
            else:
                return f"{max_index + 5}"  # f"L{max_index + 1}失速"
        else:
            return "9"  # "その他"

    def get_srb_sokuho_df(self, filename):
        names = ['RACE_KEY', 'KAISAI_KEY', 'ハロンタイム０１', 'ハロンタイム０２', 'ハロンタイム０３', 'ハロンタイム０４', 'ハロンタイム０５', 'ハロンタイム０６',
                 'ハロンタイム０７', 'ハロンタイム０８', 'ハロンタイム０９', 'ハロンタイム１０',
                 'ハロンタイム１１', 'ハロンタイム１２', 'ハロンタイム１３', 'ハロンタイム１４', 'ハロンタイム１５', 'ハロンタイム１６', 'ハロンタイム１７', 'ハロンタイム１８',
                 '１コーナー', '２コーナー', '３コーナー',
                 '４コーナー', 'ペースアップ位置', '１角１', '１角２', '１角３', '２角１', '２角２', '２角３', '向正１', '向正２', '向正３', '３角１', '３角２',
                 '３角３', '４角０', '４角１', '４角２', '４角３',
                 '４角４', '直線０', '直線１', '直線２', '直線３', '直線４', 'レースコメント', 'target_date']
        print(filename)
        if os.path.exists(self.jrdb_folder_path + 'sokuho/' + filename):
            with open(self.jrdb_folder_path + 'sokuho/' + filename, 'r', encoding="ms932") as fh:
                df = pd.DataFrame(index=[], columns=names)
                for line in fh:
                    new_line = self.replace_line(line)
                    sr = pd.Series([
                        new_line[0:8]  # RACE_KEY
                        , new_line[0:6]  # KAISAI_KEY
                        , self.int_0(new_line[8:11])  # HARON_01
                        , self.int_0(new_line[11:14])  # HARON_02
                        , self.int_0(new_line[14:17])  # HARON_03
                        , self.int_0(new_line[17:20])  # HARON_04
                        , self.int_0(new_line[20:23])  # HARON_05
                        , self.int_0(new_line[23:26])  # HARON_06
                        , self.int_0(new_line[26:29])  # HARON_07
                        , self.int_0(new_line[29:32])  # HARON_08
                        , self.int_0(new_line[32:35])  # HARON_09
                        , self.int_0(new_line[35:38])  # HARON_10
                        , self.int_0(new_line[38:41])  # HARON_11
                        , self.int_0(new_line[41:44])  # HARON_12
                        , self.int_0(new_line[44:47])  # HARON_13
                        , self.int_0(new_line[47:50])  # HARON_14
                        , self.int_0(new_line[50:53])  # HARON_15
                        , self.int_0(new_line[53:56])  # HARON_16
                        , self.int_0(new_line[56:59])  # HARON_17
                        , self.int_0(new_line[59:62])  # HARON_18
                        , self.str_null(new_line[62:126])  # CORNER_1
                        , self.str_null(new_line[126:190])  # CORNER_2
                        , self.str_null(new_line[190:254])  # CORNER_3
                        , self.str_null(new_line[254:318])  # CORNER_4
                        , self.int_null(new_line[318:320])  # PACE_UP_POINT
                        , self.str_null(new_line[320:321])  # TRACK_BIAS_1KAKU
                        , self.str_null(new_line[321:322])  # TRACK_BIAS_1KAKU
                        , self.str_null(new_line[322:323])  # TRACK_BIAS_1KAKU
                        , self.str_null(new_line[323:324])  # TRACK_BIAS_2KAKU
                        , self.str_null(new_line[324:325])  # TRACK_BIAS_2KAKU
                        , self.str_null(new_line[325:326])  # TRACK_BIAS_2KAKU
                        , self.str_null(new_line[326:327])  # TRACK_BIAS_MUKAI
                        , self.str_null(new_line[327:328])  # TRACK_BIAS_MUKAI
                        , self.str_null(new_line[328:329])  # TRACK_BIAS_MUKAI
                        , self.str_null(new_line[329:330])  # TRACK_BIAS_3KAKU
                        , self.str_null(new_line[330:331])  # TRACK_BIAS_3KAKU
                        , self.str_null(new_line[331:332])  # TRACK_BIAS_3KAKU
                        , self.str_null(new_line[332:333])  # TRACK_BIAS_4KAKU
                        , self.str_null(new_line[333:334])  # TRACK_BIAS_4KAKU
                        , self.str_null(new_line[334:335])  # TRACK_BIAS_4KAKU
                        , self.str_null(new_line[335:336])  # TRACK_BIAS_4KAKU
                        , self.str_null(new_line[336:337])  # TRACK_BIAS_4KAKU
                        , self.str_null(new_line[337:338])  # TRACK_BIAS_CHOKUSEN
                        , self.str_null(new_line[338:339])  # TRACK_BIAS_CHOKUSEN
                        , self.str_null(new_line[339:340])  # TRACK_BIAS_CHOKUSEN
                        , self.str_null(new_line[340:341])  # TRACK_BIAS_CHOKUSEN
                        , self.str_null(new_line[341:342])  # TRACK_BIAS_CHOKUSEN
                        , self.str_null(new_line[342:842])  # .replace(" ", "")  # RACE_COMMENT
                        , self.get_kaisai_date(filename)  # NENGAPPI
                    ], index=names)
                    df = df.append(sr, ignore_index=True)
            haron_list = ['ハロンタイム０１', 'ハロンタイム０２', 'ハロンタイム０３', 'ハロンタイム０４', 'ハロンタイム０５', 'ハロンタイム０６', 'ハロンタイム０７',
                          'ハロンタイム０８',
                          'ハロンタイム０９', 'ハロンタイム１０',
                          'ハロンタイム１１', 'ハロンタイム１２', 'ハロンタイム１３', 'ハロンタイム１４', 'ハロンタイム１５', 'ハロンタイム１６', 'ハロンタイム１７',
                          'ハロンタイム１８']
            df.loc[:, "ラップリスト"] = df.apply(lambda x: x[haron_list].tolist(), axis=1)
            df.loc[:, "ラップリスト"] = df["ラップリスト"].apply(lambda x: [s for s in x if s != 0])
            df.loc[:, "ラスト５ハロン"] = df["ラップリスト"].apply(lambda x: x[-5] if len(x) >= 5 else 0)
            df.loc[:, "ラスト４ハロン"] = df["ラップリスト"].apply(lambda x: x[-4] if len(x) >= 5 else 0)
            df.loc[:, "ラスト３ハロン"] = df["ラップリスト"].apply(lambda x: x[-3] if len(x) >= 5 else 0)
            df.loc[:, "ラスト２ハロン"] = df["ラップリスト"].apply(lambda x: x[-2] if len(x) >= 5 else 0)
            df.loc[:, "ラスト１ハロン"] = df["ラップリスト"].apply(lambda x: x[-1] if len(x) >= 5 else 0)
            df.loc[:, "ラップ差４ハロン"] = df["ラスト５ハロン"] - df["ラスト４ハロン"]
            df.loc[:, "ラップ差３ハロン"] = df["ラスト４ハロン"] - df["ラスト３ハロン"]
            df.loc[:, "ラップ差２ハロン"] = df["ラスト３ハロン"] - df["ラスト２ハロン"]
            df.loc[:, "ラップ差１ハロン"] = df["ラスト２ハロン"] - df["ラスト１ハロン"]
            df.loc[:, "abs_diff"] = df.apply(
                lambda x: np.abs([x["ラップ差４ハロン"], x["ラップ差３ハロン"], x["ラップ差２ハロン"], x["ラップ差１ハロン"]]), axis=1)
            df.loc[:, "max_index"] = df["abs_diff"].apply(lambda x: np.argmax(x))
            df.loc[:, "max_rap"] = df.apply(lambda x: x["ラップ差４ハロン"] if x["max_index"] == 0 else (
                x["ラップ差３ハロン"] if x["max_index"] == 1 else (x["ラップ差２ハロン"] if x["max_index"] == 2 else x["ラップ差１ハロン"])),
                                            axis=1)
            df.loc[:, "RAP_TYPE"] = df.apply(lambda x: "一貫" if -5 <= x["max_rap"] <= 5 else (
                f"L{4 - x['max_index']}加速" if np.sign(x["max_rap"]) > 0 else f"L{4- x['max_index']}失速"), axis=1)
            df.drop(["ラップリスト", "abs_diff", "max_index", "max_rap"], axis=1, inplace=True)
            df = df.astype({'ハロンタイム０１': 'int16', 'ハロンタイム０２': 'int16', 'ハロンタイム０３': 'int16', 'ハロンタイム０４': 'int16',
                            'ハロンタイム０５': 'int16',
                            'ハロンタイム０６': 'int16', 'ハロンタイム０７': 'int16', 'ハロンタイム０８': 'int16', 'ハロンタイム０９': 'int16',
                            'ハロンタイム１０': 'int16',
                            'ハロンタイム１１': 'int16', 'ハロンタイム１２': 'int16', 'ハロンタイム１３': 'int16', 'ハロンタイム１４': 'int16',
                            'ハロンタイム１５': 'int16',
                            'ハロンタイム１６': 'int16', 'ハロンタイム１７': 'int16', 'ハロンタイム１８': 'int16'})
        else:
            df = pd.DataFrame()
        return df

    def get_skb_df(self, filename, type):
        names = ['RACE_KEY', 'UMABAN', '血統登録番号', 'NENGAPPI', 'KYOSO_RESULT_KEY', '特記コード１', '特記コード２', '特記コード３', '特記コード４',
                 '特記コード５', '特記コード６', '馬具コード１', '馬具コード２', '馬具コード３', '馬具コード４',
                 '馬具コード５', '馬具コード６', '馬具コード７', '馬具コード８', '総合１', '総合２', '総合３', '左前１', '左前２', '左前３', '右前１',
                 '右前２', '右前３', '左後１', '左後２', '左後３', '右後１', '右後２', '右後３', 'パドックコメント', '脚元コメント', '馬具(その他)コメント',
                 'レース馬コメント', 'ハミ', 'バンテージ', '蹄鉄', '蹄状態',
                 'ソエ', '骨瘤', 'target_date']
        if filename[-3:] == "pkl":
            df = pd.read_pickle(self.jrdb_folder_path + type + '/' + filename)
        else:
            print(filename)
            with open(self.jrdb_folder_path + type + '/' + filename, 'r', encoding="ms932") as fh:
                df = pd.DataFrame(index=[], columns=names)
                for line in fh:
                    new_line = self.replace_line(line)
                    sr = pd.Series([
                        new_line[0:8]  # RACE_KEY
                        , self.str_null(new_line[8:10])  # UMABAN
                        , self.str_null(new_line[10:18])  # KETTO_TOROKU_BANGO
                        , self.str_null(new_line[18:26])  # NENGAPPI
                        , self.str_null(new_line[10:26])  # KYOSO_RESULT_KEY
                        , self.str_null(new_line[26:29])  # TOKKI_1
                        , self.str_null(new_line[29:32])  # TOKKI_2
                        , self.str_null(new_line[32:35])  # TOKKI_3
                        , self.str_null(new_line[35:38])  # TOKKI_4
                        , self.str_null(new_line[38:41])  # TOKKI_5
                        , self.str_null(new_line[41:44])  # TOKKI_6
                        , self.str_null(new_line[44:47])  # BAGU_1
                        , self.str_null(new_line[47:50])  # BAGU_2
                        , self.str_null(new_line[50:53])  # BAGU_3
                        , self.str_null(new_line[53:56])  # BAGU_4
                        , self.str_null(new_line[56:59])  # BAGU_5
                        , self.str_null(new_line[59:62])  # BAGU_6
                        , self.str_null(new_line[62:65])  # BAGU_7
                        , self.str_null(new_line[65:68])  # BAGU_8
                        , self.str_null(new_line[68:71])  # ASHIMOTO_SOGO_1
                        , self.str_null(new_line[71:74])  # ASHIMOTO_SOGO_2
                        , self.str_null(new_line[74:77])  # ASHIMOTO_SOGO_3
                        , self.str_null(new_line[77:80])  # ASHIMOTO_HIDARI_MAE_1
                        , self.str_null(new_line[80:83])  # ASHIMOTO_HIDARI_MAE_2
                        , self.str_null(new_line[83:86])  # ASHIMOTO_HIDARI_MAE_3
                        , self.str_null(new_line[86:89])  # ASHIMOTO_MIGI_MAE_1
                        , self.str_null(new_line[89:92])  # ASHIMOTO_MIGI_MAE_2
                        , self.str_null(new_line[92:95])  # ASHIMOTO_MIGI_MAE_3
                        , self.str_null(new_line[95:98])  # ASHIMOTO_HIDARI_USHIRO_1
                        , self.str_null(new_line[98:101])  # ASHIMOTO_HIDARI_USHIRO_2
                        , self.str_null(new_line[101:104])  # ASHIMOTO_HIDARI_USHIRO_3
                        , self.str_null(new_line[104:107])  # ASHIMOTO_MIGI_USHIRO_1
                        , self.str_null(new_line[107:110])  # ASHIMOTO_MIGI_USHIRO_2
                        , self.str_null(new_line[110:113])  # ASHIMOTO_MIGI_USHIRO_3
                        , self.str_null(new_line[113:153])  # .replace(" ", "")  # PADOC_COMMENNT
                        , self.str_null(new_line[153:193])  # .replace(" ", "")  # ASHIMOTO_COMMENT
                        , self.str_null(new_line[193:233])  # .replace(" ", "")  # BAGU_COMMENT
                        , self.str_null(new_line[233:273])  # .replace(" ", "")  # RACE_COMMENT
                        , self.str_null(new_line[273:276])  # HAMI
                        , self.str_null(new_line[276:279])  # BANTEGE
                        , self.str_null(new_line[279:282])  # TEITETSU
                        , self.str_null(new_line[282:285])  # HIDUME_JOTAI
                        , self.str_null(new_line[285:288])  # SOE
                        , self.str_null(new_line[288:291])  # KOTURYU
                        , self.get_kaisai_date(filename)  # target_date
                    ], index=names)
                    df = df.append(sr, ignore_index=True)
            df.to_pickle(self.jrdb_folder_path + type + '/' + filename + ".pkl")
            shutil.move(self.jrdb_folder_path + type + '/' + filename, self.jrdb_folder_path + 'backup/' + filename)
        return df

    def get_hjc_df(self, filename):
        names = ['RACE_KEY', '単勝馬番１', '単勝払戻１', '単勝馬番２', '単勝払戻２', '単勝馬番３', '単勝払戻３', '複勝馬番１', '複勝払戻１', '複勝馬番２', '複勝払戻２',
                 '複勝馬番３', '複勝払戻３', '複勝馬番４', '複勝払戻４',
                 '複勝馬番５', '複勝払戻５', '枠連馬番１１', '枠連馬番１２', '枠連払戻１', '枠連馬番２１', '枠連馬番２２', '枠連払戻２', '枠連枠番３１', '枠連枠番３２',
                 '枠連払戻３', '馬連馬番１１', '馬連馬番１２', '馬連払戻１',
                 '馬連馬番２１', '馬連馬番２２', '馬連払戻２', '馬連馬番３１', '馬連馬番３２', '馬連払戻３', 'ワイド馬番１１', 'ワイド馬番１２', 'ワイド払戻１', 'ワイド馬番２１',
                 'ワイド馬番２２', 'ワイド払戻２', 'ワイド馬番３１',
                 'ワイド馬番３２', 'ワイド払戻３', 'ワイド馬番４１', 'ワイド馬番４２', 'ワイド払戻４', 'ワイド馬番５１', 'ワイド馬番５２', 'ワイド払戻５', 'ワイド馬番６１',
                 'ワイド馬番６２', 'ワイド払戻６', 'ワイド馬番７１',
                 'ワイド馬番７２', 'ワイド払戻７', '馬単馬番１１', '馬単馬番１２',
                 '馬単払戻１', '馬単馬番２１', '馬単馬番２２', '馬単払戻２', '馬単馬番３１', '馬単馬番３２', '馬単払戻３', '馬単馬番４１', '馬単馬番４２', '馬単払戻４',
                 '馬単馬番５１', '馬単馬番５２', '馬単払戻５', '馬単馬番６１',
                 '馬単馬番６２', '馬単払戻６', '３連複馬番１１', '３連複馬番１２', '３連複馬番１３', '３連複払戻１', '３連複馬番２１', '３連複馬番２２', '３連複馬番２３',
                 '３連複払戻２', '３連複馬番３１', '３連複馬番３２', '３連複馬番３３',
                 '３連複払戻３', '３連単馬番１１', '３連単馬番１２', '３連単馬番１３', '３連単払戻１', '３連単馬番２１', '３連単馬番２２', '３連単馬番２３', '３連単払戻２',
                 '３連単馬番３１', '３連単馬番３２', '３連単馬番３３', '３連単払戻３',
                 '３連単馬番４１', '３連単馬番４２', '３連単馬番４３', '３連単払戻４', '３連単馬番５１', '３連単馬番５２', '３連単馬番５３', '３連単払戻５', '３連単馬番６１',
                 '３連単馬番６２', '３連単馬番６３', '３連単払戻６', 'target_date']
        if filename[-3:] == "pkl":
            df = pd.read_pickle(self.jrdb_folder_path + 'HJC/' + filename)
        else:
            print(filename)
            with open(self.jrdb_folder_path + 'HJC/' + filename, 'r', encoding="ms932") as fh:
                df = pd.DataFrame(index=[], columns=names)
                for line in fh:
                    new_line = self.replace_line(line)
                    sr = pd.Series([
                        new_line[0:8]  # RACE_KEY
                        # TANSHO1_UMABAN
                        # TANSHO1_HARAIMODOSHI
                        # TANSHO2_UMABAN
                        # TANSHO2_HARAIMODOSHI
                        # TANSHO3_UMABAN
                        # TANSHO3_HARAIMODOSHI
                        # FUKUSHO1_UMABAN
                        # FUKUAHO1_HARAIMODOSHI
                        # FUKUSHO2_UMABAN
                        # FUKUAHO2_HARAIMODOSHI
                        # FUKUSHO3_UMABAN
                        # FUKUAHO3_HARAIMODOSHI
                        # FUKUSHO4_UMABAN
                        # FUKUAHO4_HARAIMODOSHI
                        # FUKUSHO5_UMABAN
                        # FUKUAHO5_HARAIMODOSHI
                        # WAKUREN1_WAKUBAN1
                        # WAKUREN1_WAKUBAN2
                        # WAKUREN1_HARAIMODOSHI
                        # WAKUREN2_WAKUBAN1
                        # WAKUREN2_WAKUBAN2
                        # WAKUREN2_HARAIMODOSHI
                        # WAKUREN3_WAKUBAN1
                        # WAKUREN3_WAKUBAN2
                        # WAKUREN3_HARAIMODOSHI
                        # UMAREN1_UMABAN1
                        # UMAREN1_UMABAN2
                        # UMAREN1_HARAIMODOSHI
                        # UMAREN2_UMABAN1
                        # UMAREN2_UMABAN2
                        # UMAREN2_HARAIMODOSHI
                        # UMAREN3_UMABAN1
                        # UMAREN3_UMABAN2
                        # UMAREN3_HARAIMODOSHI
                        # WIDE1_UMABAN1
                        # WIDE1_UMABAN2
                        # WIDE1_HARAIMODOSHI
                        # WIDE2_UMABAN1
                        # WIDE2_UMABAN2
                        # WIDE2_HARAIMODOSHI
                        # WIDE3_UMABAN1
                        # WIDE3_UMABAN2
                        # WIDE3_HARAIMODOSHI
                        # WIDE4_UMABAN1
                        # WIDE4_UMABAN2
                        # WIDE4_HARAIMODOSHI
                        # WIDE5_UMABAN1
                        # WIDE5_UMABAN2
                        # WIDE5_HARAIMODOSHI
                        # WIDE6_UMABAN1
                        # WIDE6_UMABAN2
                        # WIDE6_HARAIMODOSHI
                        # WIDE7_UMABAN1
                        # WIDE7_UMABAN2
                        # WIDE7_HARAIMODOSHI
                        # UMATAN1_UMABAN1
                        # UMATAN1_UMABAN2
                        # UMATAN1_HARAIMODOSHI
                        # UMATAN2_UMABAN1
                        # UMATAN2_UMABAN2
                        # UMATAN2_HARAIMODOSHI
                        # UMATAN3_UMABAN1
                        # UMATAN3_UMABAN2
                        # UMATAN3_HARAIMODOSHI
                        # UMATAN4_UMABAN1
                        # UMATAN4_UMABAN2
                        # UMATAN4_HARAIMODOSHI
                        # UMATAN5_UMABAN1
                        # UMATAN5_UMABAN2
                        # UMATAN5_HARAIMODOSHI
                        # UMATAN6_UMABAN1
                        # UMATAN6_UMABAN2
                        # UMATAN6_HARAIMODOSHI
                        # SANRENPUKU1_UMABAN1
                        # SANRENPUKU1_UMABAN2
                        # SANRENPUKU1_UMABAN3
                        # SANRENPUKU1_HARAIMODOSHI
                        # SANRENPUKU2_UMABAN1
                        # SANRENPUKU2_UMABAN2
                        # SANRENPUKU2_UMABAN3
                        # SANRENPUKU2_HARAIMODOSHI
                        # SANRENPUKU3_UMABAN1
                        # SANRENPUKU3_UMABAN2
                        # SANRENPUKU3_UMABAN3
                        # SANRENPUKU3_HARAIMODOSHI
                        # SANRENTAN1_UMABAN1
                        # SANRENTAN1_UMABAN2
                        # SANRENTAN1_UMABAN3
                        # SANRENTAN1_HARAIMODOSHI
                        # SANRENTAN2_UMABAN1
                        # SANRENTAN2_UMABAN2
                        # SANRENTAN2_UMABAN3
                        # SANRENTAN2_HARAIMODOSHI
                        # SANRENTAN3_UMABAN1
                        # SANRENTAN3_UMABAN2
                        # SANRENTAN3_UMABAN3
                        # SANRENTAN3_HARAIMODOSHI
                        # SANRENTAN4_UMABAN1
                        # SANRENTAN4_UMABAN2
                        # SANRENTAN4_UMABAN3
                        # SANRENTAN4_HARAIMODOSHI
                        # SANRENTAN5_UMABAN1
                        # SANRENTAN5_UMABAN2
                        # SANRENTAN5_UMABAN3
                        # SANRENTAN5_HARAIMODOSHI
                        # SANRENTAN6_UMABAN1
                        # SANRENTAN6_UMABAN2
                        # SANRENTAN6_UMABAN3
                        # SANRENTAN6_HARAIMODOSHI
                        , new_line[8 + 2 * 0 + 7 * 0: 8 + 2 * 1 + 7 * 0]
                        , self.int_0(new_line[8 + 2 * 1 + 7 * 0: 8 + 2 * 1 + 7 * 1])
                        , new_line[8 + 2 * 1 + 7 * 1: 8 + 2 * 2 + 7 * 1]
                        , self.int_0(new_line[8 + 2 * 2 + 7 * 1: 8 + 2 * 2 + 7 * 2])
                        , new_line[8 + 2 * 2 + 7 * 2: 8 + 2 * 3 + 7 * 2]
                        , self.int_0(new_line[8 + 2 * 3 + 7 * 2: 8 + 2 * 3 + 7 * 3])
                        , new_line[35 + 2 * 0 + 7 * 0: 35 + 2 * 1 + 7 * 0]
                        , self.int_0(new_line[35 + 2 * 1 + 7 * 0: 35 + 2 * 1 + 7 * 1])
                        , new_line[35 + 2 * 1 + 7 * 1: 35 + 2 * 2 + 7 * 1]
                        , self.int_0(new_line[35 + 2 * 2 + 7 * 1: 35 + 2 * 2 + 7 * 2])
                        , new_line[35 + 2 * 2 + 7 * 2: 35 + 2 * 3 + 7 * 2]
                        , self.int_0(new_line[35 + 2 * 3 + 7 * 2: 35 + 2 * 3 + 7 * 3])
                        , new_line[35 + 2 * 3 + 7 * 3: 35 + 2 * 4 + 7 * 3]
                        , self.int_0(new_line[35 + 2 * 4 + 7 * 3: 35 + 2 * 4 + 7 * 4])
                        , new_line[35 + 2 * 4 + 7 * 4: 35 + 2 * 5 + 7 * 4]
                        , self.int_0(new_line[35 + 2 * 5 + 7 * 4: 35 + 2 * 5 + 7 * 5])
                        , new_line[80 + 1 * 0 + 7 * 0: 80 + 1 * 1 + 7 * 0]
                        , new_line[80 + 1 * 1 + 7 * 0: 80 + 1 * 2 + 7 * 0]
                        , self.int_0(new_line[80 + 1 * 2 + 7 * 0: 80 + 1 * 2 + 7 * 1])
                        , new_line[80 + 1 * 2 + 7 * 1: 80 + 1 * 3 + 7 * 1]
                        , new_line[80 + 1 * 3 + 7 * 1: 80 + 1 * 4 + 7 * 1]
                        , self.int_0(new_line[80 + 1 * 4 + 7 * 1: 80 + 1 * 4 + 7 * 2])
                        , new_line[80 + 1 * 4 + 7 * 2: 80 + 1 * 5 + 7 * 2]
                        , new_line[80 + 1 * 5 + 7 * 2: 80 + 1 * 6 + 7 * 2]
                        , self.int_0(new_line[80 + 1 * 6 + 7 * 2: 80 + 1 * 6 + 7 * 3])
                        , new_line[107 + 2 * 0 + 8 * 0:107 + 2 * 1 + 8 * 0]
                        , new_line[107 + 2 * 1 + 8 * 0:107 + 2 * 2 + 8 * 0]
                        , self.int_0(new_line[107 + 2 * 2 + 8 * 0:107 + 2 * 2 + 8 * 1])
                        , new_line[107 + 2 * 2 + 8 * 1:107 + 2 * 3 + 8 * 1]
                        , new_line[107 + 2 * 3 + 8 * 1:107 + 2 * 4 + 8 * 1]
                        , self.int_0(new_line[107 + 2 * 4 + 8 * 1:107 + 2 * 4 + 8 * 2])
                        , new_line[107 + 2 * 4 + 8 * 2:107 + 2 * 5 + 8 * 2]
                        , new_line[107 + 2 * 5 + 8 * 2:107 + 2 * 6 + 8 * 2]
                        , self.int_0(new_line[107 + 2 * 6 + 8 * 2:107 + 2 * 6 + 8 * 3])
                        , new_line[143 + 2 * 0 + 8 * 0: 143 + 2 * 1 + 8 * 0]
                        , new_line[143 + 2 * 1 + 8 * 0: 143 + 2 * 2 + 8 * 0]
                        , self.int_0(new_line[143 + 2 * 2 + 8 * 0: 143 + 2 * 2 + 8 * 1])
                        , new_line[143 + 2 * 2 + 8 * 1: 143 + 2 * 3 + 8 * 1]
                        , new_line[143 + 2 * 3 + 8 * 1: 143 + 2 * 4 + 8 * 1]
                        , self.int_0(new_line[143 + 2 * 4 + 8 * 1: 143 + 2 * 4 + 8 * 2])
                        , new_line[143 + 2 * 4 + 8 * 2: 143 + 2 * 5 + 8 * 2]
                        , new_line[143 + 2 * 5 + 8 * 2: 143 + 2 * 6 + 8 * 2]
                        , self.int_0(new_line[143 + 2 * 6 + 8 * 2: 143 + 2 * 6 + 8 * 3])
                        , new_line[143 + 2 * 6 + 8 * 3: 143 + 2 * 7 + 8 * 3]
                        , new_line[143 + 2 * 7 + 8 * 3: 143 + 2 * 8 + 8 * 3]
                        , self.int_0(new_line[143 + 2 * 8 + 8 * 3: 143 + 2 * 8 + 8 * 4])
                        , new_line[143 + 2 * 8 + 8 * 4: 143 + 2 * 9 + 8 * 4]
                        , new_line[143 + 2 * 9 + 8 * 4: 143 + 2 * 10 + 8 * 4]
                        , self.int_0(new_line[143 + 2 * 10 + 8 * 4: 143 + 2 * 10 + 8 * 5])
                        , new_line[143 + 2 * 10 + 8 * 5: 143 + 2 * 11 + 8 * 5]
                        , new_line[143 + 2 * 11 + 8 * 5: 143 + 2 * 12 + 8 * 5]
                        , self.int_0(new_line[143 + 2 * 12 + 8 * 5: 143 + 2 * 12 + 8 * 6])
                        , new_line[143 + 2 * 12 + 8 * 6: 143 + 2 * 13 + 8 * 6]
                        , new_line[143 + 2 * 13 + 8 * 6: 143 + 2 * 14 + 8 * 6]
                        , self.int_0(new_line[143 + 2 * 14 + 8 * 6: 143 + 2 * 14 + 8 * 7])
                        , new_line[227 + 2 * 0 + 8 * 0: 227 + 2 * 1 + 8 * 0]
                        , new_line[227 + 2 * 1 + 8 * 0: 227 + 2 * 2 + 8 * 0]
                        , self.int_0(new_line[227 + 2 * 2 + 8 * 0: 227 + 2 * 2 + 8 * 1])
                        , new_line[227 + 2 * 2 + 8 * 1: 227 + 2 * 3 + 8 * 1]
                        , new_line[227 + 2 * 3 + 8 * 1: 227 + 2 * 4 + 8 * 1]
                        , self.int_0(new_line[227 + 2 * 4 + 8 * 1: 227 + 2 * 4 + 8 * 2])
                        , new_line[227 + 2 * 4 + 8 * 2: 227 + 2 * 5 + 8 * 2]
                        , new_line[227 + 2 * 5 + 8 * 2: 227 + 2 * 6 + 8 * 2]
                        , self.int_0(new_line[227 + 2 * 6 + 8 * 2: 227 + 2 * 6 + 8 * 3])
                        , new_line[227 + 2 * 6 + 8 * 3: 227 + 2 * 7 + 8 * 3]
                        , new_line[227 + 2 * 7 + 8 * 3: 227 + 2 * 8 + 8 * 3]
                        , self.int_0(new_line[227 + 2 * 8 + 8 * 3: 227 + 2 * 8 + 8 * 4])
                        , new_line[227 + 2 * 8 + 8 * 4: 227 + 2 * 9 + 8 * 4]
                        , new_line[227 + 2 * 9 + 8 * 4: 227 + 2 * 10 + 8 * 4]
                        , self.int_0(new_line[227 + 2 * 10 + 8 * 4: 227 + 2 * 10 + 8 * 5])
                        , new_line[227 + 2 * 10 + 8 * 5: 227 + 2 * 11 + 8 * 5]
                        , new_line[227 + 2 * 11 + 8 * 5: 227 + 2 * 12 + 8 * 5]
                        , self.int_0(new_line[227 + 2 * 12 + 8 * 5: 227 + 2 * 12 + 8 * 6])
                        , new_line[299 + 2 * 0 + 8 * 0: 299 + 2 * 1 + 8 * 0]
                        , new_line[299 + 2 * 1 + 8 * 0: 299 + 2 * 2 + 8 * 0]
                        , new_line[299 + 2 * 2 + 8 * 0: 299 + 2 * 3 + 8 * 0]
                        , self.int_0(new_line[299 + 2 * 3 + 8 * 0: 299 + 2 * 3 + 8 * 1])
                        , new_line[299 + 2 * 3 + 8 * 1: 299 + 2 * 4 + 8 * 1]
                        , new_line[299 + 2 * 4 + 8 * 1: 299 + 2 * 5 + 8 * 1]
                        , new_line[299 + 2 * 5 + 8 * 1: 299 + 2 * 6 + 8 * 1]
                        , self.int_0(new_line[299 + 2 * 6 + 8 * 1: 299 + 2 * 6 + 8 * 2])
                        , new_line[299 + 2 * 6 + 8 * 2: 299 + 2 * 7 + 8 * 2]
                        , new_line[299 + 2 * 7 + 8 * 2: 299 + 2 * 8 + 8 * 2]
                        , new_line[299 + 2 * 8 + 8 * 2: 299 + 2 * 9 + 8 * 2]
                        , self.int_0(new_line[299 + 2 * 9 + 8 * 2: 299 + 2 * 9 + 8 * 3])
                        , new_line[341 + 2 * 0 + 9 * 0: 341 + 2 * 1 + 9 * 0]
                        , new_line[341 + 2 * 1 + 9 * 0: 341 + 2 * 2 + 9 * 0]
                        , new_line[341 + 2 * 2 + 9 * 0: 341 + 2 * 3 + 9 * 0]
                        , self.int_0(new_line[341 + 2 * 3 + 9 * 0: 341 + 2 * 3 + 9 * 1])
                        , new_line[341 + 2 * 3 + 9 * 1: 341 + 2 * 4 + 9 * 1]
                        , new_line[341 + 2 * 4 + 9 * 1: 341 + 2 * 5 + 9 * 1]
                        , new_line[341 + 2 * 5 + 9 * 1: 341 + 2 * 6 + 9 * 1]
                        , self.int_0(new_line[341 + 2 * 6 + 9 * 1: 341 + 2 * 6 + 9 * 2])
                        , new_line[341 + 2 * 6 + 9 * 2: 341 + 2 * 7 + 9 * 2]
                        , new_line[341 + 2 * 7 + 9 * 2: 341 + 2 * 8 + 9 * 2]
                        , new_line[341 + 2 * 8 + 9 * 2: 341 + 2 * 9 + 9 * 2]
                        , self.int_0(new_line[341 + 2 * 9 + 9 * 2: 341 + 2 * 9 + 9 * 3])
                        , new_line[341 + 2 * 9 + 9 * 3: 341 + 2 * 10 + 9 * 3]
                        , new_line[341 + 2 * 10 + 9 * 3: 341 + 2 * 11 + 9 * 3]
                        , new_line[341 + 2 * 11 + 9 * 3: 341 + 2 * 12 + 9 * 3]
                        , self.int_0(new_line[341 + 2 * 12 + 9 * 3: 341 + 2 * 12 + 9 * 4])
                        , new_line[341 + 2 * 12 + 9 * 4: 341 + 2 * 13 + 9 * 4]
                        , new_line[341 + 2 * 13 + 9 * 4: 341 + 2 * 14 + 9 * 4]
                        , new_line[341 + 2 * 14 + 9 * 4: 341 + 2 * 15 + 9 * 4]
                        , self.int_0(new_line[341 + 2 * 15 + 9 * 4: 341 + 2 * 15 + 9 * 5])
                        , new_line[341 + 2 * 15 + 9 * 5: 341 + 2 * 16 + 9 * 5]
                        , new_line[341 + 2 * 16 + 9 * 5: 341 + 2 * 17 + 9 * 5]
                        , new_line[341 + 2 * 17 + 9 * 5: 341 + 2 * 18 + 9 * 5]
                        , self.int_0(new_line[341 + 2 * 18 + 9 * 5: 341 + 2 * 18 + 9 * 6])
                        , self.get_kaisai_date(filename)  # NENGAPPI
                    ], index=names)
                    df = df.append(sr, ignore_index=True)
            df.to_pickle(self.jrdb_folder_path + 'HJC/' + filename + ".pkl")
            shutil.move(self.jrdb_folder_path + 'HJC/' + filename, self.jrdb_folder_path + 'backup/' + filename)
        return df

    def get_hjc_sokuho_df(self, filename):
        names = ['RACE_KEY', '単勝馬番１', '単勝払戻１', '単勝馬番２', '単勝払戻２', '単勝馬番３', '単勝払戻３', '複勝馬番１', '複勝払戻１', '複勝馬番２', '複勝払戻２',
                 '複勝馬番３', '複勝払戻３', '複勝馬番４', '複勝払戻４',
                 '複勝馬番５', '複勝払戻５', '枠連馬番１１', '枠連馬番１２', '枠連払戻１', '枠連馬番２１', '枠連馬番２２', '枠連払戻２', '枠連枠番３１', '枠連枠番３２',
                 '枠連払戻３', '馬連馬番１１', '馬連馬番１２', '馬連払戻１',
                 '馬連馬番２１', '馬連馬番２２', '馬連払戻２', '馬連馬番３１', '馬連馬番３２', '馬連払戻３', 'ワイド馬番１１', 'ワイド馬番１２', 'ワイド払戻１', 'ワイド馬番２１',
                 'ワイド馬番２２', 'ワイド払戻２', 'ワイド馬番３１',
                 'ワイド馬番３２', 'ワイド払戻３', 'ワイド馬番４１', 'ワイド馬番４２', 'ワイド払戻４', 'ワイド馬番５１', 'ワイド馬番５２', 'ワイド払戻５', 'ワイド馬番６１',
                 'ワイド馬番６２', 'ワイド払戻６', 'ワイド馬番７１',
                 'ワイド馬番７２', 'ワイド払戻７', '馬単馬番１１', '馬単馬番１２',
                 '馬単払戻１', '馬単馬番２１', '馬単馬番２２', '馬単払戻２', '馬単馬番３１', '馬単馬番３２', '馬単払戻３', '馬単馬番４１', '馬単馬番４２', '馬単払戻４',
                 '馬単馬番５１', '馬単馬番５２', '馬単払戻５', '馬単馬番６１',
                 '馬単馬番６２', '馬単払戻６', '３連複馬番１１', '３連複馬番１２', '３連複馬番１３', '３連複払戻１', '３連複馬番２１', '３連複馬番２２', '３連複馬番２３',
                 '３連複払戻２', '３連複馬番３１', '３連複馬番３２', '３連複馬番３３',
                 '３連複払戻３', '３連単馬番１１', '３連単馬番１２', '３連単馬番１３', '３連単払戻１', '３連単馬番２１', '３連単馬番２２', '３連単馬番２３', '３連単払戻２',
                 '３連単馬番３１', '３連単馬番３２', '３連単馬番３３', '３連単払戻３',
                 '３連単馬番４１', '３連単馬番４２', '３連単馬番４３', '３連単払戻４', '３連単馬番５１', '３連単馬番５２', '３連単馬番５３', '３連単払戻５', '３連単馬番６１',
                 '３連単馬番６２', '３連単馬番６３', '３連単払戻６', 'target_date']
        print(filename)
        with open(self.jrdb_folder_path + 'sokuho/' + filename, 'r', encoding="ms932") as fh:
            df = pd.DataFrame(index=[], columns=names)
            for line in fh:
                new_line = self.replace_line(line)
                sr = pd.Series([
                    new_line[0:8]  # RACE_KEY
                    # TANSHO1_UMABAN
                    # TANSHO1_HARAIMODOSHI
                    # TANSHO2_UMABAN
                    # TANSHO2_HARAIMODOSHI
                    # TANSHO3_UMABAN
                    # TANSHO3_HARAIMODOSHI
                    # FUKUSHO1_UMABAN
                    # FUKUAHO1_HARAIMODOSHI
                    # FUKUSHO2_UMABAN
                    # FUKUAHO2_HARAIMODOSHI
                    # FUKUSHO3_UMABAN
                    # FUKUAHO3_HARAIMODOSHI
                    # FUKUSHO4_UMABAN
                    # FUKUAHO4_HARAIMODOSHI
                    # FUKUSHO5_UMABAN
                    # FUKUAHO5_HARAIMODOSHI
                    # WAKUREN1_WAKUBAN1
                    # WAKUREN1_WAKUBAN2
                    # WAKUREN1_HARAIMODOSHI
                    # WAKUREN2_WAKUBAN1
                    # WAKUREN2_WAKUBAN2
                    # WAKUREN2_HARAIMODOSHI
                    # WAKUREN3_WAKUBAN1
                    # WAKUREN3_WAKUBAN2
                    # WAKUREN3_HARAIMODOSHI
                    # UMAREN1_UMABAN1
                    # UMAREN1_UMABAN2
                    # UMAREN1_HARAIMODOSHI
                    # UMAREN2_UMABAN1
                    # UMAREN2_UMABAN2
                    # UMAREN2_HARAIMODOSHI
                    # UMAREN3_UMABAN1
                    # UMAREN3_UMABAN2
                    # UMAREN3_HARAIMODOSHI
                    # WIDE1_UMABAN1
                    # WIDE1_UMABAN2
                    # WIDE1_HARAIMODOSHI
                    # WIDE2_UMABAN1
                    # WIDE2_UMABAN2
                    # WIDE2_HARAIMODOSHI
                    # WIDE3_UMABAN1
                    # WIDE3_UMABAN2
                    # WIDE3_HARAIMODOSHI
                    # WIDE4_UMABAN1
                    # WIDE4_UMABAN2
                    # WIDE4_HARAIMODOSHI
                    # WIDE5_UMABAN1
                    # WIDE5_UMABAN2
                    # WIDE5_HARAIMODOSHI
                    # WIDE6_UMABAN1
                    # WIDE6_UMABAN2
                    # WIDE6_HARAIMODOSHI
                    # WIDE7_UMABAN1
                    # WIDE7_UMABAN2
                    # WIDE7_HARAIMODOSHI
                    # UMATAN1_UMABAN1
                    # UMATAN1_UMABAN2
                    # UMATAN1_HARAIMODOSHI
                    # UMATAN2_UMABAN1
                    # UMATAN2_UMABAN2
                    # UMATAN2_HARAIMODOSHI
                    # UMATAN3_UMABAN1
                    # UMATAN3_UMABAN2
                    # UMATAN3_HARAIMODOSHI
                    # UMATAN4_UMABAN1
                    # UMATAN4_UMABAN2
                    # UMATAN4_HARAIMODOSHI
                    # UMATAN5_UMABAN1
                    # UMATAN5_UMABAN2
                    # UMATAN5_HARAIMODOSHI
                    # UMATAN6_UMABAN1
                    # UMATAN6_UMABAN2
                    # UMATAN6_HARAIMODOSHI
                    # SANRENPUKU1_UMABAN1
                    # SANRENPUKU1_UMABAN2
                    # SANRENPUKU1_UMABAN3
                    # SANRENPUKU1_HARAIMODOSHI
                    # SANRENPUKU2_UMABAN1
                    # SANRENPUKU2_UMABAN2
                    # SANRENPUKU2_UMABAN3
                    # SANRENPUKU2_HARAIMODOSHI
                    # SANRENPUKU3_UMABAN1
                    # SANRENPUKU3_UMABAN2
                    # SANRENPUKU3_UMABAN3
                    # SANRENPUKU3_HARAIMODOSHI
                    # SANRENTAN1_UMABAN1
                    # SANRENTAN1_UMABAN2
                    # SANRENTAN1_UMABAN3
                    # SANRENTAN1_HARAIMODOSHI
                    # SANRENTAN2_UMABAN1
                    # SANRENTAN2_UMABAN2
                    # SANRENTAN2_UMABAN3
                    # SANRENTAN2_HARAIMODOSHI
                    # SANRENTAN3_UMABAN1
                    # SANRENTAN3_UMABAN2
                    # SANRENTAN3_UMABAN3
                    # SANRENTAN3_HARAIMODOSHI
                    # SANRENTAN4_UMABAN1
                    # SANRENTAN4_UMABAN2
                    # SANRENTAN4_UMABAN3
                    # SANRENTAN4_HARAIMODOSHI
                    # SANRENTAN5_UMABAN1
                    # SANRENTAN5_UMABAN2
                    # SANRENTAN5_UMABAN3
                    # SANRENTAN5_HARAIMODOSHI
                    # SANRENTAN6_UMABAN1
                    # SANRENTAN6_UMABAN2
                    # SANRENTAN6_UMABAN3
                    # SANRENTAN6_HARAIMODOSHI
                    , new_line[8 + 2 * 0 + 7 * 0: 8 + 2 * 1 + 7 * 0]
                    , self.int_0(new_line[8 + 2 * 1 + 7 * 0: 8 + 2 * 1 + 7 * 1])
                    , new_line[8 + 2 * 1 + 7 * 1: 8 + 2 * 2 + 7 * 1]
                    , self.int_0(new_line[8 + 2 * 2 + 7 * 1: 8 + 2 * 2 + 7 * 2])
                    , new_line[8 + 2 * 2 + 7 * 2: 8 + 2 * 3 + 7 * 2]
                    , self.int_0(new_line[8 + 2 * 3 + 7 * 2: 8 + 2 * 3 + 7 * 3])
                    , new_line[35 + 2 * 0 + 7 * 0: 35 + 2 * 1 + 7 * 0]
                    , self.int_0(new_line[35 + 2 * 1 + 7 * 0: 35 + 2 * 1 + 7 * 1])
                    , new_line[35 + 2 * 1 + 7 * 1: 35 + 2 * 2 + 7 * 1]
                    , self.int_0(new_line[35 + 2 * 2 + 7 * 1: 35 + 2 * 2 + 7 * 2])
                    , new_line[35 + 2 * 2 + 7 * 2: 35 + 2 * 3 + 7 * 2]
                    , self.int_0(new_line[35 + 2 * 3 + 7 * 2: 35 + 2 * 3 + 7 * 3])
                    , new_line[35 + 2 * 3 + 7 * 3: 35 + 2 * 4 + 7 * 3]
                    , self.int_0(new_line[35 + 2 * 4 + 7 * 3: 35 + 2 * 4 + 7 * 4])
                    , new_line[35 + 2 * 4 + 7 * 4: 35 + 2 * 5 + 7 * 4]
                    , self.int_0(new_line[35 + 2 * 5 + 7 * 4: 35 + 2 * 5 + 7 * 5])
                    , new_line[80 + 1 * 0 + 7 * 0: 80 + 1 * 1 + 7 * 0]
                    , new_line[80 + 1 * 1 + 7 * 0: 80 + 1 * 2 + 7 * 0]
                    , self.int_0(new_line[80 + 1 * 2 + 7 * 0: 80 + 1 * 2 + 7 * 1])
                    , new_line[80 + 1 * 2 + 7 * 1: 80 + 1 * 3 + 7 * 1]
                    , new_line[80 + 1 * 3 + 7 * 1: 80 + 1 * 4 + 7 * 1]
                    , self.int_0(new_line[80 + 1 * 4 + 7 * 1: 80 + 1 * 4 + 7 * 2])
                    , new_line[80 + 1 * 4 + 7 * 2: 80 + 1 * 5 + 7 * 2]
                    , new_line[80 + 1 * 5 + 7 * 2: 80 + 1 * 6 + 7 * 2]
                    , self.int_0(new_line[80 + 1 * 6 + 7 * 2: 80 + 1 * 6 + 7 * 3])
                    , new_line[107 + 2 * 0 + 8 * 0:107 + 2 * 1 + 8 * 0]
                    , new_line[107 + 2 * 1 + 8 * 0:107 + 2 * 2 + 8 * 0]
                    , self.int_0(new_line[107 + 2 * 2 + 8 * 0:107 + 2 * 2 + 8 * 1])
                    , new_line[107 + 2 * 2 + 8 * 1:107 + 2 * 3 + 8 * 1]
                    , new_line[107 + 2 * 3 + 8 * 1:107 + 2 * 4 + 8 * 1]
                    , self.int_0(new_line[107 + 2 * 4 + 8 * 1:107 + 2 * 4 + 8 * 2])
                    , new_line[107 + 2 * 4 + 8 * 2:107 + 2 * 5 + 8 * 2]
                    , new_line[107 + 2 * 5 + 8 * 2:107 + 2 * 6 + 8 * 2]
                    , self.int_0(new_line[107 + 2 * 6 + 8 * 2:107 + 2 * 6 + 8 * 3])
                    , new_line[143 + 2 * 0 + 8 * 0: 143 + 2 * 1 + 8 * 0]
                    , new_line[143 + 2 * 1 + 8 * 0: 143 + 2 * 2 + 8 * 0]
                    , self.int_0(new_line[143 + 2 * 2 + 8 * 0: 143 + 2 * 2 + 8 * 1])
                    , new_line[143 + 2 * 2 + 8 * 1: 143 + 2 * 3 + 8 * 1]
                    , new_line[143 + 2 * 3 + 8 * 1: 143 + 2 * 4 + 8 * 1]
                    , self.int_0(new_line[143 + 2 * 4 + 8 * 1: 143 + 2 * 4 + 8 * 2])
                    , new_line[143 + 2 * 4 + 8 * 2: 143 + 2 * 5 + 8 * 2]
                    , new_line[143 + 2 * 5 + 8 * 2: 143 + 2 * 6 + 8 * 2]
                    , self.int_0(new_line[143 + 2 * 6 + 8 * 2: 143 + 2 * 6 + 8 * 3])
                    , new_line[143 + 2 * 6 + 8 * 3: 143 + 2 * 7 + 8 * 3]
                    , new_line[143 + 2 * 7 + 8 * 3: 143 + 2 * 8 + 8 * 3]
                    , self.int_0(new_line[143 + 2 * 8 + 8 * 3: 143 + 2 * 8 + 8 * 4])
                    , new_line[143 + 2 * 8 + 8 * 4: 143 + 2 * 9 + 8 * 4]
                    , new_line[143 + 2 * 9 + 8 * 4: 143 + 2 * 10 + 8 * 4]
                    , self.int_0(new_line[143 + 2 * 10 + 8 * 4: 143 + 2 * 10 + 8 * 5])
                    , new_line[143 + 2 * 10 + 8 * 5: 143 + 2 * 11 + 8 * 5]
                    , new_line[143 + 2 * 11 + 8 * 5: 143 + 2 * 12 + 8 * 5]
                    , self.int_0(new_line[143 + 2 * 12 + 8 * 5: 143 + 2 * 12 + 8 * 6])
                    , new_line[143 + 2 * 12 + 8 * 6: 143 + 2 * 13 + 8 * 6]
                    , new_line[143 + 2 * 13 + 8 * 6: 143 + 2 * 14 + 8 * 6]
                    , self.int_0(new_line[143 + 2 * 14 + 8 * 6: 143 + 2 * 14 + 8 * 7])
                    , new_line[227 + 2 * 0 + 8 * 0: 227 + 2 * 1 + 8 * 0]
                    , new_line[227 + 2 * 1 + 8 * 0: 227 + 2 * 2 + 8 * 0]
                    , self.int_0(new_line[227 + 2 * 2 + 8 * 0: 227 + 2 * 2 + 8 * 1])
                    , new_line[227 + 2 * 2 + 8 * 1: 227 + 2 * 3 + 8 * 1]
                    , new_line[227 + 2 * 3 + 8 * 1: 227 + 2 * 4 + 8 * 1]
                    , self.int_0(new_line[227 + 2 * 4 + 8 * 1: 227 + 2 * 4 + 8 * 2])
                    , new_line[227 + 2 * 4 + 8 * 2: 227 + 2 * 5 + 8 * 2]
                    , new_line[227 + 2 * 5 + 8 * 2: 227 + 2 * 6 + 8 * 2]
                    , self.int_0(new_line[227 + 2 * 6 + 8 * 2: 227 + 2 * 6 + 8 * 3])
                    , new_line[227 + 2 * 6 + 8 * 3: 227 + 2 * 7 + 8 * 3]
                    , new_line[227 + 2 * 7 + 8 * 3: 227 + 2 * 8 + 8 * 3]
                    , self.int_0(new_line[227 + 2 * 8 + 8 * 3: 227 + 2 * 8 + 8 * 4])
                    , new_line[227 + 2 * 8 + 8 * 4: 227 + 2 * 9 + 8 * 4]
                    , new_line[227 + 2 * 9 + 8 * 4: 227 + 2 * 10 + 8 * 4]
                    , self.int_0(new_line[227 + 2 * 10 + 8 * 4: 227 + 2 * 10 + 8 * 5])
                    , new_line[227 + 2 * 10 + 8 * 5: 227 + 2 * 11 + 8 * 5]
                    , new_line[227 + 2 * 11 + 8 * 5: 227 + 2 * 12 + 8 * 5]
                    , self.int_0(new_line[227 + 2 * 12 + 8 * 5: 227 + 2 * 12 + 8 * 6])
                    , new_line[299 + 2 * 0 + 8 * 0: 299 + 2 * 1 + 8 * 0]
                    , new_line[299 + 2 * 1 + 8 * 0: 299 + 2 * 2 + 8 * 0]
                    , new_line[299 + 2 * 2 + 8 * 0: 299 + 2 * 3 + 8 * 0]
                    , self.int_0(new_line[299 + 2 * 3 + 8 * 0: 299 + 2 * 3 + 8 * 1])
                    , new_line[299 + 2 * 3 + 8 * 1: 299 + 2 * 4 + 8 * 1]
                    , new_line[299 + 2 * 4 + 8 * 1: 299 + 2 * 5 + 8 * 1]
                    , new_line[299 + 2 * 5 + 8 * 1: 299 + 2 * 6 + 8 * 1]
                    , self.int_0(new_line[299 + 2 * 6 + 8 * 1: 299 + 2 * 6 + 8 * 2])
                    , new_line[299 + 2 * 6 + 8 * 2: 299 + 2 * 7 + 8 * 2]
                    , new_line[299 + 2 * 7 + 8 * 2: 299 + 2 * 8 + 8 * 2]
                    , new_line[299 + 2 * 8 + 8 * 2: 299 + 2 * 9 + 8 * 2]
                    , self.int_0(new_line[299 + 2 * 9 + 8 * 2: 299 + 2 * 9 + 8 * 3])
                    , new_line[341 + 2 * 0 + 9 * 0: 341 + 2 * 1 + 9 * 0]
                    , new_line[341 + 2 * 1 + 9 * 0: 341 + 2 * 2 + 9 * 0]
                    , new_line[341 + 2 * 2 + 9 * 0: 341 + 2 * 3 + 9 * 0]
                    , self.int_0(new_line[341 + 2 * 3 + 9 * 0: 341 + 2 * 3 + 9 * 1])
                    , new_line[341 + 2 * 3 + 9 * 1: 341 + 2 * 4 + 9 * 1]
                    , new_line[341 + 2 * 4 + 9 * 1: 341 + 2 * 5 + 9 * 1]
                    , new_line[341 + 2 * 5 + 9 * 1: 341 + 2 * 6 + 9 * 1]
                    , self.int_0(new_line[341 + 2 * 6 + 9 * 1: 341 + 2 * 6 + 9 * 2])
                    , new_line[341 + 2 * 6 + 9 * 2: 341 + 2 * 7 + 9 * 2]
                    , new_line[341 + 2 * 7 + 9 * 2: 341 + 2 * 8 + 9 * 2]
                    , new_line[341 + 2 * 8 + 9 * 2: 341 + 2 * 9 + 9 * 2]
                    , self.int_0(new_line[341 + 2 * 9 + 9 * 2: 341 + 2 * 9 + 9 * 3])
                    , new_line[341 + 2 * 9 + 9 * 3: 341 + 2 * 10 + 9 * 3]
                    , new_line[341 + 2 * 10 + 9 * 3: 341 + 2 * 11 + 9 * 3]
                    , new_line[341 + 2 * 11 + 9 * 3: 341 + 2 * 12 + 9 * 3]
                    , self.int_0(new_line[341 + 2 * 12 + 9 * 3: 341 + 2 * 12 + 9 * 4])
                    , new_line[341 + 2 * 12 + 9 * 4: 341 + 2 * 13 + 9 * 4]
                    , new_line[341 + 2 * 13 + 9 * 4: 341 + 2 * 14 + 9 * 4]
                    , new_line[341 + 2 * 14 + 9 * 4: 341 + 2 * 15 + 9 * 4]
                    , self.int_0(new_line[341 + 2 * 15 + 9 * 4: 341 + 2 * 15 + 9 * 5])
                    , new_line[341 + 2 * 15 + 9 * 5: 341 + 2 * 16 + 9 * 5]
                    , new_line[341 + 2 * 16 + 9 * 5: 341 + 2 * 17 + 9 * 5]
                    , new_line[341 + 2 * 17 + 9 * 5: 341 + 2 * 18 + 9 * 5]
                    , self.int_0(new_line[341 + 2 * 18 + 9 * 5: 341 + 2 * 18 + 9 * 6])
                    , self.get_kaisai_date(filename)  # NENGAPPI
                ], index=names)
                df = df.append(sr, ignore_index=True)
        return df

    def get_tyb_df(self, filename):
        names = ['RACE_KEY', 'UMABAN', 'オッズ指数', 'パドック指数', '総合指数', '馬具変更情報', '脚元情報', '取消フラグ', '騎手コード', '馬場状態コード',
                 '天候コード', '単勝オッズ', '複勝オッズ', 'オッズ取得時間', '馬体重', '馬体重増減', 'オッズ印', 'パドック印', '直前総合印', 'target_date']
        if filename[-3:] == "pkl":
            df = pd.read_pickle(self.jrdb_folder_path + 'TYB/' + filename)
        else:
            print(filename)
            with open(self.jrdb_folder_path + 'TYB/' + filename, 'r', encoding="ms932") as fh:
                df = pd.DataFrame(index=[], columns=names)
                for line in fh:
                    new_line = self.replace_line(line)
                    sr = pd.Series([
                        new_line[0:8]  # RACE_KEY
                        , self.str_null(new_line[8:10])  # UMABAN
                        , self.float_null(new_line[25:30])  # ODDS_RECORD
                        , self.float_null(new_line[30:35])  # PADOC_RECORD
                        , self.float_null(new_line[40:45])  # TOTAL_RECORD
                        , self.str_null(new_line[45:46])  # BAGU_HENOU
                        , self.str_null(new_line[46:47])  # ASHIMOTO_JOHO
                        , self.str_null(new_line[47:48])  # TORIKESHI_FLAG
                        , self.str_null(new_line[48:53])  # KISHU_CODE
                        , self.str_null(new_line[69:71])  # BABA_JOTAI_CODE
                        , self.str_null(new_line[71:72])  # TENKO_CODE
                        , self.float_null(new_line[72:78])  # TANSHO_ODDS
                        , self.float_null(new_line[78:84])  # FUKUSHO_ODDS
                        , self.str_null(new_line[84:88])  # ODDS_TIME
                        , self.int_0(new_line[88:91])  # BATAIJU
                        , self.int_bataiju_zogen(new_line[91:94])  # ZOGEN
                        , self.str_null(new_line[94:95])  # ODDS_MARK
                        , self.str_null(new_line[95:96])  # PADOC_MARK
                        , self.str_null(new_line[96:97])  # CHOKUZEN_MARK
                        , self.get_kaisai_date(filename)  # target_date
                    ], index=names)
                    df = df.append(sr, ignore_index=True)
            df.to_pickle(self.jrdb_folder_path + 'TYB/' + filename + ".pkl")
            shutil.move(self.jrdb_folder_path + 'TYB/' + filename, self.jrdb_folder_path + 'backup/' + filename)
        return df

    def get_tyb_sokuho_df(self, filename):
        names = ['RACE_KEY', 'UMABAN', 'オッズ指数', 'パドック指数', '総合指数', '馬具変更情報', '脚元情報', '取消フラグ', '騎手コード', '馬場状態コード',
                 '天候コード', '単勝オッズ', '複勝オッズ', 'オッズ取得時間', '馬体重', '馬体重増減', 'オッズ印', 'パドック印', '直前総合印', 'target_date']
        print(filename)
        with open(self.jrdb_folder_path + 'sokuho/' + filename, 'r', encoding="ms932") as fh:
            df = pd.DataFrame(index=[], columns=names)
            for line in fh:
                new_line = self.replace_line(line)
                sr = pd.Series([
                    new_line[0:8]  # RACE_KEY
                    , self.str_null(new_line[8:10])  # UMABAN
                    , self.float_null(new_line[25:30])  # ODDS_RECORD
                    , self.float_null(new_line[30:35])  # PADOC_RECORD
                    , self.float_null(new_line[40:45])  # TOTAL_RECORD
                    , self.str_null(new_line[45:46])  # BAGU_HENOU
                    , self.str_null(new_line[46:47])  # ASHIMOTO_JOHO
                    , self.str_null(new_line[47:48])  # TORIKESHI_FLAG
                    , self.str_null(new_line[48:53])  # KISHU_CODE
                    , self.str_null(new_line[69:71])  # BABA_JOTAI_CODE
                    , self.str_null(new_line[71:72])  # TENKO_CODE
                    , self.float_null(new_line[72:78])  # TANSHO_ODDS
                    , self.float_null(new_line[78:84])  # FUKUSHO_ODDS
                    , self.str_null(new_line[84:88])  # ODDS_TIME
                    , self.int_0(new_line[88:91])  # BATAIJU
                    , self.int_bataiju_zogen(new_line[91:94])  # ZOGEN
                    , self.str_null(new_line[94:95])  # ODDS_MARK
                    , self.str_null(new_line[95:96])  # PADOC_MARK
                    , self.str_null(new_line[96:97])  # CHOKUZEN_MARK
                    , self.get_kaisai_date(filename)  # target_date
                ], index=names)
                df = df.append(sr, ignore_index=True)
        return df

    def get_oz_df(self, filename):
        names = ['RACE_KEY', 'UMABAN', '単勝オッズ', '複勝オッズ', '馬連オッズ０１', '馬連オッズ０２', '馬連オッズ０３', '馬連オッズ０４', '馬連オッズ０５',
                 '馬連オッズ０６', '馬連オッズ０７',
                 '馬連オッズ０８', '馬連オッズ０９', '馬連オッズ１０', '馬連オッズ１１', '馬連オッズ１２', '馬連オッズ１３', '馬連オッズ１４', '馬連オッズ１５', '馬連オッズ１６',
                 '馬連オッズ１７', '馬連オッズ１８', 'target_date']
        if filename[-3:] == "pkl":
            odds_df = pd.read_pickle(self.jrdb_folder_path + 'OZ/' + filename)
        else:
            print(filename)
            with open(self.jrdb_folder_path + 'OZ/' + filename, 'r', encoding="ms932") as fh:
                df = pd.DataFrame(index=[], columns=names)
                target_date = self.get_kaisai_date_for_oz(filename)
                for line in fh:
                    new_line = self.replace_line(line)
                    race_key = new_line[0:8]
                    tosu = int(new_line[8:10])
                    odds_tansho_text = new_line[10:100]
                    odds_tansho_list = deque(
                        [odds_tansho_text[i: i + 5] for i in range(0, len(odds_tansho_text), 5)])
                    odds_fukusho_text = new_line[100:190]
                    odds_fukusho_list = deque(
                        [odds_fukusho_text[i: i + 5] for i in range(0, len(odds_fukusho_text), 5)])
                    odds_umaren_text = new_line[190:955]
                    odds_umaren_list = deque(
                        [odds_umaren_text[i: i + 5] for i in range(0, len(odds_umaren_text), 5)])
                    for uma1 in range(1, 19):
                        base_tansho_odds = self.float_null(
                            odds_tansho_list.popleft())
                        base_fukusho_odds = self.float_null(
                            odds_fukusho_list.popleft())
                        bul = [0] * 19  # base_umaren_list
                        for uma2 in range(uma1 + 1, 19):
                            bul[uma2] = self.float_null(odds_umaren_list.popleft())
                        sr = pd.Series([
                            race_key, str(uma1).zfill(2), base_tansho_odds, base_fukusho_odds, bul[1], bul[2], bul[3],
                            bul[4], bul[5], bul[6], bul[7], bul[
                                8], bul[9], bul[10], bul[11], bul[12], bul[13], bul[14], bul[15], bul[16], bul[17],
                            bul[18], target_date
                        ], index=names)
                        df = df.append(sr, ignore_index=True)
                race_key_list = df["RACE_KEY"].drop_duplicates()
                odds_columns = ['馬連オッズ０１', '馬連オッズ０２', '馬連オッズ０３', '馬連オッズ０４', '馬連オッズ０５', '馬連オッズ０６', '馬連オッズ０７',
                                '馬連オッズ０８', '馬連オッズ０９', '馬連オッズ１０', '馬連オッズ１１', '馬連オッズ１２', '馬連オッズ１３', '馬連オッズ１４', '馬連オッズ１５',
                                '馬連オッズ１６', '馬連オッズ１７', '馬連オッズ１８']
                odds_df = pd.DataFrame()
                for race_key in race_key_list:
                    temp_df = df.query(f"RACE_KEY == '{race_key}'").reset_index(drop=True).copy()
                    odds_np = temp_df[odds_columns].to_numpy()
                    odds_np_t = odds_np.T
                    temp_odds_df = pd.DataFrame(odds_np + odds_np_t, columns=odds_columns)
                    temp_odds_df = pd.concat(
                        [temp_df[["RACE_KEY", "UMABAN", "単勝オッズ", "複勝オッズ", "target_date"]], temp_odds_df], axis=1)
                    odds_df = pd.concat([odds_df, temp_odds_df])
            odds_df.to_pickle(self.jrdb_folder_path + 'OZ/' + filename + ".pkl")
            shutil.move(self.jrdb_folder_path + 'OZ/' + filename, self.jrdb_folder_path + 'backup/' + filename)
        return odds_df

    def get_ot_df(self, filename):
        names = ['RACE_KEY', 'UMABAN_1', 'UMABAN_2', 'UMABAN_3',
                 '３連複オッズ', 'target_date']
        if filename[-3:] == "pkl":
            df = pd.read_pickle(self.jrdb_folder_path + 'OT/' + filename)
        else:
            print(filename)
            with open(self.jrdb_folder_path + 'OT/' + filename, 'r', encoding="ms932") as fh:
                df = pd.DataFrame(index=[], columns=names)
                target_date = self.get_kaisai_date_for_oz(filename)
                print(dt.datetime.now())
                for line in fh:
                    new_line = self.replace_line(line)
                    race_key = new_line[0:8]
                    odds_text = new_line[10:4906]
                    base_odds_list = deque([odds_text[i: i + 6]
                                            for i in range(0, len(odds_text), 6)])
                    for uma1 in range(1, 17):
                        for uma2 in range(uma1 + 1, 18):
                            for uma3 in range(uma2 + 1, 19):
                                base_odds = self.float_null(
                                    base_odds_list.popleft())
                                if base_odds != None:
                                    sr = pd.Series([
                                        race_key, str(uma1).zfill(2), str(uma2).zfill(2), str(
                                            uma3).zfill(2), base_odds, target_date
                                    ], index=names)
                                    df = df.append(sr, ignore_index=True)
            df.to_pickle(self.jrdb_folder_path + 'OT/' + filename + ".pkl")
            shutil.move(self.jrdb_folder_path + 'OT/' + filename, self.jrdb_folder_path + 'backup/' + filename)
        return df

    def get_ow_df(self, filename):
        names = ['RACE_KEY', 'UMABAN_1', 'UMABAN_2', 'ワイドオッズ', 'target_date']
        if filename[-3:] == "pkl":
            df = pd.read_pickle(self.jrdb_folder_path + 'OW/' + filename)
        else:
            print(filename)
            with open(self.jrdb_folder_path + 'OW/' + filename, 'r', encoding="ms932") as fh:
                df = pd.DataFrame(index=[], columns=names)
                target_date = self.get_kaisai_date_for_oz(filename)
                print(dt.datetime.now())
                for line in fh:
                    new_line = self.replace_line(line)
                    race_key = new_line[0:8]
                    odds_text = new_line[10:775]
                    base_odds_list = deque([odds_text[i: i + 5]
                                            for i in range(0, len(odds_text), 5)])
                    for uma1 in range(1, 18):
                        for uma2 in range(uma1 + 1, 19):
                            base_odds = self.float_null(
                                base_odds_list.popleft())
                            if base_odds != None:
                                sr = pd.Series([
                                    race_key, str(uma1).zfill(2), str(uma2).zfill(2), base_odds, target_date
                                ], index=names)
                                df = df.append(sr, ignore_index=True)
            df.to_pickle(self.jrdb_folder_path + 'OW/' + filename + ".pkl")
            shutil.move(self.jrdb_folder_path + 'OW/' + filename, self.jrdb_folder_path + 'backup/' + filename)
        return df

    def get_ou_df(self, filename):
        names = ['RACE_KEY', 'UMABAN_1', 'UMABAN_2', '馬単オッズ', 'target_date']
        if filename[-3:] == "pkl":
            df = pd.read_pickle(self.jrdb_folder_path + 'OU/' + filename)
        else:
            print(filename)
            with open(self.jrdb_folder_path + 'OU/' + filename, 'r', encoding="ms932") as fh:
                df = pd.DataFrame(index=[], columns=names)
                target_date = self.get_kaisai_date_for_oz(filename)
                print(dt.datetime.now())
                for line in fh:
                    new_line = self.replace_line(line)
                    race_key = new_line[0:8]
                    odds_text = new_line[10:1846]
                    base_odds_list = deque([odds_text[i: i + 6]
                                            for i in range(0, len(odds_text), 6)])
                    for uma1 in range(1, 19):
                        for uma2 in range(1, 19):
                            if uma1 != uma2:
                                base_odds = self.float_null(
                                    base_odds_list.popleft())
                                if base_odds != None:
                                    sr = pd.Series([
                                        race_key, str(uma1).zfill(2), str(uma2).zfill(2), base_odds, target_date
                                    ], index=names)
                                    df = df.append(sr, ignore_index=True)
            df.to_pickle(self.jrdb_folder_path + 'OU/' + filename + ".pkl")
            shutil.move(self.jrdb_folder_path + 'OU/' + filename, self.jrdb_folder_path + 'backup/' + filename)
        return df

    def str_null(self, str):
        """ 文字列を返し、空白の場合はNoneに変換する

        :param str str:
        :return:
        """
        cnt = len(str)
        empty_val = ''
        for i in range(cnt):
            empty_val += ' '
        if str == empty_val:
            return '0'
        else:
            return str.replace(" ", "").strip()

    def int_shokin(self, str):
        """ 文字列をintにし、空白の場合は0に変換する

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
            return int(math.floor(float(str)))

    def int_0(self, str):
        """ 文字列をintにし、空白の場合は0に変換する

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

    def int_null(self, str):
        """ 文字列をintにし、空白の場合はNoneに変換する

        :param str str:
        :return:
        """
        cnt = len(str)
        empty_val = ''
        for i in range(cnt):
            empty_val += ' '
        if str == empty_val:
            return np.nan
        else:
            return int(str)

    def float_null(self, str):
        """ 文字列をfloatにし、空白の場合はNoneに変換する

        :param str str:
        :return:
        """
        cnt = len(str)
        empty_val = ''
        for i in range(cnt):
            empty_val += ' '
        if str == empty_val:
            return np.nan
        else:
            return float(str)

    def get_kaisai_date(self, filename):
        """ ファイル名から開催年月日を取得する(ex.20181118)

        :param str filename:
        :return:
        """
        return '20' + filename[3:9]

    def get_kaisai_date_for_oz(self, filename):
        """ ファイル名から開催年月日を取得する(ex.20181118)

        :param str filename:
        :return:
        """
        return '20' + filename[2:8]

    def int_bataiju_zogen(self, str):
        """ 文字列の馬体重増減を数字に変換する

        :param str str:
        :return:
        """
        fugo = str[0:1]
        if fugo == "+":
            return int(str[1:3])
        elif fugo == "-":
            return int(str[1:3]) * (-1)
        else:
            return 0

    def int_haito(self, str):
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

    def convert_time(self, str):
        """ 文字列のタイム(ex.1578)を秒数(ex.1178)に変換する

        :param str str:
        :return:
        """
        min = str[0:1]
        if min != ' ':
            return int(min) * 600 + int(str[1:4])
        else:
            return self.int_0(str[1:4])

    def replace_line(self, line):
        count = 0
        new_line = ''
        for c in line:
            if unicodedata.east_asian_width(c) in 'FWA':
                new_line += c + ' '
                count += 2
            else:
                new_line += c
                count += 1
        return new_line

    def get_tansho_df(self, df):
        """ 単勝配当のデータフレームを作成する。同着対応のため横になっているデータを縦に結合する。

        :param dataframe df:
        :return: dataframe
        """
        tansho_df1 = df[["RACE_KEY", "単勝馬番１", "単勝払戻１"]]
        tansho_df2 = df[["RACE_KEY", "単勝馬番２", "単勝払戻２"]]
        tansho_df3 = df[["RACE_KEY", "単勝馬番３", "単勝払戻３"]]
        df_list = [tansho_df1, tansho_df2, tansho_df3]
        return_df = self.arrange_return1_df(df_list)
        return return_df

    def get_fukusho_df(self, df):
        """ 複勝配当のデータフレームを作成する。同着対応のため横になっているデータを縦に結合する。

        :param dataframe df:
        :return: dataframe
        """
        fukusho_df1 = df[["RACE_KEY", "複勝馬番１", "複勝払戻１"]]
        fukusho_df2 = df[["RACE_KEY", "複勝馬番２", "複勝払戻２"]]
        fukusho_df3 = df[["RACE_KEY", "複勝馬番３", "複勝払戻３"]]
        fukusho_df4 = df[["RACE_KEY", "複勝馬番４", "複勝払戻４"]]
        fukusho_df5 = df[["RACE_KEY", "複勝馬番５", "複勝払戻５"]]
        df_list = [fukusho_df1, fukusho_df2, fukusho_df3, fukusho_df4, fukusho_df5]
        return_df = self.arrange_return1_df(df_list)
        return return_df

    def arrange_return1_df(self, df_list):
        """ 内部処理用、配当データの列をRACE_KEY、馬番、払戻に統一する

        :param list df_list: dataframeのリスト
        :return: dataframe
        """
        for df in df_list:
            df.columns = ["RACE_KEY", "UMABAN", "払戻"]
        return_df = pd.concat(df_list)
        temp_return_df = return_df[return_df["払戻"] != 0]
        return temp_return_df

    def get_wide_df(self, df):
        """ ワイド配当のデータフレームを作成する。同着対応のため横になっているデータを縦に結合する。

        :param dataframe df:
        :return: dataframe
        """
        wide_df1 = df[["RACE_KEY", "ワイド馬番１１", "ワイド馬番１２", "ワイド払戻１"]]
        wide_df2 = df[["RACE_KEY", "ワイド馬番２１", "ワイド馬番２２", "ワイド払戻２"]]
        wide_df3 = df[["RACE_KEY", "ワイド馬番３１", "ワイド馬番３２", "ワイド払戻３"]]
        wide_df4 = df[["RACE_KEY", "ワイド馬番４１", "ワイド馬番４２", "ワイド払戻４"]]
        wide_df5 = df[["RACE_KEY", "ワイド馬番５１", "ワイド馬番５２", "ワイド払戻５"]]
        wide_df6 = df[["RACE_KEY", "ワイド馬番６１", "ワイド馬番６２", "ワイド払戻６"]]
        wide_df7 = df[["RACE_KEY", "ワイド馬番７１", "ワイド馬番７２", "ワイド払戻７"]]
        df_list = [wide_df1, wide_df2, wide_df3, wide_df4, wide_df5, wide_df6, wide_df7]
        return_df = self.arrange_return2_df(df_list)
        return return_df

    def get_umaren_df(self, df):
        """ 馬連配当のデータフレームを作成する。同着対応のため横になっているデータを縦に結合する。

        :param dataframe df:
        :return: dataframe
        """
        umaren_df1 = df[["RACE_KEY", "馬連馬番１１", "馬連馬番１２", "馬連払戻１"]]
        umaren_df2 = df[["RACE_KEY", "馬連馬番２１", "馬連馬番２２", "馬連払戻２"]]
        umaren_df3 = df[["RACE_KEY", "馬連馬番３１", "馬連馬番３２", "馬連払戻３"]]
        df_list = [umaren_df1, umaren_df2, umaren_df3]
        return_df = self.arrange_return2_df(df_list)
        return return_df

    def get_umatan_df(self, df):
        """ 馬単配当のデータフレームを作成する。同着対応のため横になっているデータを縦に結合する。

        :param dataframe df:
        :return: dataframe
        """
        umatan_df1 = df[["RACE_KEY", "馬単馬番１１", "馬単馬番１２", "馬単払戻１"]]
        umatan_df2 = df[["RACE_KEY", "馬単馬番２１", "馬単馬番２２", "馬単払戻２"]]
        umatan_df3 = df[["RACE_KEY", "馬単馬番３１", "馬単馬番３２", "馬単払戻３"]]
        umatan_df4 = df[["RACE_KEY", "馬単馬番４１", "馬単馬番４２", "馬単払戻４"]]
        umatan_df5 = df[["RACE_KEY", "馬単馬番５１", "馬単馬番５２", "馬単払戻５"]]
        umatan_df6 = df[["RACE_KEY", "馬単馬番６１", "馬単馬番６２", "馬単払戻６"]]
        df_list = [umatan_df1, umatan_df2, umatan_df3, umatan_df4, umatan_df5, umatan_df6]
        return_df = self.arrange_return2_df(df_list)
        return return_df

    def arrange_return2_df(self, df_list):
        """ 内部処理用、配当データの列をRACE_KEY、馬番、払戻に統一する

        :param list df_list: dataframeのリスト
        :return: dataframe
        """
        for df in df_list:
            df.columns = ["RACE_KEY", "馬番1", "馬番2", "払戻"]
        return_df = pd.concat(df_list)
        return_df.loc[:, "UMABAN"] = return_df.apply(lambda x: [x["馬番1"], x["馬番2"]], axis=1)
        temp_return_df = return_df[return_df["払戻"] != 0]
        return temp_return_df[["RACE_KEY", "UMABAN", "払戻"]]

    def get_sanrenpuku_df(self, df):
        """  三連複配当のデータフレームを作成する。同着対応のため横になっているデータを縦に結合する。

        :param dataframe df:
        :return: dataframe
        """
        sanrenpuku1 = df[["RACE_KEY", "３連複馬番１１", "３連複馬番１２", "３連複馬番１３", "３連複払戻１"]]
        sanrenpuku2 = df[["RACE_KEY", "３連複馬番２１", "３連複馬番２２", "３連複馬番２３", "３連複払戻２"]]
        sanrenpuku3 = df[["RACE_KEY", "３連複馬番３１", "３連複馬番３２", "３連複馬番３３", "３連複払戻３"]]
        df_list = [sanrenpuku1, sanrenpuku2, sanrenpuku3]
        return_df = self.arrange_return3_df(df_list)
        return return_df

    def get_sanrentan_df(self, df):
        """ 三連単配当のデータフレームを作成する。同着対応のため横になっているデータを縦に結合する。

        :param dataframe df:
        :return: dataframe
        """
        sanrentan1 = df[["RACE_KEY", "３連単馬番１１", "３連単馬番１２", "３連単馬番１３", "３連単払戻１"]]
        sanrentan2 = df[["RACE_KEY", "３連単馬番２１", "３連単馬番２２", "３連単馬番２３", "３連単払戻２"]]
        sanrentan3 = df[["RACE_KEY", "３連単馬番３１", "３連単馬番３２", "３連単馬番３３", "３連単払戻３"]]
        sanrentan4 = df[["RACE_KEY", "３連単馬番４１", "３連単馬番４２", "３連単馬番４３", "３連単払戻４"]]
        sanrentan5 = df[["RACE_KEY", "３連単馬番５１", "３連単馬番５２", "３連単馬番５３", "３連単払戻５"]]
        sanrentan6 = df[["RACE_KEY", "３連単馬番６１", "３連単馬番６２", "３連単馬番６３", "３連単払戻６"]]
        df_list = [sanrentan1, sanrentan2, sanrentan3, sanrentan4, sanrentan5, sanrentan6]
        return_df = self.arrange_return3_df(df_list)
        return return_df

    def arrange_return3_df(self, df_list):
        """ 内部処理用、配当データの列をRACE_KEY、馬番、払戻に統一する

        :param list df_list: dataframeのリスト
        :return: dataframe
        """
        for df in df_list:
            df.columns = ["RACE_KEY", "馬番1", "馬番2", "馬番3", "払戻"]
        return_df = pd.concat(df_list)
        return_df.loc[:, "UMABAN"] = return_df.apply(lambda x: [x["馬番1"], x["馬番2"], x["馬番3"]], axis=1)
        temp_return_df = return_df[return_df["払戻"] != 0]
        return temp_return_df[["RACE_KEY", "UMABAN", "払戻"]]

    def get_haraimodoshi_dict(self, haraimodoshi_df):
        """ 払戻用のデータを作成する。extオブジェクトから各払戻データを取得して辞書化して返す。

        :return: dict {"tansho_df": tansho_df, "fukusho_df": fukusho_df}
        """
        tansho_df = self.get_tansho_df(haraimodoshi_df)
        fukusho_df = self.get_fukusho_df(haraimodoshi_df)
        umaren_df = self.get_umaren_df(haraimodoshi_df)
        wide_df = self.get_wide_df(haraimodoshi_df)
        umatan_df = self.get_umatan_df(haraimodoshi_df)
        sanrenpuku_df = self.get_sanrenpuku_df(haraimodoshi_df)
        sanrentan_df = self.get_sanrentan_df(haraimodoshi_df)
        dict_haraimodoshi = {"tansho_df": tansho_df, "fukusho_df": fukusho_df, "umaren_df": umaren_df,
                             "wide_df": wide_df, "umatan_df": umatan_df, "sanrenpuku_df": sanrenpuku_df,
                             "sanrentan_df": sanrentan_df}
        return dict_haraimodoshi
