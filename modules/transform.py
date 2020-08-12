import os
import modules.util as mu

import pandas as pd
import numpy as np
import os
import math
import sys
from factor_analyzer import FactorAnalyzer


class Transform(object):
    """
    データ変換に関する共通処理を定義する
    辞書データの格納等の処理があるため、基本的にはインスタンス化して実行するが、不要な場合はクラスメソッドでの実行を可とする。
    辞書データの作成は、ない場合は作成、ある場合は読み込み、を基本方針とする(learning_modeによる判別はしない）
    """

    def __init__(self, start_date, end_date):
        self.start_date = start_date
        self.end_date = end_date

    ##################### race_df ###############################
    def encode_race_df(self, race_df):
        """  列をエンコードする処理（ラベルエンコーディング、onehotエンコーディング等）"""
        race_df.loc[:, '曜日'] = race_df['曜日'].apply(lambda x: self._convert_weekday(x))
        return race_df

    def normalize_race_df(self, race_df):
        return race_df

    def standardize_race_df(self, race_df):
        return race_df

    def create_feature_race_df(self, race_df):
        """ 特徴となる値を作成する。ナイター、季節、非根幹、距離グループ、頭数グループを作成して列として付与する。

        :param dataframe race_df:
        :return: dataframe
        """
        race_df.loc[:, '場名'] = race_df['RACE_KEY'].str[:2]
        race_df.loc[:, '条件'] = race_df['条件'].apply(lambda x: self._convert_joken(x))
        race_df.loc[:, '月'] = race_df['NENGAPPI'].str[4:6].astype(int)
        race_df.loc[:, "非根幹"] = race_df["距離"].apply(lambda x: 0 if x % 400 == 0 else 1)
        race_df.loc[:, "距離グループ"] = race_df["距離"] // 400
        return race_df

    def choose_race_df_columns(self, race_df):
        race_df = race_df[['RACE_KEY', 'KAISAI_KEY', 'NENGAPPI', '発走時間', '距離', '芝ダ障害コード', '右左',
       '内外', '種別', '条件', '記号', '重量','頭数', 'コース', '開催区分',
       'データ区分', 'WIN5フラグ', 'target_date', '曜日', '場名',
       '天候コード', '芝馬場状態コード', '芝馬場状態内', '芝馬場状態中', '芝馬場状態外', '芝馬場差', '直線馬場差最内',
       '直線馬場差内', '直線馬場差中', '直線馬場差外', '直線馬場差大外', 'ダ馬場状態コード', 'ダ馬場状態内', 'ダ馬場状態中',
       'ダ馬場状態外', 'ダ馬場差', '連続何日目', '芝種類', '草丈', '転圧', '凍結防止剤', '中間降水量', '月',
       '非根幹', '距離グループ', "course_cluster"]].copy()
        race_df = race_df.astype({'距離': int, '芝ダ障害コード': int, '右左': int, '内外': int, '種別': int, '記号': int, '重量': int,
                                  'コース': int, '天候コード': int, '芝馬場状態コード': int, '芝馬場状態内': int, '芝馬場状態中': int, #'曜日': int,
                                  '芝馬場状態外': int, '直線馬場差最内': int, '直線馬場差内': int, '直線馬場差中': int, '直線馬場差外': int, '直線馬場差大外': int,
                                  'ダ馬場状態コード': int, 'ダ馬場状態内': int, 'ダ馬場状態中': int, 'ダ馬場状態外': int, '芝種類': int, '転圧': int,
                                  '凍結防止剤': int,'距離グループ': int})
        return race_df

    ##################### raceuma_df ###############################
    def encode_raceuma_df(self, raceuma_df, dict_folder):
        """  列をエンコードする処理。騎手名、所属、転厩をラベルエンコーディングして値を置き換える。learning_modeがTrueの場合は辞書生成がされる。

        :param dataframe raceuma_df:
        :return: dataframe
        """
        raceuma_df.loc[:, 'ペース予想'] = raceuma_df['ペース予想'].apply(lambda x: self._convert_pace(x))
        raceuma_df.loc[:, '調教曜日'] = raceuma_df['調教曜日'].apply(lambda x: self._convert_weekday(x))
        raceuma_df.loc[:, '放牧先ランク'] = raceuma_df['放牧先ランク'].apply(lambda x: self._convert_rank(x))
        raceuma_df.loc[:, '調教量評価'] = raceuma_df['調教量評価'].apply(lambda x: self._convert_rank(x))
        raceuma_df.loc[:, '調教師所属'] = mu.label_encoding(raceuma_df['調教師所属'], '調教師所属', dict_folder).astype(str)
        raceuma_df.loc[:, '馬主名'] = mu.label_encoding(raceuma_df['馬主名'], '馬主名', dict_folder).astype(str)
        raceuma_df.loc[:, '放牧先'] = mu.label_encoding(raceuma_df['放牧先'], '放牧先', dict_folder).astype(str)
        raceuma_df.loc[:, '激走タイプ'] = mu.label_encoding(raceuma_df['激走タイプ'], '激走タイプ', dict_folder).astype(str)
        raceuma_df.loc[:, '調教コースコード'] = mu.label_encoding(raceuma_df['調教コースコード'], '調教コースコード', dict_folder).astype(str)
        raceuma_df.loc[:, 'LS評価'] = raceuma_df['LS評価'].apply(lambda x: mu.convert_ls_hyouka(x))
        #hash_taikei_column = ["走法", "体型０１", "体型０２", "体型０３", "体型０４", "体型０５", "体型０６", "体型０７", "体型０８", "体型０９", "体型１０", "体型１１", "体型１２", "体型１３", "体型１４", "体型１５", "体型１６", "体型１７", "体型１８", "体型総合１", "体型総合２", "体型総合３"]
        #hash_taikei_dict_name = "raceuma_before_taikei"
        #raceuma_df = mu.hash_eoncoding(raceuma_df, hash_taikei_column, 3, hash_taikei_dict_name, dict_folder)
        #hash_tokki_column = ["馬特記１", "馬特記２", "馬特記３"]
        #hash_tokki_dict_name = "raceuma_before_tokki"
        #raceuma_df = mu.hash_eoncoding(raceuma_df, hash_tokki_column, 2, hash_tokki_dict_name, dict_folder)


        return raceuma_df

    def normalize_raceuma_df(self, raceuma_df):
        """ 数値系データを平準化する処理。偏差値に変換して置き換える。対象列は負担重量、予想タイム指数、デフォルト得点、得点V1、得点V2、得点V3。偏差がない場合は５０に設定

        :param dataframe raceuma_df:
        :return: dataframe
        """
        norm_list = ['IDM', '騎手指数', '情報指数', '総合指数', '人気指数', '調教指数', '厩舎指数', 'テン指数', 'ペース指数', '上がり指数', '位置指数', '万券指数',
                                 'テンＦ指数', '中間Ｆ指数', '終いＦ指数', '追切指数', '仕上指数']
        temp_raceuma_df = raceuma_df[norm_list].astype(float)
        temp_raceuma_df.loc[:, "RACE_KEY"] = raceuma_df["RACE_KEY"]
        temp_raceuma_df.loc[:, "UMABAN"] = raceuma_df["UMABAN"]
        grouped_df = temp_raceuma_df[['RACE_KEY'] + norm_list].groupby('RACE_KEY').agg(['mean', 'std']).reset_index()
        grouped_df.columns = ['RACE_KEY', 'IDM_mean', 'IDM_std', '騎手指数_mean', '騎手指数_std', '情報指数_mean', '情報指数_std', '総合指数_mean', '総合指数_std',
                              '人気指数_mean', '人気指数_std', '調教指数_mean', '調教指数_std', '厩舎指数_mean', '厩舎指数_std', 'テン指数_mean', 'テン指数_std',
                              'ペース指数_mean', 'ペース指数_std', '上がり指数_mean', '上がり指数_std', '位置指数_mean', '位置指数_std', '万券指数_mean', '万券指数_std',
                              'テンＦ指数_mean', 'テンＦ指数_std', '中間Ｆ指数_mean', '中間Ｆ指数_std', '終いＦ指数_mean', '終いＦ指数_std', '追切指数_mean', '追切指数_std', '仕上指数_mean', '仕上指数_std']
        temp_raceuma_df = pd.merge(temp_raceuma_df, grouped_df, on='RACE_KEY')
        for norm in norm_list:
            temp_raceuma_df[f'{norm}偏差'] = temp_raceuma_df.apply(lambda x: (x[norm] - x[f'{norm}_mean']) / x[f'{norm}_std'] * 10 + 50 if x[f'{norm}_std'] != 0 else 50, axis=1)
            temp_raceuma_df = temp_raceuma_df.drop([norm, f'{norm}_mean', f'{norm}_std'], axis=1)
            raceuma_df = raceuma_df.drop(norm, axis=1)
            temp_raceuma_df = temp_raceuma_df.rename(columns={f'{norm}偏差': norm})
        raceuma_df = pd.merge(raceuma_df, temp_raceuma_df, on=["RACE_KEY", "UMABAN"])
        return raceuma_df.copy()

    def standardize_raceuma_df(self, raceuma_df):
        """ 数値データを整備する。無限大(inf）をnanに置き換える

        :param dataframe raceuma_df:
        :return: dataframe
        """
        return raceuma_df

    def create_feature_raceuma_df(self, raceuma_df):
        """  raceuma_dfの特徴量を作成する。

        :param dataframe raceuma_df:
        :return: dataframe
        """
        raceuma_df.loc[:, "馬番グループ"] = raceuma_df["UMABAN"].apply(lambda x: int(x) // 4)
        raceuma_df.loc[:, "基準人気グループ"] = raceuma_df["基準人気順位"].apply(lambda x : self._convert_ninki_group(x))
        result_type_list = ["ＪＲＡ成績", "交流成績", "他成績", "芝ダ障害別成績", "芝ダ障害別距離成績", "トラック距離成績", "ローテ成績", "回り成績", "騎手成績", "良成績", "稍成績",
                       "重成績", "Ｓペース成績", "Ｍペース成績", "Ｈペース成績", "季節成績", "枠成績", "騎手距離成績", "騎手トラック距離成績", "騎手調教師別成績", "騎手馬主別成績",
                       "騎手ブリンカ成績", "調教師馬主別成績"]
        for type in result_type_list:
            raceuma_df[f'{type}１着'] = raceuma_df[f'{type}１着'].apply(lambda x: x if x else 0)
            raceuma_df[f'{type}２着'] = raceuma_df[f'{type}２着'].apply(lambda x: x if x else 0)
            raceuma_df[f'{type}３着'] = raceuma_df[f'{type}３着'].apply(lambda x: x if x else 0)
            raceuma_df[f'{type}着外'] = raceuma_df[f'{type}着外'].apply(lambda x: x if x else 0)
            raceuma_df.loc[:, type] = raceuma_df.apply(lambda x: np.nan if (x[f'{type}１着'] + x[f'{type}２着'] + x[f'{type}３着'] + x[f'{type}着外'] == 0)
            else (x[f'{type}１着'] + x[f'{type}２着'] + x[f'{type}３着'])/ (x[f'{type}１着'] + x[f'{type}２着'] + x[f'{type}３着'] + x[f'{type}着外']), axis=1)
            raceuma_df.drop([f'{type}１着', f'{type}２着', f'{type}３着', f'{type}着外'], axis=1, inplace=True)
        return raceuma_df

    def choose_raceuma_df_columns(self, raceuma_df):
        raceuma_df = raceuma_df[['RACE_KEY', 'UMABAN', 'NENGAPPI', '血統登録番号', '脚質', '距離適性', '上昇度', 'ローテーション', '基準オッズ', '基準人気順位', '基準複勝オッズ',
                                 '基準複勝人気順位', '特定情報◎', '特定情報○', '特定情報▲', '特定情報△', '特定情報×', '総合情報◎', '総合情報○', '総合情報▲', '総合情報△',
                                 '総合情報×', '調教矢印コード', '厩舎評価コード', '騎手期待連対率', '激走指数', '蹄コード', '重適正コード', 'クラスコード', 'ブリンカー', '騎手名',
                                 '負担重量', '見習い区分', '調教師名', '調教師所属', 'ZENSO1_KYOSO_RESULT', 'ZENSO2_KYOSO_RESULT', 'ZENSO3_KYOSO_RESULT',
                                 'ZENSO4_KYOSO_RESULT', 'ZENSO5_KYOSO_RESULT', 'ZENSO1_RACE_KEY', 'ZENSO2_RACE_KEY', 'ZENSO3_RACE_KEY', 'ZENSO4_RACE_KEY',
                                 'ZENSO5_RACE_KEY', '枠番', '総合印', 'ＩＤＭ印', '情報印', '騎手印', '厩舎印', '調教印', '激走印', '芝適性コード', 'ダ適性コード', '騎手コード',
                                 '調教師コード', '獲得賞金', '収得賞金', '条件クラス', 'ペース予想', '道中順位', '道中差', '道中内外', '後３Ｆ順位', '後３Ｆ差', '後３Ｆ内外',
                                 'ゴール順位', 'ゴール差', 'ゴール内外', '展開記号', '距離適性２', '枠確定馬体重', '枠確定馬体重増減', '取消フラグ', '性別コード', '馬主名',
                                 '馬主会コード', '馬記号コード', '激走順位', 'LS指数順位', 'テン指数順位', 'ペース指数順位', '上がり指数順位', '位置指数順位', '騎手期待単勝率',
                                 '騎手期待３着内率', '輸送区分', '馬スタート指数', '馬出遅率', '万券印', '降級フラグ', '激走タイプ',
                                 '休養理由分類コード', '芝ダ障害フラグ', '距離フラグ', 'クラスフラグ', '転厩フラグ', '去勢フラグ', '乗替フラグ', '入厩何走目', '入厩年月日',
                                 '入厩何日前', '放牧先', '放牧先ランク', '厩舎ランク', 'target_date', 'CID調教素点', 'CID厩舎素点', 'CID素点', 'CID', 'LS指数', 'LS評価',
                                 'EM', '厩舎ＢＢ印', '厩舎ＢＢ◎単勝回収率', '厩舎ＢＢ◎連対率', '騎手ＢＢ印', '騎手ＢＢ◎単勝回収率', '騎手ＢＢ◎連対率', '調教曜日', '調教回数',
                                 '調教コースコード', '追切種類', '追い状態', '乗り役', '調教Ｆ', 'テンＦ', '中間Ｆ', '終いＦ', '併せ結果', '併せ追切種類', '併せ年齢', '併せクラス',
                                 '調教タイプ', '調教コース種別', '調教コース坂', '調教コースW', '調教コースダ', '調教コース芝', '調教コースプール', '調教コース障害', '調教コースポリ',
                                 '調教距離', '調教重点', '調教量評価', '仕上指数変化', '調教評価', '父馬産駒芝連対率', '父馬産駒ダ連対率',
                                 '父馬産駒連対平均距離', '母父馬産駒芝連対率', '母父馬産駒ダ連対率', '母父馬産駒連対平均距離', 'IDM', '騎手指数', '情報指数', '総合指数', '人気指数',
                                 # 'raceuma_before_taikei_0', 'raceuma_before_taikei_1','raceuma_before_taikei_2', 'raceuma_before_tokki_0', 'raceuma_before_tokki_1',
                                 '調教指数', '厩舎指数', 'テン指数', 'ペース指数', '上がり指数', '位置指数', '万券指数', 'テンＦ指数', '中間Ｆ指数', '終いＦ指数', '追切指数',
                                 '仕上指数', '馬番グループ', '基準人気グループ', 'ＪＲＡ成績', '交流成績', '他成績', '芝ダ障害別成績', '芝ダ障害別距離成績', 'トラック距離成績',
                                 'ローテ成績', '回り成績', '騎手成績', '良成績', '稍成績', '重成績', 'Ｓペース成績', 'Ｍペース成績', 'Ｈペース成績', '季節成績', '枠成績',
                                 '騎手距離成績', '騎手トラック距離成績', '騎手調教師別成績', '騎手馬主別成績', '騎手ブリンカ成績', '調教師馬主別成績']].copy()
        raceuma_df = raceuma_df.astype({'脚質': int, '距離適性': int, '上昇度': int, '調教矢印コード': int,'厩舎評価コード': int, '激走指数': int,'蹄コード': int, '重適正コード': int,
                                        'クラスコード': int, 'ブリンカー': int, '負担重量': int, '見習い区分': int, '枠番': int, '総合印': int, 'ＩＤＭ印': int,
                                        '情報印': int, '騎手印': int, '厩舎印': int, '調教印': int, '激走印': int, '芝適性コード': int, 'ダ適性コード': int, '条件クラス': int,
                                        '道中内外': int, '後３Ｆ内外': int, 'ゴール内外': int, '展開記号': int, '距離適性２': int, '枠確定馬体重': int, '枠確定馬体重増減': int, '取消フラグ': int,
                                        '性別コード': int, '馬主会コード': int, '馬記号コード': int, '輸送区分': int, '万券印': int, '降級フラグ': int, '休養理由分類コード': int,
                                        '芝ダ障害フラグ': int, '距離フラグ': int, 'クラスフラグ': int, '転厩フラグ': int, '去勢フラグ': int, '乗替フラグ': int, '入厩何走目': int, '入厩何日前': int,
                                        'CID': int, 'EM': int, '厩舎ＢＢ印': int, '騎手ＢＢ印': int, '調教回数': int, #'調教曜日': int, '放牧先ランク': int, '調教師所属': int, '激走タイプ': int, '調教コースコード': int,
                                        '追切種類': int, '追い状態': int, '乗り役': int, '併せ結果': int, '併せ追切種類': int, '併せ年齢': int,
                                        '調教タイプ': int, '調教コース種別': int, '調教距離': int, '調教重点': int, '仕上指数変化': int, '調教評価': int})
        return raceuma_df

    ##################### horse_df ###############################
    def encode_horse_df(self, horse_df, dict_folder):
        """  列をエンコードする処理（ラベルエンコーディング、onehotエンコーディング等）"""
        horse_df.loc[:, '父馬名'] = mu.label_encoding(horse_df['父馬名'], '父馬名', dict_folder).astype(str)
        horse_df.loc[:, '母父馬名'] = mu.label_encoding(horse_df['母父馬名'], '母父馬名', dict_folder).astype(str)
        horse_df.loc[:, '生産者名'] = mu.label_encoding(horse_df['生産者名'], '生産者名', dict_folder).astype(str)
        horse_df.loc[:, '産地名'] = mu.label_encoding(horse_df['産地名'], '産地名', dict_folder).astype(str)
        return horse_df

    def normalize_horse_df(self, horse_df):
        return horse_df

    def standardize_horse_df(self, horse_df):
        return horse_df

    def create_feature_horse_df(self, horse_df):
        return horse_df

    def choose_horse_df_column(self, horse_df):
        """ 馬データから必要な列に絞り込む。対象は血統登録番号、繁殖登録番号１、繁殖登録番号５、東西所属コード、生産者コード、馬主コード

        :param dataframe raceuma_df:
        :return: dataframe
        """
        horse_df = horse_df[['血統登録番号', '毛色コード', '父馬名', '母父馬名', '生産者名', '産地名', '父系統コード', '母父系統コード', 'target_date']].copy()
        horse_df = horse_df.astype({'毛色コード': int, '父系統コード': int, '母父系統コード': int})#'父馬名': int, '母父馬名': int, '生産者名': int, '産地名': int,
        return horse_df


    ##################### race_result_df ###############################
    ### 前処理
    def cluster_course_df(self, race_df, dict_path):
        """ コースを10(0-9)にクラスタリング。当てはまらないものは10にする

        """
        dict_folder = dict_path + 'dict/jra_common/'
        fa_dict_name = "course_cluster_df.pkl"
        cluster_df = pd.read_pickle(dict_folder + fa_dict_name)
        race_df.loc[:, "COURSE_KEY"] = race_df["RACE_KEY"].str[:2] + race_df["距離"].astype(str) + race_df["芝ダ障害コード"] + race_df["内外"]
        race_df.loc[:, "場コード"] = race_df["RACE_KEY"].str[:2]
        df = pd.merge(race_df, cluster_df, on="COURSE_KEY", how="left").rename(columns={"cluster": "course_cluster"})
        df.loc[:, "course_cluster"] = df["course_cluster"].fillna(10)
        return df


    def cluster_raceuma_result_df(self, raceuma_result_df, dict_path):
        """ 出走結果をクラスタリング。それぞれ以下の意味
        # 激走: 4:前目の位置につけて能力以上の激走
        # 好走：1:後方から上がり上位で能力通り好走 　7:前目の位置につけて能力通り好走
        # ふつう：0:なだれ込み能力通りの凡走    5:前目の位置につけて上りの足が上がり能力通りの凡走 6:後方から足を使うも能力通り凡走
        # 凡走（下位）：2:前目の位置から上りの足が上がって能力以下の凡走　
        # 大凡走　3:後方追走いいとこなしで能力以下の大凡走
        # 障害、出走取消等→8

        """
        dict_folder = dict_path + 'dict/jra_common/'
        fa_dict_name = "cluster_raceuma_result"
        cluster = mu.load_dict(fa_dict_name, dict_folder)
        fa_list = ["RACE_KEY", "UMABAN", "IDM", "RAP_TYPE", "着順", "確定単勝人気順位", "ＩＤＭ結果", "コーナー順位２",
                   "コーナー順位３", "コーナー順位４", "タイム", "距離", "芝ダ障害コード", "後３Ｆタイム", "テン指数結果順位",
                   "上がり指数結果順位", "頭数", "前３Ｆ先頭差", "後３Ｆ先頭差", "異常区分"]
        temp_df = raceuma_result_df.query("異常区分 not in ('1','2') and 芝ダ障害コード != '3' and 頭数 != 0")
        df = temp_df[fa_list].copy()
        df.loc[:, "追走力"] = df.apply(
            lambda x: x["コーナー順位２"] - x["コーナー順位４"] if x["コーナー順位２"] != 0 else x["コーナー順位３"] - x["コーナー順位４"], axis=1)
        df.loc[:, "追上力"] = df.apply(lambda x: x["コーナー順位４"] - x["着順"], axis=1)
        df.loc[:, "１ハロン平均"] = df.apply(lambda x: x["タイム"] / x["距離"] * 200, axis=1)
        df.loc[:, "後傾指数"] = df.apply(lambda x: x["１ハロン平均"] * 3 / x["後３Ｆタイム"] if x["後３Ｆタイム"] != 0 else 1, axis=1)
        df.loc[:, "馬番"] = df["UMABAN"].astype(int) / df["頭数"]
        df.loc[:, "IDM差"] = df["ＩＤＭ結果"] - df["IDM"]
        df.loc[:, "コーナー順位４"] = df["コーナー順位４"] / df["頭数"]
        df.loc[:, "CHAKU_RATE"] = df["着順"] / df["頭数"]
        df.loc[:, "確定単勝人気順位"] = df["確定単勝人気順位"] / df["頭数"]
        df.loc[:, "テン指数結果順位"] = df["テン指数結果順位"] / df["頭数"]
        df.loc[:, "上がり指数結果順位"] = df["上がり指数結果順位"] / df["頭数"]
        df.loc[:, "上り最速"] = df["上がり指数結果順位"].apply(lambda x: 1 if x == 1 else 0)
        df.loc[:, "逃げ"] = df["テン指数結果順位"].apply(lambda x: 1 if x == 1 else 0)
        df.loc[:, "勝ち"] = df["着順"].apply(lambda x: 1 if x == 1 else 0)
        df.loc[:, "連対"] = df["着順"].apply(lambda x: 1 if x in (1, 2) else 0)
        df.loc[:, "３着内"] = df["着順"].apply(lambda x: 1 if x in (1, 2, 3) else 0)
        df.loc[:, "掲示板前後"] = df["着順"].apply(lambda x: 1 if x in (4, 5, 6) else 0)
        df.loc[:, "着外"] = df["CHAKU_RATE"].apply(lambda x: 1 if x >= 0.4 else 0)
        df.loc[:, "凡走"] = df.apply(lambda x: 1 if x["CHAKU_RATE"] >= 0.6 and x["確定単勝人気順位"] <= 0.3 else 0, axis=1)
        df.loc[:, "激走"] = df.apply(lambda x: 1 if x["CHAKU_RATE"] <= 0.3 and x["確定単勝人気順位"] >= 0.7 else 0, axis=1)
        df.loc[:, "異常"] = df["異常区分"].apply(lambda x: 1 if x != '0' else 0)
        numerical_feats = df.dtypes[df.dtypes != "object"].index
        km_df = df[numerical_feats].drop(["タイム", "後３Ｆタイム", "頭数", 'ＩＤＭ結果', "IDM", "１ハロン平均", 'コーナー順位２', 'コーナー順位３', '距離'],axis=1)
        # print(km_df.columns)
        # Index(['着順', '確定単勝人気順位', 'コーナー順位４', 'テン指数結果順位', '上がり指数結果順位', '前３Ｆ先頭差',
        #        '後３Ｆ先頭差', '追走力', '追上力', '後傾指数', '馬番', 'IDM差', 'CHAKU_RATE', '上り最速',
        #        '逃げ', '勝ち', '連対', '３着内', '掲示板前後', '着外', '凡走', '激走', '異常'],
        pred = cluster.predict(km_df)
        temp_df.loc[:, "ru_cluster"] = pred
        other_df = raceuma_result_df.query("異常区分 in ('1','2') or 芝ダ障害コード == '3'")
        other_df.loc[:, "ru_cluster"] = 8
        return_df = pd.concat([temp_df, other_df])
        return return_df

    def factory_analyze_race_result_df(self, raceuma_result_df, dict_path):
        """ レースの因子を計算。それぞれ以下の意味
        # fa1:　数値大：底力指数
        # fa2:　数値大：末脚指数
        # fa3:　数値大：スタミナ指数
        # fa4: 両方向：レースタイプ
        # fa5:　数値大：高速スピード指数
        """
        dict_folder = dict_path + 'dict/jra_common/'
        fa_dict_name = "fa_raceuma_result_df"
        fa = mu.load_dict(fa_dict_name, dict_folder)
        fa_list = ['１着算入賞金', 'ラップ差４ハロン', 'ラップ差３ハロン', 'ラップ差２ハロン', 'ラップ差１ハロン',
               'TRACK_BIAS_ZENGO', 'TRACK_BIAS_UCHISOTO', 'ハロン数', '芝', '外', '重', '軽',
               '１ハロン平均_mean', 'ＩＤＭ結果_mean', 'テン指数結果_mean', '上がり指数結果_mean',
               'ペース指数結果_mean', '追走力_mean', '追上力_mean', '後傾指数_mean', '１ハロン平均_std',
               '上がり指数結果_std', 'ペース指数結果_std']
        df_data_org = raceuma_result_df[fa_list]
        sc_dict_name = "fa_sc_raceuma_result_df"
        scaler = mu.load_dict(sc_dict_name, dict_folder)
        df_data = pd.DataFrame(scaler.transform(df_data_org), columns=df_data_org.columns)
        fa_df = pd.DataFrame(fa.transform(df_data.fillna(0)), columns=["fa_1", "fa_2", "fa_3", "fa_4", "fa_5"])
        raceuma_result_df = pd.concat([raceuma_result_df, fa_df], axis=1)
        return raceuma_result_df

    ##################### raceuma_result_df ###############################
    def encode_raceuma_result_df(self, raceuma_df, dict_folder):
        raceuma_df.loc[:, 'RAP_TYPE'] = raceuma_df['RAP_TYPE'].apply(lambda x: mu.encode_rap_type(x))
        raceuma_df.loc[:, '条件'] = raceuma_df['条件'].apply(lambda x: self._convert_joken(x))
        raceuma_df.loc[:, 'レースペース'] = raceuma_df['レースペース'].apply(lambda x: self._convert_pace(x))
        raceuma_df.loc[:, '馬ペース'] = raceuma_df['馬ペース'].apply(lambda x: self._convert_pace(x))
        raceuma_df.loc[:, 'ペースアップ位置'] = raceuma_df['ペースアップ位置'].fillna(0)
        raceuma_df.loc[:, '芝種類'] = raceuma_df['芝種類'].fillna('0')
        raceuma_df.loc[:, '転圧'] = raceuma_df['転圧'].fillna('0')
        raceuma_df.loc[:, '凍結防止剤'] = raceuma_df['凍結防止剤'].fillna('0')
        hash_tokki_column = ["特記コード１", "特記コード２", "特記コード３", "特記コード４", "特記コード５", "特記コード６"]
        hash_tokki_dict_name = "raceuma_result_tokki"
        temp_raceuma_df = mu.hash_eoncoding(raceuma_df, hash_tokki_column, 3, hash_tokki_dict_name, dict_folder)
        hash_bagu_column = ["馬具コード１", "馬具コード２", "馬具コード３", "馬具コード４", "馬具コード５", "馬具コード６", "馬具コード７", "馬具コード８", "ハミ", "バンテージ", "蹄鉄"]
        hash_bagu_dict_name = "raceuma_result_bagu"
        temp_raceuma_df = mu.hash_eoncoding(temp_raceuma_df, hash_bagu_column, 3, hash_bagu_dict_name, dict_folder)
        hash_taikei_column = ["総合１", "総合２", "総合３", "左前１", "左前２", "左前３", "右前１", "右前２", "右前３", "左後１", "左後２", "左後３", "右後１", "右後２", "右後３", "蹄状態", "ソエ", "骨瘤"]
        hash_taikei_dict_name = "raceuma_result_taikei"
        temp_raceuma_df = mu.hash_eoncoding(temp_raceuma_df, hash_taikei_column, 3, hash_taikei_dict_name, dict_folder)
        return raceuma_df

    def normalize_raceuma_result_df(self, raceuma_df):
        """ 数値系データを平準化する処理。偏差値に変換して置き換える。対象列は負担重量、予想タイム指数、デフォルト得点、得点V1、得点V2、得点V3。偏差がない場合は５０に設定

        :param dataframe raceuma_df:
        :return: dataframe
        """
        norm_list = ['ＩＤＭ結果', '斤量', 'テン指数結果', '上がり指数結果', 'ペース指数結果', 'レースＰ指数結果', '馬体重']
        temp_raceuma_df = raceuma_df[norm_list].astype(float)
        temp_raceuma_df.loc[:, "RACE_KEY"] = raceuma_df["RACE_KEY"]
        temp_raceuma_df.loc[:, "UMABAN"] = raceuma_df["UMABAN"]
        grouped_df = temp_raceuma_df[['RACE_KEY'] + norm_list].groupby('RACE_KEY').agg(['mean', 'std']).reset_index()
        grouped_df.columns = ['RACE_KEY', 'ＩＤＭ結果_mean', 'ＩＤＭ結果_std', '斤量_mean', '斤量_std', 'テン指数結果_mean',
                              'テン指数結果_std', '上がり指数結果_mean', '上がり指数結果_std', 'ペース指数結果_mean', 'ペース指数結果_std', 'レースＰ指数結果_mean',
                              'レースＰ指数結果_std', '馬体重_mean', '馬体重_std']
        temp_raceuma_df = pd.merge(temp_raceuma_df, grouped_df, on='RACE_KEY')
        for norm in norm_list:
            temp_raceuma_df[f'{norm}偏差'] = temp_raceuma_df.apply(lambda x: (x[norm] - x[f'{norm}_mean']) / x[f'{norm}_std'] * 10 + 50 if x[f'{norm}_std'] != 0 else 50, axis=1)
            temp_raceuma_df = temp_raceuma_df.drop([norm, f'{norm}_mean', f'{norm}_std'], axis=1)
            raceuma_df = raceuma_df.drop(norm, axis=1)
            temp_raceuma_df = temp_raceuma_df.rename(columns={f'{norm}偏差': norm})
        raceuma_df = pd.merge(raceuma_df, temp_raceuma_df, on=["RACE_KEY", "UMABAN"])
        return raceuma_df.copy()


    def standardize_raceuma_result_df(self, raceuma_df):
        return raceuma_df

    def create_feature_raceuma_result_df(self, raceuma_df):
        temp_raceuma_df = raceuma_df.query("頭数 != 0")
        temp_raceuma_df.loc[:, "非根幹"] = temp_raceuma_df["距離"].apply(lambda x: 0 if x % 400 == 0 else 1)
        temp_raceuma_df.loc[:, "距離グループ"] = temp_raceuma_df["距離"] // 400
        temp_raceuma_df.loc[:, "先行率"] = (temp_raceuma_df["コーナー順位４"] / temp_raceuma_df["頭数"])
        temp_raceuma_df.loc[:, "人気率"] = (temp_raceuma_df["確定単勝人気順位"] / temp_raceuma_df["頭数"])
        temp_raceuma_df.loc[:, "着順率"] = (temp_raceuma_df["着順"] / temp_raceuma_df["頭数"])
        temp_raceuma_df.loc[:, "追込率"] = (temp_raceuma_df["コーナー順位４"] - temp_raceuma_df["着順"]) / temp_raceuma_df["頭数"]
        temp_raceuma_df.loc[:, "平均タイム"] = temp_raceuma_df["タイム"] / temp_raceuma_df["距離"] * 200
        temp_raceuma_df.loc[:, "コーナー順位１率"] = (temp_raceuma_df["コーナー順位１"] / temp_raceuma_df["頭数"])
        temp_raceuma_df.loc[:, "コーナー順位２率"] = (temp_raceuma_df["コーナー順位２"] / temp_raceuma_df["頭数"])
        temp_raceuma_df.loc[:, "コーナー順位３率"] = (temp_raceuma_df["コーナー順位３"] / temp_raceuma_df["頭数"])
        temp_raceuma_df.loc[:, "コーナー順位４率"] = (temp_raceuma_df["コーナー順位４"] / temp_raceuma_df["頭数"])
        temp_raceuma_df.loc[:, "平坦コース"] = temp_raceuma_df["場コード"].apply(lambda x: 1 if x in ('01', '02', '03', '04', '08', '10') else 0)
        temp_raceuma_df.loc[:, "急坂コース"] = temp_raceuma_df["場コード"].apply(lambda x: 1 if x in ('06', '07', '09') else 0)
        temp_raceuma_df.loc[:, "好走"] = temp_raceuma_df["ru_cluster"].apply(lambda x: 1 if x in (1, 4, 7) else (-1 if x in (2, 3) else 0))
        temp_raceuma_df.loc[:, "内枠"] = temp_raceuma_df.apply(lambda x: 1 if int(x["UMABAN"]) / x["頭数"] <= 0.5 else 0, axis=1)
        temp_raceuma_df.loc[:, "外枠"] = temp_raceuma_df.apply(lambda x: 1 if int(x["UMABAN"]) / x["頭数"] > 0.5 else 0, axis=1)
        temp_raceuma_df.loc[:, "中山"] = temp_raceuma_df["場コード"].apply(lambda x: 1 if x == '06' else 0)
        temp_raceuma_df.loc[:, "東京"] = temp_raceuma_df["場コード"].apply(lambda x: 1 if x == '05' else 0)
        temp_raceuma_df.loc[:, "上がりかかる"] = temp_raceuma_df["後３Ｆタイム"].apply(lambda x: 1 if x >= 355 else 0)
        temp_raceuma_df.loc[:, "上がり速い"] = temp_raceuma_df["後３Ｆタイム"].apply(lambda x: 1 if x <= 335 else 0)
        temp_raceuma_df.loc[:, "ダート道悪"] = temp_raceuma_df.apply(lambda x: 1 if x["芝ダ障害コード"] == '2' and int(x["馬場状態"]) >= 30 else 0, axis=1)
        temp_raceuma_df.loc[:, "前崩れレース"] = temp_raceuma_df["TRACK_BIAS_ZENGO"].apply(lambda x: 1 if x >= 2 else 0)
        temp_raceuma_df.loc[:, "先行惜敗"] = temp_raceuma_df.apply(lambda x: 1 if x["着順"] in (2,3,4,5) and x["コーナー順位４"] in (1,2,3,4) else 0, axis=1)
        temp_raceuma_df.loc[:, "前残りレース"] = temp_raceuma_df["TRACK_BIAS_ZENGO"].apply(lambda x: 1 if x <= -2 else 0)
        temp_raceuma_df.loc[:, "差し損ね"] = temp_raceuma_df["ru_cluster"].apply(lambda x: 1 if x in (1, 6) else 0)
        temp_raceuma_df.loc[:, "上がり幅小さいレース"] = temp_raceuma_df["上がり指数結果_std"].apply(lambda x: 1 if x <= 2 else 0)
        temp_raceuma_df.loc[:, "内枠砂被り"] = temp_raceuma_df.apply(lambda x: 1 if x["芝ダ障害コード"] == '2' and x["内枠"] == 1 and x["着順"] > 5 else 0, axis =1)
        return temp_raceuma_df.copy()

    def choose_raceuma_result_df_columns(self, raceuma_df):
        raceuma_df = raceuma_df[['RACE_KEY', 'UMABAN', 'target_date', 'KYOSO_RESULT_KEY', '血統登録番号', 'NENGAPPI', '距離', '芝ダ障害コード', '右左', '内外', '馬場状態',
                                 '種別', '条件', '記号', '重量', '頭数', '着順', 'タイム', '確定単勝オッズ', '確定単勝人気順位', '素点', '馬場差', '不利',
                                 '1(2)着タイム差', '前３Ｆタイム', '後３Ｆタイム', '確定複勝オッズ下', 'コーナー順位１', 'コーナー順位２', 'コーナー順位３', 'コーナー順位４', '前３Ｆ先頭差',
                                 '後３Ｆ先頭差', '騎手コード', '調教師コード', '馬体重増減', 'レース脚質', '４角コース取り', 'テン指数結果順位', '上がり指数結果順位', 'ペース指数結果順位',
                                 'IDM', '特記コード１', '特記コード２', '特記コード３', '特記コード４', '特記コード５', '特記コード６', 'ペースアップ位置', 'ラップ差４ハロン', 'ラップ差３ハロン',
                                 'ラップ差２ハロン', 'ラップ差１ハロン', 'RAP_TYPE', '芝種類', '草丈', '転圧','コーナー順位１率', 'コーナー順位２率', 'コーナー順位３率', 'コーナー順位４率',
                                 '凍結防止剤', 'ハロン数', '芝', '外', '重', '軽', 'TRACK_BIAS_ZENGO', 'TRACK_BIAS_UCHISOTO', '１ハロン平均_mean',
                                 'ＩＤＭ結果_mean', 'テン指数結果_mean', '上がり指数結果_mean', 'ペース指数結果_mean', '前３Ｆタイム_mean', '後３Ｆタイム_mean', 'コーナー順位１_mean',
                                 'コーナー順位２_mean', 'コーナー順位３_mean', 'コーナー順位４_mean', '前３Ｆ先頭差_mean', '後３Ｆ先頭差_mean', '追走力_mean', '追上力_mean', '後傾指数_mean',
                                 '１ハロン平均_std', '上がり指数結果_std', 'ペース指数結果_std', '場コード', 'course_cluster', 'ru_cluster', 'fa_1', 'fa_2', 'fa_3',
                                 'fa_4', 'fa_5', 'ＩＤＭ結果', '斤量', 'テン指数結果', '上がり指数結果', 'ペース指数結果', 'レースＰ指数結果', '馬体重', '非根幹', '距離グループ', '先行率',
                                 '人気率', '着順率', '追込率', '平均タイム', '平坦コース', '急坂コース', '好走', '内枠', '外枠', '中山', '東京', '上がりかかる', '上がり速い', 'ダート道悪',
                                 '前崩れレース', '先行惜敗', '前残りレース', '差し損ね', '上がり幅小さいレース', '内枠砂被り']].copy()
        raceuma_df = raceuma_df.astype({'芝ダ障害コード': int, '右左': int, '内外': int, '馬場状態': int, '種別': int, '記号': int, '重量': int, 'レース脚質': int,
                                        '４角コース取り': int, '特記コード１': int, '特記コード２': int, '特記コード３': int, '特記コード４': int,
                                        '特記コード５': int, '特記コード６': int, 'ペースアップ位置': int, 'RAP_TYPE': int, '芝種類': int, '転圧': int, '凍結防止剤': int, '場コード': int})
        return raceuma_df


    def _convert_pace(self, pace):
        if pace == "S":
            return 1
        elif pace == "M":
            return 2
        elif pace == "H":
            return 3
        else:
            return 0

    def _convert_joken(self, joken):
        if joken == 'A1':
            return 1
        elif joken == 'A2':
            return 2
        elif joken == 'A3':
            return 3
        elif joken == 'OP':
            return 99
        elif joken is None:
            return 0
        else:
            return int(joken)

    def _convert_ninki_group(self, ninki):
        if ninki == 1:
            return 1
        elif ninki in (2, 3):
            return 2
        elif ninki in (4, 5, 6):
            return 3
        else:
            return 4


    def _convert_weekday(self, yobi):
        if yobi == "日":
            return '0'
        elif yobi == "月":
            return '1'
        elif yobi == "火":
            return '2'
        elif yobi == "水":
            return '3'
        elif yobi == "木":
            return '4'
        elif yobi == "金":
            return '5'
        elif yobi == "土":
            return '6'
        else:
            return yobi

    def _convert_rank(self, rank):
        if rank == "A":
            return 1
        elif rank == "B":
            return 2
        elif rank == "C":
            return 3
        elif rank == "D":
            return 4
        elif rank == "E":
            return 5
        elif rank == "F":
            return 6
        elif rank == "G":
            return 7
        elif rank == "H":
            return 8
        elif rank == "I":
            return 9
        else:
            return rank


    def choose_upper_n_count(self, df, column_name, n, dict_folder):
        """ 指定したカラム名の上位N出現以外をその他にまとめる

        :param df:
        :param column_name:
        :param n:
        :return: df
        """
        dict_name = "choose_upper_" + str(n) + "_" + column_name
        file_name = dict_folder + dict_name + ".pkl"
        if os.path.exists(file_name):
            temp_df = mu.load_dict(dict_name, dict_folder)
        else:
            temp_df = df[column_name].value_counts().iloc[:n].index
            mu.save_dict(temp_df, dict_name, dict_folder)
        df.loc[:, column_name] = df[column_name].apply(lambda x: x if x in temp_df else 'その他')
        return df