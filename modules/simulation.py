from modules.extract import Extract

import sys
import pandas as pd
import numpy as np
import re
import itertools
import modules.util as mu

class Simulation(object):
    """
    馬券シミュレーションに関する処理をまとめたクラス
    """

    def __init__(self, start_date, end_date, mock_flag, raceuma_df):
        self.start_date = start_date
        self.end_date = end_date
        self.mock_flag = mock_flag
        self.ext = self._get_extract_object(start_date, end_date, mock_flag)
        self.raceuma_df = raceuma_df.rename(columns={"RACE_KEY": "RACE_KEY", "UMABAN": "UMABAN"})
        self.haraimodoshi_df = self.ext.get_haraimodoshi_table_base()

    def _get_extract_object(self, start_date, end_date, mock_flag):
        """ 利用するExtクラスを指定する """
        ext = Extract(start_date, end_date, mock_flag)
        return ext

    def simulation_tansho(self, cond1, odds_cond):
        kaime_df = self.create_tansho_base_df(cond1)
        bet_df = kaime_df.query(odds_cond) if odds_cond != "" else kaime_df
        result_df = self.ext.get_tansho_df(self.haraimodoshi_df)
        check_df = pd.merge(bet_df, result_df, on=["RACE_KEY", "UMABAN"], how="left").fillna(0)
        cond_text = cond1
        sr = self._calc_summary(check_df, cond_text)
        return sr

    def create_tansho_base_df(self, cond1):
        df1 = self.raceuma_df.query(cond1)[["RACE_KEY", "UMABAN"]]
        odds_df = self.ext.get_odds_df("単勝")
        kaime_df = pd.merge(df1, odds_df, on=["RACE_KEY", "UMABAN"])
        return kaime_df

    def simulation_fukusho(self, cond1, odds_cond):
        kaime_df = self.create_fukusho_base_df(cond1)
        bet_df = kaime_df.query(odds_cond) if odds_cond != "" else kaime_df
        result_df = self.ext.get_fukusho_df(self.haraimodoshi_df)
        check_df = pd.merge(bet_df, result_df, on=["RACE_KEY", "UMABAN"], how="left").fillna(0)
        cond_text = cond1
        sr = self._calc_summary(check_df, cond_text)
        return sr

    def create_fukusho_base_df(self, cond1):
        df1 = self.raceuma_df.query(cond1)[["RACE_KEY", "UMABAN"]]
        odds_df = self.ext.get_odds_df("複勝")
        kaime_df = pd.merge(df1, odds_df, on=["RACE_KEY", "UMABAN"])
        return kaime_df

    def simulation_umaren(self, cond1, cond2, odds_cond):
        kaime_df = self.create_umaren_base_df(cond1, cond2)
        bet_df = kaime_df.query(odds_cond) if odds_cond != "" else kaime_df
        result_df = self.ext.get_umaren_df(self.haraimodoshi_df)
        check_df = self._check_result_kaime(bet_df, result_df)
        cond_text = "馬1." + cond1 + " AND 馬2." + cond2
        sr = self._calc_summary(check_df, cond_text)
        return sr

    def create_umaren_base_df(self, cond1, cond2):
        df1 = self.raceuma_df.query(cond1)[["RACE_KEY", "UMABAN"]]
        df2 = self.raceuma_df.query(cond2)[["RACE_KEY", "UMABAN"]]
        odds_df = self.ext.get_odds_df("馬連")
        kaime_df = self._get_umaren_kaime(df1, df2, odds_df)
        return kaime_df

    def simulation_wide(self, cond1, cond2, odds_cond):
        kaime_df = self.create_wide_base_df(cond1, cond2)
        bet_df = kaime_df.query(odds_cond) if odds_cond != "" else kaime_df
        result_df = self.ext.get_wide_df(self.haraimodoshi_df)
        check_df = self._check_result_kaime(bet_df, result_df)
        cond_text = "馬1." + cond1 + " AND 馬2." + cond2
        sr = self._calc_summary(check_df, cond_text)
        return sr

    def create_wide_base_df(self, cond1, cond2):
        df1 = self.raceuma_df.query(cond1)[["RACE_KEY", "UMABAN"]]
        df2 = self.raceuma_df.query(cond2)[["RACE_KEY", "UMABAN"]]
        odds_df = self.ext.get_odds_df("ワイド")
        kaime_df = self._get_2tou_kaime(df1, df2, odds_df)
        return kaime_df

    def simulation_umatan(self, cond1, cond2, odds_cond):
        kaime_df = self.create_umatan_base_df(cond1, cond2)
        bet_df = kaime_df.query(odds_cond) if odds_cond != "" else kaime_df
        result_df = self.ext.get_umatan_df(self.haraimodoshi_df)
        check_df = self._check_result_kaime(bet_df, result_df)
        cond_text = "馬1." + cond1 + " AND 馬2." + cond2
        sr = self._calc_summary(check_df, cond_text)
        return sr

    def create_umatan_base_df(self, cond1, cond2):
        df1 = self.raceuma_df.query(cond1)[["RACE_KEY", "UMABAN"]]
        df2 = self.raceuma_df.query(cond2)[["RACE_KEY", "UMABAN"]]
        odds_df = self.ext.get_odds_df("馬単")
        kaime_df = self._get_2tou_kaime(df1, df2, odds_df, ren=False)
        return kaime_df


    def simulation_sanrenpuku(self, cond1, cond2, cond3, odds_cond):
        kaime_df = self.create_sanrenpuku_base_df(cond1, cond2, cond3)
        bet_df = kaime_df.query(odds_cond) if odds_cond != "" else kaime_df
        result_df = self.ext.get_sanrenpuku_df(self.haraimodoshi_df)
        check_df = self._check_result_kaime(bet_df, result_df)
        cond_text = "馬1." + cond1 + " AND 馬2." + cond2 + " AND 馬3." + cond3
        sr = self._calc_summary(check_df, cond_text)
        return sr

    def create_sanrenpuku_base_df(self, cond1, cond2, cond3):
        df1 = self.raceuma_df.query(cond1)[["RACE_KEY", "UMABAN"]]
        df2 = self.raceuma_df.query(cond2)[["RACE_KEY", "UMABAN"]]
        df3 = self.raceuma_df.query(cond3)[["RACE_KEY", "UMABAN"]]
        odds_df = self.ext.get_odds_df("三連複")
        kaime_df = self._get_3tou_kaime(df1, df2, df3, odds_df)
        return kaime_df


    def _get_umaren_kaime(self, df1, df2, odds_df, ren=True):
        """ df1とdf2の組み合わせの馬連の買い目リストを作成, dfはRACE_KEY,UMABANのセット """
        temp_df = pd.merge(df1, df2, on="RACE_KEY")
        temp_df.loc[:, "連番１"] = temp_df["UMABAN_x"] + temp_df["UMABAN_y"]
        temp_df.loc[:, "連番２"] = temp_df["UMABAN_y"] + temp_df["UMABAN_x"]
        temp_df = temp_df.query("UMABAN_x != UMABAN_y")
        if ren:
            temp_df = temp_df.drop_duplicates(subset=["RACE_KEY", "連番１", "連番２"])
        odds_df = odds_df.drop("target_date", axis=1).set_index(["RACE_KEY", "UMABAN"])
        odds_df.columns = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12", "13", "14", "15", "16", "17", "18"]
        odds_df = odds_df.stack().reset_index()
        odds_df.columns = ["RACE_KEY", "UMABAN_x", "UMABAN_y", "オッズ"]
        base_df = pd.merge(temp_df, odds_df, on=["RACE_KEY", "UMABAN_x", "UMABAN_y"])
        if ren:
            base_df.loc[:, "UMABAN"] = base_df.apply(lambda x: [x["UMABAN_x"], x["UMABAN_y"]] if x["UMABAN_x"] < x["UMABAN_y"] else [x["UMABAN_y"], x["UMABAN_x"]], axis=1)
        else:
            base_df.loc[:, "UMABAN"] = base_df.apply(lambda x: [x["UMABAN_x"], x["UMABAN_y"]], axis=1)
        kaime_df = base_df[["RACE_KEY", "UMABAN", "オッズ"]].copy()
        return kaime_df


    def _get_2tou_kaime(self, df1, df2, odds_df, ren=True):
        """ df1とdf2の組み合わせの馬連の買い目リストを作成, dfはRACE_KEY,UMABANのセット """
        temp_df = pd.merge(df1, df2, on="RACE_KEY")
        temp_df.loc[:, "連番１"] = temp_df["UMABAN_x"] + temp_df["UMABAN_y"]
        temp_df.loc[:, "連番２"] = temp_df["UMABAN_y"] + temp_df["UMABAN_x"]
        temp_df = temp_df.query("UMABAN_x != UMABAN_y")
        if ren:
            temp_df = temp_df.drop_duplicates(subset=["RACE_KEY", "連番１", "連番２"])
            temp_df.loc[:, "UMABAN_1"] = temp_df.apply(lambda x: x["UMABAN_x"] if x["UMABAN_x"] <= x["UMABAN_y"] else x["UMABAN_y"], axis=1)
            temp_df.loc[:, "UMABAN_2"] = temp_df.apply(lambda x: x["UMABAN_y"] if x["UMABAN_x"] <= x["UMABAN_y"] else x["UMABAN_x"], axis=1)
        else:
            temp_df.loc[:, "UMABAN_1"] = temp_df["UMABAN_x"]
            temp_df.loc[:, "UMABAN_2"] = temp_df["UMABAN_y"]
        odds_df.columns = ["RACE_KEY", "UMABAN_1", "UMABAN_2", "オッズ", "target_date"]
        base_df = pd.merge(temp_df, odds_df, on=["RACE_KEY", "UMABAN_1", "UMABAN_2"])
        base_df.loc[:, "UMABAN"] = base_df.apply(lambda x: [x["UMABAN_1"], x["UMABAN_2"]], axis=1)
        kaime_df = base_df[["RACE_KEY", "UMABAN", "オッズ"]].copy()
        return kaime_df

    def _get_3tou_kaime(self, df1, df2, df3, odds_df, ren=True):
        """ df1とdf2.df3の組み合わせの馬連の買い目リストを作成, dfはRACE_KEY,UMABANのセット """
        #### 注意、馬１－馬２－馬３の組み合わせで馬１にある数字は馬２から除外されるようなので得点は小数点で計算する方がよさそう
        temp_df = pd.merge(df1, df2, on="RACE_KEY")
        temp_df = pd.merge(temp_df, df3, on="RACE_KEY")
        temp_df.loc[:, "連番"] = temp_df.apply(lambda x: sorted(list([x["UMABAN_x"], x["UMABAN_y"], x["UMABAN"]])), axis=1)
        temp_df = temp_df.query("UMABAN_x != UMABAN_y and UMABAN_x != UMABAN and UMABAN_y != UMABAN")
        if ren:
            temp_df.loc[:, "重複チェック"] = temp_df["連番"].apply(lambda x: "".join(x))
            temp_df = temp_df.drop_duplicates(subset=["RACE_KEY", "重複チェック"])
            temp_df.loc[:, "UMABAN_1"] = temp_df["連番"].apply(lambda x: x[0])
            temp_df.loc[:, "UMABAN_2"] = temp_df["連番"].apply(lambda x: x[1])
            temp_df.loc[:, "UMABAN_3"] = temp_df["連番"].apply(lambda x: x[2])
        else:
            temp_df.loc[:, "UMABAN_1"] = temp_df["UMABAN_x"]
            temp_df.loc[:, "UMABAN_2"] = temp_df["UMABAN_y"]
            temp_df.loc[:, "UMABAN_3"] = temp_df["UMABAN"]
        odds_df.columns = ["RACE_KEY", "UMABAN_1", "UMABAN_2", "UMABAN_3", "オッズ", "target_date"]
        base_df = pd.merge(temp_df, odds_df, on=["RACE_KEY", "UMABAN_1", "UMABAN_2", "UMABAN_3"])
        base_df.loc[:, "UMABAN"] = base_df.apply(lambda x: [x["UMABAN_1"], x["UMABAN_2"], x["UMABAN_3"]], axis=1)
        kaime_df = base_df[["RACE_KEY", "UMABAN", "オッズ"]].copy()
        return kaime_df

    def _check_result_kaime(self, kaime_df, result_df):
        """ 買い目DFと的中結果を返す """
        kaime_df["UMABAN"] = kaime_df["UMABAN"].apply(lambda x: ', '.join(map(str, x)))
        result_df["UMABAN"] = result_df["UMABAN"].apply(lambda x: ', '.join(map(str, x)))
        merge_df = pd.merge(kaime_df, result_df, on=["RACE_KEY", "UMABAN"], how="left").fillna(0)
        return merge_df

    def _calc_summary(self, df, cond_text):
        all_count = len(df)
        race_count = len(df["RACE_KEY"].drop_duplicates())
        hit_df = df[df["払戻"] != 0]
        hit_count = len(hit_df)
        avg_return = round(hit_df["払戻"].mean(), 0)
        std_return = round(hit_df["払戻"].std(), 0)
        max_return = hit_df["払戻"].max()
        sum_return = hit_df["払戻"].sum()
        avg = round(df["払戻"].mean() , 1)
        hit_rate = round(hit_count / all_count * 100 , 1) if all_count !=0 else 0
        race_hit_rate = round(hit_count / race_count * 100 , 1) if race_count !=0 else 0
        sr = pd.Series(data=[cond_text, all_count, hit_count, race_count, avg, hit_rate, race_hit_rate, avg_return, std_return, max_return, all_count * 100 , sum_return]
                       , index=["条件", "件数", "的中数", "レース数", "回収率", "的中率", "R的中率", "払戻平均", "払戻偏差", "最大払戻", "購入総額", "払戻総額"])
        return sr.fillna(0)
