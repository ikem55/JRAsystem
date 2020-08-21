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
    win_prob_list = [0, 0.4, 0.5, 0.6]
    jiku_prob_list = [0, 0.3, 0.4, 0.5]
    ana_prob_list = [0, 0.2, 0.3]
    #win_prob_list = [0, 0.5]
    #jiku_prob_list = [0, 0.4]
    #ana_prob_list = [0, 0.2]

    def __init__(self, start_date, end_date, mock_flag, raceuma_df):
        # raceuma_dfはscoreとrankまで計算済みのもの。各prob,std値を持つ
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

    def calc_sim_df(self, type, cond, odds_cond):
        base_df, low_odds_list, high_odds_list = self._subproc_get_odds_base_df(type, cond)
        sim_df = base_df.query(odds_cond).copy()
        base_sr = self._calc_summary(sim_df, cond)
        return base_sr


    def calc_monthly_sim_df(self, type, cond, odds_cond):
        base_df, low_odds_list, high_odds_list = self._subproc_get_odds_base_df(type, cond)
        sim_df = base_df.query(odds_cond).copy()
        summary_df = pd.DataFrame()
        ym_list = sim_df["年月"].drop_duplicates().tolist()
        for ym in ym_list:
            temp_ym_df = sim_df.query(f"年月 == '{ym}'")
            sim_sr = self._calc_summary(temp_ym_df, cond)
            sim_sr["年月"] = ym
            summary_df = summary_df.append(sim_sr, ignore_index=True)
        return summary_df

    def _subproc_get_odds_base_df(self, type, cond):
        result_df, raceuma_df = self._subproc_get_result_data(type)
        odds_df = self.ext.get_odds_df(type)
        if type == "馬連":
            odds_df = odds_df.drop("target_date", axis=1).set_index(["RACE_KEY", "UMABAN"])
            odds_df.columns = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12", "13", "14", "15",
                               "16", "17", "18"]
            odds_df = odds_df.stack().reset_index()
            odds_df.columns = ["RACE_KEY", "UMABAN_1", "UMABAN_2", "オッズ"]

        if type == "単勝" or type == "複勝":
            temp_raceuma1_df = raceuma_df.query(cond).copy()
            base_df = pd.merge(temp_raceuma1_df, odds_df, on=["RACE_KEY", "UMABAN", "target_date"])
            if type == "単勝":
                base_df = base_df.rename(columns={"単勝オッズ": "オッズ"})
            else:
                base_df = base_df.rename(columns={"複勝オッズ": "オッズ"})
            low_odds_list = [5, 8, 10]
            high_odds_list = [10, 20, 30, 50]
        elif type == "馬連" or type == "ワイド":
            temp_raceuma1_df = raceuma_df.query(cond[0]).copy()
            temp_raceuma2_df = raceuma_df.query(cond[1]).copy()
            temp_raceuma_df = self._subproc_create_umaren_df(temp_raceuma1_df, temp_raceuma2_df, result_df)
            temp_raceuma_df.loc[:, "UMABAN_1"] = temp_raceuma_df["UMABAN"].str[0:2]
            temp_raceuma_df.loc[:, "UMABAN_2"] = temp_raceuma_df["UMABAN"].str[4:6]
            base_df = pd.merge(temp_raceuma_df, odds_df, on=["RACE_KEY", "UMABAN_1", "UMABAN_2"])
            if type == "ワイド":
                base_df = base_df.rename(columns={"ワイドオッズ": "オッズ"})
            low_odds_list = [10, 15, 20]
            high_odds_list = [30, 50, 80, 100]
        elif type == "馬単":
            temp_raceuma1_df = raceuma_df.query(cond[0]).copy()
            temp_raceuma2_df = raceuma_df.query(cond[1]).copy()
            temp_raceuma_df = self._subproc_create_umaren_df(temp_raceuma1_df, temp_raceuma2_df, result_df, ren=False)
            temp_raceuma_df.loc[:, "UMABAN_1"] = temp_raceuma_df["UMABAN"].str[0:2]
            temp_raceuma_df.loc[:, "UMABAN_2"] = temp_raceuma_df["UMABAN"].str[4:6]
            base_df = pd.merge(temp_raceuma_df, odds_df, on=["RACE_KEY", "UMABAN_1", "UMABAN_2"])
            base_df = base_df.rename(columns={"馬単オッズ": "オッズ"})
            low_odds_list = [10, 20, 30]
            high_odds_list = [30, 50, 80, 100, 150]
        elif type == "三連複":
            temp_raceuma1_df = raceuma_df.query(cond[0]).copy()
            temp_raceuma2_df = raceuma_df.query(cond[1]).copy()
            temp_raceuma3_df = raceuma_df.query(cond[2]).copy()
            temp_raceuma_df = self._subproc_create_sanrenpuku_df(temp_raceuma1_df, temp_raceuma2_df, temp_raceuma3_df, result_df)
            temp_raceuma_df.loc[:, "UMABAN_1"] = temp_raceuma_df["UMABAN"].str[0:2]
            temp_raceuma_df.loc[:, "UMABAN_2"] = temp_raceuma_df["UMABAN"].str[4:6]
            temp_raceuma_df.loc[:, "UMABAN_3"] = temp_raceuma_df["UMABAN"].str[8:10]
            base_df = pd.merge(temp_raceuma_df, odds_df, on=["RACE_KEY", "UMABAN_1", "UMABAN_2", "UMABAN_3"])
            base_df = base_df.rename(columns={"３連複オッズ": "オッズ"})
            low_odds_list = [10, 20, 30]
            high_odds_list = [30, 50, 80, 100, 150]
        else:
            base_df = ""; low_odds_list = []; high_odds_list = []
        return base_df, low_odds_list, high_odds_list

    def proc_fold_odds_simulation(self, type, cond):
        ym_list = self.raceuma_df["年月"].drop_duplicates().tolist()
        # 対象券種毎の結果データを取得
        base_df, low_odds_list, high_odds_list = self._subproc_get_odds_base_df(type, cond)
        base_sr = self._calc_summary(base_df, cond)
        sim_df = pd.DataFrame()
        for low_odds in low_odds_list:
            for high_odds in high_odds_list:
                i = 0
                if low_odds < high_odds:
                    odds_cond = f"{low_odds} <= オッズ <= {high_odds}"
                    temp_base_df = base_df.query(odds_cond).copy()
                    temp_base_sim_sr = self._calc_summary(temp_base_df, cond)
                    if temp_base_sim_sr["回収率"] >= base_sr["回収率"]:
                        for ym in ym_list:
                            temp_ym_base_df = temp_base_df.query(f"年月 == '{ym}'")
                            temp_sim_sr = self._calc_summary(temp_ym_base_df, cond)
                            if temp_sim_sr["回収率"] >= 100:
                                i = i + 1
                        temp_base_sim_sr["条件数"] = i
                        temp_base_sim_sr["オッズ条件"] = odds_cond
                        sim_df = sim_df.append(temp_base_sim_sr, ignore_index=True)
        if len(sim_df.index) > 0:
            final_sim_df = sim_df.sort_values(["条件数", "的中数", "回収率"], ascending=False).reset_index()
            print(final_sim_df.head())
            return final_sim_df.iloc[0]
        else:
            base_sr["オッズ条件"] = "0 <= オッズ <= 100000"
            return base_sr




    def _subproc_get_result_data(self, type):
        if type == "単勝":
            result_df = self.ext.get_tansho_df(self.haraimodoshi_df)
            raceuma_df = pd.merge(self.raceuma_df, result_df, on=["RACE_KEY", "UMABAN"], how="left").fillna(0)
        elif type == "複勝":
            result_df = self.ext.get_fukusho_df(self.haraimodoshi_df)
            raceuma_df = pd.merge(self.raceuma_df, result_df, on=["RACE_KEY", "UMABAN"], how="left").fillna(0)
        elif type == "馬連":
            result_df = self.ext.get_umaren_df(self.haraimodoshi_df)
            result_df["UMABAN"] = result_df["UMABAN"].apply(lambda x: ', '.join(map(str, x)))
            raceuma_df = self.raceuma_df.copy()
        elif type == "馬単":
            result_df = self.ext.get_umatan_df(self.haraimodoshi_df)
            result_df["UMABAN"] = result_df["UMABAN"].apply(lambda x: ', '.join(map(str, x)))
            raceuma_df = self.raceuma_df.copy()
        elif type == "ワイド":
            result_df = self.ext.get_wide_df(self.haraimodoshi_df)
            result_df["UMABAN"] = result_df["UMABAN"].apply(lambda x: ', '.join(map(str, x)))
            raceuma_df = self.raceuma_df.copy()
        elif type == "三連複":
            result_df = self.ext.get_sanrenpuku_df(self.haraimodoshi_df)
            result_df["UMABAN"] = result_df["UMABAN"].apply(lambda x: ', '.join(map(str, x)))
            raceuma_df = self.raceuma_df.copy()
        else:
            result_df = ""; raceuma_df = ""
        return result_df, raceuma_df

    def proc_fold_simulation(self, type):
        """ typeは単勝とか """
        ym_list = self.raceuma_df["年月"].drop_duplicates().tolist()
        # 対象券種毎の結果データを取得
        result_df, raceuma_df = self._subproc_get_result_data(type)

        # 的中件数が多く回収率が１０５％を超えている条件を抽出。的中件数の上位１０件を条件とする
        if type == "単勝" or type == "複勝":
            candidate_sim_df = self.proc_simulation_tanpuku(raceuma_df)
        elif type == "馬連" or type == "ワイド":
            candidate_sim_df = self.proc_simulation_umaren(raceuma_df, result_df)
        elif type == "馬単":
            candidate_sim_df = self.proc_simulation_umaren(raceuma_df, result_df, ren=False)
        elif type == "三連複":
            candidate_sim_df = self.proc_simulation_sanrenpuku(raceuma_df, result_df)
        if len(candidate_sim_df.index) == 0:
            print("対象なし")
            return pd.Series()
        else:
            target_sim_df = candidate_sim_df.sort_values("的中数", ascending=False).head(10)
            # 指定した条件の年月Foldを計算し、回収率１００％以上の件数が多い条件を採用。同数の場合は的中件数が多いものを採用
            print(target_sim_df)
            sim_df = pd.DataFrame()
            for index, row in target_sim_df.iterrows():
                cond = row["条件"]
                i = 0 #１００％超えカウント
                if type == "単勝" or type == "複勝":
                    temp_raceuma1_df = raceuma_df.query(cond).copy()
                elif type == "馬連" or type == "馬単" or type == "ワイド":
                    temp_raceuma1_df = raceuma_df.query(cond[0]).copy()
                    temp_raceuma2_df = raceuma_df.query(cond[1]).copy()
                elif type == "三連複":
                    temp_raceuma1_df = raceuma_df.query(cond[0]).copy()
                    temp_raceuma2_df = raceuma_df.query(cond[1]).copy()
                    temp_raceuma3_df = raceuma_df.query(cond[2]).copy()
                for ym in ym_list:
                    if type == "単勝" or type == "複勝":
                        temp_raceuma_df = temp_raceuma1_df.query(f"年月 == '{ym}'").copy()
                    elif type == "馬連" or type == "ワイド":
                        temp_ym_raceuma1_df = temp_raceuma1_df.query(f"年月 == '{ym}'").copy()
                        temp_ym_raceuma2_df = temp_raceuma2_df.query(f"年月 == '{ym}'").copy()
                        temp_raceuma_df = self._subproc_create_umaren_df(temp_ym_raceuma1_df, temp_ym_raceuma2_df, result_df)
                    elif type == "馬単":
                        temp_ym_raceuma1_df = temp_raceuma1_df.query(f"年月 == '{ym}'").copy()
                        temp_ym_raceuma2_df = temp_raceuma2_df.query(f"年月 == '{ym}'").copy()
                        temp_raceuma_df = self._subproc_create_umaren_df(temp_ym_raceuma1_df, temp_ym_raceuma2_df, result_df, ren=False)
                    elif type == "三連複":
                        temp_ym_raceuma1_df = temp_raceuma1_df.query(f"年月 == '{ym}'").copy()
                        temp_ym_raceuma2_df = temp_raceuma2_df.query(f"年月 == '{ym}'").copy()
                        temp_ym_raceuma3_df = temp_raceuma3_df.query(f"年月 == '{ym}'").copy()
                        temp_raceuma_df = self._subproc_create_sanrenpuku_df(temp_ym_raceuma1_df, temp_ym_raceuma2_df, temp_ym_raceuma3_df, result_df)

                    temp_sim_sr = self._calc_summary(temp_raceuma_df, cond)

                    if temp_sim_sr["回収率"] >= 100:
                        i = i + 1
                temp_final_sr = row
                temp_final_sr["条件数"] = i
                sim_df = sim_df.append(temp_final_sr, ignore_index=True)
            final_sim_df = sim_df.sort_values(["条件数", "的中数", "回収率"], ascending=False).reset_index()
            print(final_sim_df)
            return final_sim_df.iloc[0]


    def proc_simulation_tanpuku(self, raceuma_df):
        sim_df = pd.DataFrame()
        for rank in [1,2,3]:
            for win_prob in self.win_prob_list:
                for jiku_prob in self.jiku_prob_list:
                    for ana_prob in self.ana_prob_list:
                        cond = f"RANK <= {rank} and win_prob >= {win_prob} and jiku_prob >= {jiku_prob} and ana_prob >= {ana_prob}"
                        temp_raceuma_df = raceuma_df.query(cond)
                        if len(temp_raceuma_df.index) != 0:
                            temp_sr = self._calc_summary(temp_raceuma_df, cond)
                            if temp_sr["回収率"] >= 105 and temp_sr["購入基準"] == "1":
                                sim_df = sim_df.append(temp_sr, ignore_index=True)
        sim_df = sim_df.drop_duplicates(subset=["件数", "回収率", "払戻偏差"])
        return sim_df


    def proc_simulation_umaren(self, raceuma_df, result_df, ren=True):
        sim_df = pd.DataFrame()
        for rank1 in [1,2]:
            print(f"rank1: {rank1}")
            for win_prob in self.win_prob_list:
                for jiku_prob in self.jiku_prob_list:
                    for ana_prob in self.ana_prob_list:
                        cond1 = f"RANK <= {rank1} and win_prob >= {win_prob} and jiku_prob >= {jiku_prob} and ana_prob >= {ana_prob}"
                        temp_raceuma1_df = raceuma_df.query(cond1).copy()
                        for rank2 in [3,4,5]:
                            if rank1 < rank2:
                                for win_prob2 in [0, 0.3, 0.5]:
                                    for jiku_prob2 in [0, 0.2, 0.4]:
                                        for ana_prob2 in [0, 0.2, 0.3]:
                                            cond2 = f"RANK <= {rank2} and win_prob >= {win_prob2} and jiku_prob >= {jiku_prob2} and ana_prob >= {ana_prob2}"
                                            temp_raceuma2_df = raceuma_df.query(cond2).copy()
                                            temp_raceuma_df = self._subproc_create_umaren_df(temp_raceuma1_df, temp_raceuma2_df, result_df, ren)
                                            if len(temp_raceuma_df.index) != 0:
                                                cond = [cond1, cond2]
                                                temp_sr = self._calc_summary(temp_raceuma_df, cond)
                                                if temp_sr["回収率"] >= 110 and temp_sr["購入基準"] == "1":
                                                    sim_df = sim_df.append(temp_sr, ignore_index=True)
        sim_df = sim_df.drop_duplicates(subset=["件数", "回収率", "払戻偏差"])
        return sim_df

    def _subproc_create_umaren_df(self, raceuma1_df, raceuma2_df, result_df, ren=True):
        temp1_df = raceuma1_df[["RACE_KEY", "UMABAN", "年月"]].copy()
        temp2_df = raceuma2_df[["RACE_KEY", "UMABAN"]].copy()
        temp_df = pd.merge(temp1_df, temp2_df, on="RACE_KEY")
        temp_df = temp_df.query("UMABAN_x != UMABAN_y")
        if len(temp_df.index) != 0:
            if ren:
                temp_df.loc[:, "UMABAN"] = temp_df.apply(lambda x: x["UMABAN_x"] + ', ' + x["UMABAN_y"] if x["UMABAN_x"] < x["UMABAN_y"] else x["UMABAN_y"] + ', ' + x["UMABAN_x"], axis=1)
                temp_df = temp_df.drop_duplicates(subset=["RACE_KEY", "UMABAN"])
            else:
                temp_df.loc[:, "UMABAN"] = temp_df["UMABAN_x"] + ', ' + temp_df["UMABAN_y"]
            kaime_df = temp_df[["RACE_KEY", "UMABAN", "年月"]].copy()
            merge_df = pd.merge(kaime_df, result_df, on=["RACE_KEY", "UMABAN"], how="left").fillna(0)
            return merge_df
        else:
            return pd.DataFrame()


    def proc_simulation_sanrenpuku(self, raceuma_df, result_df):
        sim_df = pd.DataFrame()
        for win_prob in self.win_prob_list:
            for jiku_prob in self.jiku_prob_list:
                for ana_prob in self.ana_prob_list:
                    cond1 = f"RANK == 1 and win_prob >= {win_prob} and jiku_prob >= {jiku_prob} and ana_prob >= {ana_prob}"
                    temp_raceuma1_df = raceuma_df.query(cond1).copy()
                    for rank2 in [2,3]:
                        print(f"rank2: {rank2}")
                        for win_prob2 in [0, 0.3]:
                            for jiku_prob2 in [0, 0.2]:
                                for ana_prob2 in [0, 0.2]:
                                    cond2 = f"RANK <= {rank2} and win_prob >= {win_prob2} and jiku_prob >= {jiku_prob2} and ana_prob >= {ana_prob2}"
                                    temp_raceuma2_df = raceuma_df.query(cond2).copy()
                                    for rank3 in [5,6]:
                                        for win_prob3 in [0, 0.2]:
                                            for jiku_prob3 in [0, 0.1]:
                                                for ana_prob3 in [0, 0.1]:
                                                    cond3 = f"RANK <= {rank3} and win_prob >= {win_prob3} and jiku_prob >= {jiku_prob3} and ana_prob >= {ana_prob3}"
                                                    temp_raceuma3_df = raceuma_df.query(cond3).copy()
                                                    temp_raceuma_df = self._subproc_create_sanrenpuku_df(temp_raceuma1_df, temp_raceuma2_df, temp_raceuma3_df, result_df)
                                                    if len(temp_raceuma_df.index) != 0:
                                                        cond = [cond1, cond2, cond3]
                                                        temp_sr = self._calc_summary(temp_raceuma_df, cond)
                                                        if temp_sr["回収率"] >= 110 and temp_sr["購入基準"] == "1":
                                                            sim_df = sim_df.append(temp_sr, ignore_index=True)
        sim_df = sim_df.drop_duplicates(subset=["件数", "回収率", "払戻偏差"])
        return sim_df

    def _subproc_create_sanrenpuku_df(self, raceuma1_df, raceuma2_df, raceuma3_df, result_df):
        temp1_df = raceuma1_df[["RACE_KEY", "UMABAN", "年月"]].copy()
        temp2_df = raceuma2_df[["RACE_KEY", "UMABAN"]].copy()
        temp3_df = raceuma3_df[["RACE_KEY", "UMABAN"]].copy()
        temp_df = pd.merge(temp1_df, temp2_df, on="RACE_KEY")
        temp_df = pd.merge(temp_df, temp3_df, on="RACE_KEY")
        temp_df = temp_df.query("UMABAN_x != UMABAN_y and UMABAN_x != UMABAN and UMABAN_y != UMABAN")
        if len(temp_df.index) != 0:
            temp_df.loc[:, "連番"] = temp_df.apply(lambda x: sorted(list([x["UMABAN_x"], x["UMABAN_y"], x["UMABAN"]])), axis=1)
            temp_df.loc[:, "UMABAN"] = temp_df["連番"].apply(lambda x: ', '.join(map(str, x)))
            temp_df = temp_df.drop_duplicates(subset=["RACE_KEY", "UMABAN"])
            kaime_df = temp_df[["RACE_KEY", "UMABAN", "年月"]].copy()
            merge_df = pd.merge(kaime_df, result_df, on=["RACE_KEY", "UMABAN"], how="left").fillna(0)
            return merge_df
        else:
            return pd.DataFrame()


    def simulation_tansho(self, cond1, odds_cond):
        print("------- simulation_tansho -----------")
        print(f" 条件: {cond1}  オッズ:{odds_cond}")
        kaime_df = self.create_tansho_base_df(cond1)
        bet_df = kaime_df.query(odds_cond) if odds_cond != "" else kaime_df
        result_df = self.ext.get_tansho_df(self.haraimodoshi_df)
        check_df = pd.merge(bet_df, result_df, on=["RACE_KEY", "UMABAN"], how="left").fillna(0)
        cond_text = cond1
        sr = self._calc_summary(check_df, cond_text)
        return sr

    def create_tansho_base_df(self, cond1):
        print(cond1)
        df1 = self.raceuma_df.query(cond1)[["RACE_KEY", "UMABAN"]]
        odds_df = self.ext.get_odds_df("単勝")
        kaime_df = pd.merge(df1, odds_df.rename(columns={"単勝オッズ":"オッズ"}), on=["RACE_KEY", "UMABAN"])
        print(kaime_df.head())
        return kaime_df

    def simulation_fukusho(self, cond1, odds_cond):
        print("------- simulation_fukusho -----------")
        print(f" 条件: {cond1}  オッズ:{odds_cond}")
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
        kaime_df = pd.merge(df1, odds_df.rename(columns={"複勝オッズ":"オッズ"}), on=["RACE_KEY", "UMABAN"])
        return kaime_df

    def simulation_umaren(self, cond1, cond2, odds_cond):
        print("------- simulation_umaren -----------")
        print(f" 条件: {cond1}/{cond2}  オッズ:{odds_cond}")
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
        print(kaime_df.head())
        return kaime_df

    def simulation_wide(self, cond1, cond2, odds_cond):
        print("------- simulation_wide -----------")
        print(f" 条件: {cond1}/{cond2}  オッズ:{odds_cond}")
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
        print("------- simulation_umatan -----------")
        print(f" 条件: {cond1}/{cond2}  オッズ:{odds_cond}")
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
        print("------- simulation_sanrenpuku -----------")
        print(f" 条件: {cond1}/{cond2}/{cond3}  オッズ:{odds_cond}")
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
        race_hit_count = len(hit_df["RACE_KEY"].drop_duplicates())
        avg_return = round(hit_df["払戻"].mean(), 0)
        std_return = round(hit_df["払戻"].std(), 0)
        max_return = hit_df["払戻"].max()
        sum_return = hit_df["払戻"].sum()
        avg = round(df["払戻"].mean() , 1)
        hit_rate = round(hit_count / all_count * 100 , 1) if all_count !=0 else 0
        race_hit_rate = round(race_hit_count / race_count * 100 , 1) if race_count !=0 else 0
        vote_check = "1" if sum_return - max_return > all_count * 100 else "0"
        sr = pd.Series(data=[cond_text, all_count, hit_count, race_count, avg, hit_rate, race_hit_rate, avg_return, std_return, max_return, all_count * 100 , sum_return, vote_check]
                       , index=["条件", "件数", "的中数", "レース数", "回収率", "的中率", "R的中率", "払戻平均", "払戻偏差", "最大払戻", "購入総額", "払戻総額", "購入基準"])
        return sr.fillna(0)
