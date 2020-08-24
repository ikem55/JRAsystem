from modules.extract import Extract
from modules.transform import Transform
from modules.simulation import Simulation
import modules.util as mu

import slackweb
import pandas as pd
import my_config as mc
import os
from datetime import datetime as dt
from datetime import timedelta
import glob

class Output(object):
    slack_operation_url = mc.SLACK_operation_webhook_url
    slack_summary_url = mc.SLACK_summary_webhook_url
    slack_realtime_url = mc.SLACK_realtime_webhook_url
    kaime_columns = ["RACE_ID", "エリア", "券種", "購入金額", "目１", "目２", "目３"]

    def __init__(self, start_date, end_date, term_start_date, term_end_date, test_flag, target_sr, cond_df):
        self.start_date = start_date
        self.end_date = end_date
        self.term_start_date = term_start_date
        self.term_end_date = term_end_date
        mock_flag = False
        self.ext = Extract(start_date, end_date, mock_flag)
        self.tf = Transform(start_date, end_date)
        self.dict_path = mc.return_jra_path(test_flag)
        self.pred_folder_path = self.dict_path + 'pred/'
        self.target_path = mc.return_target_path(test_flag)
        self.ext_score_path = self.target_path + 'ORIGINAL_DATA/'
        self.auto_bet_path = self.target_path + 'AUTO_BET/'
        self.for_pbi_path = self.target_path + 'pbi/'
        self._set_file_list()
        self._set_vote_condition(target_sr, cond_df)

    def _set_vote_condition(self, target_sr, cond_df):
        self.win_rate = target_sr["win_rate"]
        self.jiku_rate = target_sr["jiku_rate"]
        self.ana_rate = target_sr["ana_rate"]
        print(f"----- mark rate: win:{self.win_rate}% jiku:{self.jiku_rate}% ana:{self.ana_rate}%")
        tansho_sr = cond_df.query("タイプ == '単勝'")
        print(tansho_sr["オッズ条件"])
        self.tansho_flag = False if tansho_sr.empty else True
        if self.tansho_flag:
            self.tansho_cond = tansho_sr["条件"].values[0]
            self.tansho_odds_cond = tansho_sr["オッズ条件"].values[0]
        fukusho_sr = cond_df.query("タイプ == '複勝'")
        self.fukusho_flag = False if fukusho_sr.empty else True
        if self.fukusho_flag:
            self.fukusho_cond = fukusho_sr["条件"].values[0]
            self.fukusho_odds_cond = fukusho_sr["オッズ条件"].values[0]
        umaren_sr = cond_df.query("タイプ == '馬連'")
        self.umaren_flag = False if umaren_sr.empty else True
        if self.umaren_flag:
            self.umaren1_cond = umaren_sr["条件"].values[0][0]
            self.umaren2_cond = umaren_sr["条件"].values[0][1]
            self.umaren_odds_cond = umaren_sr["オッズ条件"].values[0]
        umatan_sr = cond_df.query("タイプ == '馬単'")
        self.umatan_flag = False if umatan_sr.empty else True
        if self.umatan_flag:
            self.umatan1_cond = umatan_sr["条件"].values[0][0]
            self.umatan2_cond = umatan_sr["条件"].values[0][1]
            self.umatan_odds_cond = umatan_sr["オッズ条件"].values[0]
        wide_sr = cond_df.query("タイプ == 'ワイド'")
        self.wide_flag = False if wide_sr.empty else True
        if self.wide_flag:
            self.wide1_cond = wide_sr["条件"].values[0][0]
            self.wide2_cond = wide_sr["条件"].values[0][1]
            self.wide_odds_cond = wide_sr["オッズ条件"].values[0]
        sanrenpuku_sr = cond_df.query("タイプ == '三連複'")
        self.sanrenpuku_flag = False if sanrenpuku_sr.empty else True
        if self.sanrenpuku_flag:
            self.sanrenpuku1_cond = sanrenpuku_sr["条件"].values[0][0]
            self.sanrenpuku2_cond = sanrenpuku_sr["条件"].values[0][1]
            self.sanrenpuku3_cond = sanrenpuku_sr["条件"].values[0][2]
            self.sanrenpuku_odds_cond = sanrenpuku_sr["オッズ条件"].values[0]

    def _set_file_list(self):
        race_base_df = self.ext.get_race_before_table_base()[["RACE_KEY", "NENGAPPI", "距離", "芝ダ障害コード", "内外", "条件"]]
        race_base_df.loc[:, "RACE_ID"] = race_base_df.apply(lambda x: mu.convert_jrdb_id(x["RACE_KEY"], x["NENGAPPI"]),
                                                            axis=1)
        race_base_df.loc[:, "file_id"] = race_base_df["RACE_KEY"].apply(lambda x: mu.convert_target_file(x))
        race_base_df.loc[:, "nichiji"] = race_base_df["RACE_KEY"].apply(lambda x: mu.convert_kaiji(x[5:6]))
        race_base_df.loc[:, "race_no"] = race_base_df["RACE_KEY"].str[6:8]
        race_base_df.loc[:, "rc_file_id"] = race_base_df["RACE_KEY"].apply(lambda x: "RC" + x[0:5])
        race_base_df.loc[:, "kc_file_id"] = "KC" + race_base_df["RACE_KEY"].str[0:6]

        update_term_df = race_base_df.query(f"NENGAPPI >= '{self.term_start_date}' and NENGAPPI <= '{self.term_end_date}'")
        print(race_base_df.shape)
        print(f"NENGAPPI >= '{self.term_start_date}' and NENGAPPI <= '{self.term_end_date}'")
        print(update_term_df.shape)
        self.race_base_df = race_base_df
        self.file_list = update_term_df["file_id"].drop_duplicates()
        self.date_list = update_term_df["NENGAPPI"].drop_duplicates()
        self.rc_file_list = update_term_df["rc_file_id"].drop_duplicates()
        self.kc_file_list = update_term_df["kc_file_id"].drop_duplicates()


    def get_pred_df(self, model_version, target):
        """ 予測したtargetのデータを取得する """
        target_filelist = self._get_file_list_for_pred(model_version)
        df = pd.DataFrame()
        for filename in target_filelist:
            temp_df = pd.read_pickle(filename)
            df = pd.concat([df, temp_df])
        df = df.query(f"target == '{target}'")
        return df

    def _get_file_list_for_pred(self, folder):
        """ predで予測したファイルの対象リストを取得する"""
        folder_path = self.pred_folder_path + folder + "/*.pkl"
        filelist = glob.glob(folder_path)
        file_df = pd.DataFrame({"filename": filelist})
        file_df.loc[:, "date"] = file_df["filename"].apply(lambda x: dt.strptime(x[-12:-4], '%Y%m%d'))
        target_filelist = file_df[(file_df["date"] >= self.start_date) & (file_df["date"] <= self.end_date)]["filename"].tolist()
        return sorted(target_filelist)

    def set_result_df(self):
        race_table_base_df = self.ext.get_race_table_base().drop(["馬場状態", "target_date", "距離", "芝ダ障害コード", "頭数"], axis=1)
        race_table_base_df.loc[:, "RACE_ID"] = race_table_base_df.apply(
            lambda x: mu.convert_jrdb_id(x["RACE_KEY"], x["NENGAPPI"]), axis=1)
        race_table_base_df.loc[:, "file_id"] = race_table_base_df["RACE_KEY"].apply(lambda x: mu.convert_target_file(x))
        race_table_base_df.loc[:, "nichiji"] = race_table_base_df["RACE_KEY"].apply(lambda x: mu.convert_kaiji(x[5:6]))
        race_table_base_df.loc[:, "race_no"] = race_table_base_df["RACE_KEY"].str[6:8]
        raceuma_table_base_df = self.ext.get_raceuma_table_base()
        result_df = pd.merge(race_table_base_df, raceuma_table_base_df, on="RACE_KEY")
        result_df.loc[:, "距離"] = result_df["距離"].astype(int)

        cluster_raceuma_result_df = self.tf.cluster_raceuma_result_df(result_df, self.dict_path)
        factory_analyze_race_result_df = self.tf.factory_analyze_race_result_df(result_df, self.dict_path)

        raceuma_result_df = cluster_raceuma_result_df[["RACE_KEY", "UMABAN", "ru_cluster", "ＩＤＭ結果", "レース馬コメント"]]
        race_result_df = factory_analyze_race_result_df[
            ["RACE_KEY", "target_date", "fa_1", "fa_2", "fa_3", "fa_4", "fa_5", "RAP_TYPE", "TRACK_BIAS_ZENGO",
             "TRACK_BIAS_UCHISOTO", "レースペース流れ", "レースコメント"]]

        race_result_df.loc[:, "val"] = race_result_df["RAP_TYPE"].apply(
            lambda x: mu.decode_rap_type(int(mu.encode_rap_type(x))))
        race_result_df.loc[:, "TB_ZENGO"] = race_result_df["TRACK_BIAS_ZENGO"].apply(
            lambda x: mu._decode_zengo_bias(int(mu._encode_zengo_bias(x))))
        race_result_df.loc[:, "TB_UCHISOTO"] = race_result_df["TRACK_BIAS_UCHISOTO"].apply(
            lambda x: mu._decode_uchisoto_bias(int(mu._calc_uchisoto_bias(x))))
        race_result_df.loc[:, "RACE_PACE"] = race_result_df["レースペース流れ"].apply(
            lambda x: mu._decode_race_pace(int(mu._encode_race_pace(x))))
        race_result_df.loc[:, "TB"] = race_result_df.apply(lambda x: mu.convert_bias(x["TB_UCHISOTO"], x["TB_ZENGO"]),
                                                           axis=1)
        race_result_df = race_result_df.groupby("RACE_KEY").first().reset_index()
        race_result_df = pd.merge(race_result_df, self.race_base_df, on="RACE_KEY")

        result_uchisoto_df = race_result_df[["RACE_KEY", "TB_UCHISOTO", "file_id", "nichiji", "race_no"]].rename(
            columns={"TB_UCHISOTO": "val"})
        result_zengo_df = race_result_df[["RACE_KEY", "TB_ZENGO", "file_id", "nichiji", "race_no"]].rename(
            columns={"TB_ZENGO": "val"})
        result_tb_df = race_result_df[["RACE_KEY", "TB", "file_id", "nichiji", "race_no"]].rename(columns={"TB": "val"})

        raceuma_result_df = pd.merge(raceuma_result_df, race_result_df, on="RACE_KEY")

        raceuma_result_df.loc[:, "RACEUMA_ID"] = raceuma_result_df.apply(
            lambda x: mu.convert_jrdb_id(x["RACE_KEY"], x["target_date"]) + x["UMABAN"], axis=1)
        fa_df = raceuma_result_df[["RACEUMA_ID", "fa_1", "fa_2", "fa_3", "fa_4", "fa_5", "target_date"]]
        self.race_result_df = race_result_df
        self.raceuma_result_df = raceuma_result_df

    def set_pred_df(self):
        win5_df = self.get_pred_df("win5", "WIN5_FLAG")
        win5_df.loc[:, "RACEUMA_ID"] = win5_df.apply(
            lambda x: mu.convert_jrdb_id(x["RACE_KEY"], x["target_date"]) + x["UMABAN"], axis=1)
        win5_df.loc[:, "predict_std"] = round(win5_df["predict_std"], 2)
        win5_df.loc[:, "predict_rank"] = win5_df["predict_rank"].astype(int)

        win_df = self.get_pred_df("win", "WIN_FLAG")
        win_df.loc[:, "RACEUMA_ID"] = win_df.apply(
            lambda x: mu.convert_jrdb_id(x["RACE_KEY"], x["target_date"]) + x["UMABAN"], axis=1)
        win_df.loc[:, "predict_std"] = round(win_df["predict_std"], 2)
        win_df.loc[:, "predict_rank"] = win_df["predict_rank"].astype(int)

        jiku_df = self.get_pred_df("win", "JIKU_FLAG")
        jiku_df.loc[:, "RACEUMA_ID"] = jiku_df.apply(
            lambda x: mu.convert_jrdb_id(x["RACE_KEY"], x["target_date"]) + x["UMABAN"], axis=1)
        jiku_df.loc[:, "predict_std"] = round(jiku_df["predict_std"], 2)
        jiku_df.loc[:, "predict_rank"] = jiku_df["predict_rank"].astype(int)

        ana_df = self.get_pred_df("win", "ANA_FLAG")
        ana_df.loc[:, "RACEUMA_ID"] = ana_df.apply(
            lambda x: mu.convert_jrdb_id(x["RACE_KEY"], x["target_date"]) + x["UMABAN"], axis=1)
        ana_df.loc[:, "predict_std"] = round(ana_df["predict_std"], 2)
        ana_df.loc[:, "predict_rank"] = ana_df["predict_rank"].astype(int)
        score_df = pd.merge(win_df[["RACE_KEY", "UMABAN", "RACEUMA_ID", "predict_std", "target_date"]].rename(
            columns={"predict_std": "win_std"}),
                            jiku_df[["RACEUMA_ID", "predict_std"]].rename(columns={"predict_std": "jiku_std"}),
                            on="RACEUMA_ID")
        score_df = pd.merge(score_df, ana_df[["RACEUMA_ID", "predict_std"]].rename(columns={"predict_std": "ana_std"}),
                            on="RACEUMA_ID")
        score_df.loc[:, "predict_std"] = score_df["win_std"] * self.win_rate / 100 + score_df["jiku_std"] * self.jiku_rate / 100  + score_df["ana_std"] * self.ana_rate / 100
        grouped_score_df = score_df.groupby("RACE_KEY")
        score_df.loc[:, "predict_rank"] = grouped_score_df["predict_std"].rank("dense", ascending=False)
        score_df.loc[:, "predict_std"] = round(score_df["predict_std"], 2)
        score_df.loc[:, "predict_rank"] = score_df["predict_rank"].astype(int)
        nigeuma_df = self.get_pred_df("nigeuma", "NIGEUMA")
        nigeuma_df.loc[:, "RACEUMA_ID"] = nigeuma_df.apply(
            lambda x: mu.convert_jrdb_id(x["RACE_KEY"], x["target_date"]) + x["UMABAN"], axis=1)
        nigeuma_df.loc[:, "predict_std"] = round(nigeuma_df["predict_std"], 2)
        nigeuma_df.loc[:, "predict_rank"] = nigeuma_df["predict_rank"].astype(int)
        agari_df = self.get_pred_df("nigeuma", "AGARI_SAISOKU")
        agari_df.loc[:, "RACEUMA_ID"] = agari_df.apply(
            lambda x: mu.convert_jrdb_id(x["RACE_KEY"], x["target_date"]) + x["UMABAN"], axis=1)
        agari_df.loc[:, "predict_std"] = round(agari_df["predict_std"], 2)
        agari_df.loc[:, "predict_rank"] = agari_df["predict_rank"].astype(int)
        ten_df = self.get_pred_df("nigeuma", "TEN_SAISOKU")
        ten_df.loc[:, "RACEUMA_ID"] = ten_df.apply(
            lambda x: mu.convert_jrdb_id(x["RACE_KEY"], x["target_date"]) + x["UMABAN"], axis=1)
        ten_df.loc[:, "predict_std"] = round(ten_df["predict_std"], 2)
        ten_df.loc[:, "predict_rank"] = ten_df["predict_rank"].astype(int)
        self.win5_df = win5_df
        self.win_df = win_df
        self.jiku_df = jiku_df
        self.ana_df = ana_df
        self.score_df = score_df
        self.nigeuma_df = nigeuma_df
        self.agari_df = agari_df
        self.ten_df = ten_df
        total_df = pd.concat([win_df, jiku_df, ana_df, nigeuma_df, agari_df, ten_df])[["RACE_KEY", "UMABAN", "target_date", "predict_rank"]]
        point_df = total_df.copy()
        point_df.loc[:, "predict_rank"] = point_df["predict_rank"].apply(lambda x: 5 if x == 1 else 4 if x == 2 else 3 if x == 3 else 2 if x == 4 else 1 if x == 5 else 0)
        ## 単純に順位が少ないものを評価
        total_df = total_df.groupby(["RACE_KEY", "UMABAN", "target_date"]).sum().reset_index()
        total_df.columns = ["RACE_KEY", "UMABAN", "target_date", "VALUE"]
        total_df.loc[:, "predict_rank"] = total_df.groupby("RACE_KEY")["VALUE"].rank(method='min')
        self.total_df = total_df

        ##　順位上位のものにpointをつけて評価
        point_df = point_df.groupby(["RACE_KEY", "UMABAN", "target_date"]).sum().reset_index()
        point_df.columns = ["RACE_KEY", "UMABAN", "target_date", "VALUE"]
        point_df.loc[:, "predict_rank"] = point_df.groupby("RACE_KEY")["VALUE"].rank(ascending=False, method='min')
        self.point_df = point_df

    def create_raceuma_score_file(self):
        for date in self.date_list:
            print(date)
            win5_temp_df = self.win5_df.query(f"target_date == '{date}'")[
                ["RACEUMA_ID", "prob", "predict_rank"]].sort_values("RACEUMA_ID")
            win5_temp_df.loc[:, "prob"] = (win5_temp_df["prob"] * 100 ).astype("int")
            win5_temp_df.to_csv(self.ext_score_path + "pred_win5/" + date + ".csv", header=False, index=False)

            win_temp_df = self.win_df.query(f"target_date == '{date}'")[
                ["RACEUMA_ID", "prob", "predict_rank"]].sort_values("RACEUMA_ID")
            win_temp_df.loc[:, "prob"] = (win_temp_df["prob"] * 100 ).astype("int")
            win_temp_df.to_csv(self.ext_score_path + "pred_win/" + date + ".csv", header=False, index=False)
            jiku_temp_df = self.jiku_df.query(f"target_date == '{date}'")[
                ["RACEUMA_ID", "prob", "predict_rank"]].sort_values("RACEUMA_ID")
            jiku_temp_df.loc[:, "prob"] = (jiku_temp_df["prob"] * 100 ).astype("int")
            jiku_temp_df.to_csv(self.ext_score_path + "pred_jiku/" + date + ".csv", header=False, index=False)
            ana_temp_df = self.ana_df.query(f"target_date == '{date}'")[
                ["RACEUMA_ID", "prob", "predict_rank"]].sort_values("RACEUMA_ID")
            ana_temp_df.loc[:, "prob"] = (ana_temp_df["prob"] * 100 ).astype("int")
            ana_temp_df.to_csv(self.ext_score_path + "pred_ana/" + date + ".csv", header=False, index=False)
            score_temp_df = self.score_df.query(f"target_date == '{date}'")[
                ["RACEUMA_ID", "predict_std", "predict_rank"]].sort_values("RACEUMA_ID")
            score_temp_df.loc[:, "predict_std"] = score_temp_df["predict_std"].round(0).astype("int")
            score_temp_df.to_csv(self.ext_score_path + "pred_score/" + date + ".csv", header=False, index=False)
            nigeuma_temp_df = self.nigeuma_df.query(f"target_date == '{date}'")[
                ["RACEUMA_ID", "predict_std", "predict_rank"]].sort_values("RACEUMA_ID")
            nigeuma_temp_df.to_csv(self.ext_score_path + "pred_nige/" + date + ".csv", header=False, index=False)
            agari_temp_df = self.agari_df.query(f"target_date == '{date}'")[
                ["RACEUMA_ID", "predict_std", "predict_rank"]].sort_values("RACEUMA_ID")
            agari_temp_df.to_csv(self.ext_score_path + "pred_agari/" + date + ".csv", header=False, index=False)
            ten_temp_df = self.ten_df.query(f"target_date == '{date}'")[
                ["RACEUMA_ID", "predict_std", "predict_rank"]].sort_values("RACEUMA_ID")
            ten_temp_df.to_csv(self.ext_score_path + "pred_ten/" + date + ".csv", header=False, index=False)

    def create_main_mark_file(self):
        umaren_are_df = self.get_pred_df("haito", "UMAREN_ARE")[["RACE_KEY", "pred"]].rename(
            columns={"pred": "umaren_are"})
        umatan_are_df = self.get_pred_df("haito", "UMATAN_ARE")[["RACE_KEY", "pred"]].rename(
            columns={"pred": "umatan_are"})
        sanrenpuku_are_df = self.get_pred_df("haito", "SANRENPUKU_ARE")[["RACE_KEY", "pred"]].rename(
            columns={"pred": "sanrenpuku_are"})
        are_df = pd.merge(umaren_are_df, umatan_are_df, on="RACE_KEY")
        are_df = pd.merge(are_df, sanrenpuku_are_df, on="RACE_KEY")
        are_df = pd.merge(self.race_base_df, are_df, on="RACE_KEY", how="left")
        are_df.loc[:, "val"] = are_df.apply(
            lambda x: mu.convert_are_flag(x["umaren_are"], x["umatan_are"], x["sanrenpuku_are"]), axis=1)

        main_raceuma_df = pd.merge(self.score_df, self.race_base_df, on ="RACE_KEY")
        """ 馬１、レース１用のファイルを作成 """
        for file in self.file_list:
            print(file)
            file_text = ""
            temp_df = self.race_base_df.query(f"file_id == '{file}'")
            nichiji_list = sorted(temp_df["nichiji"].drop_duplicates())
            for nichiji in nichiji_list:
                temp2_df = temp_df.query(f"nichiji == '{nichiji}'")
                race_list = sorted(temp2_df["RACE_KEY"].drop_duplicates())
                for race in race_list:
                    line_text = ""
                    temp_race_df = are_df.query(f"RACE_KEY =='{race}'")
                    if not temp_race_df.empty:
                        temp3_sr = temp_race_df.iloc[0]
                        if temp3_sr["val"] == temp3_sr["val"]:
                            line_text += temp3_sr["val"]
                        else:
                            line_text += "      "
                    else:
                        line_text += "      "
                    temp3_df = main_raceuma_df.query(f"RACE_KEY == '{race}'").sort_values("UMABAN")
                    i = 0
                    for idx, val in temp3_df.iterrows():
                        line_text += self._return_mark(val["predict_rank"])
                        i += 1
                    if i != 18:
                        for j in range(i, 18):
                            line_text += "  "
                    file_text += line_text + '\r\n'
            with open(self.target_path + "UM" + file + ".DAT", mode='w', encoding="shift-jis") as f:
                f.write(file_text.replace('\r', ''))


    def create_raceuma_mark_file(self):
        print("---- WIN5マーク --------")
        mark_path_2 = self.target_path + "UmaMark2/"
        self._proc_create_um_mark_file(self.win5_df, mark_path_2)
        print("---- pointマーク --------")
        mark_path_3 = self.target_path + "UmaMark3/"
        self._proc_create_um_mark_file(self.point_df, mark_path_3)
        print("---- 勝ちマーク --------")
        mark_path_4 = self.target_path + "UmaMark4/"
        self._proc_create_um_mark_file(self.win_df, mark_path_4)
        print("---- 軸マーク --------")
        mark_path_5 = self.target_path + "UmaMark5/"
        self._proc_create_um_mark_file(self.jiku_df, mark_path_5)
        print("---- nigeuma_df --------")
        mark_path_6 = self.target_path + "UmaMark6/"
        self._proc_create_um_mark_file(self.nigeuma_df, mark_path_6)
        print("---- agari_df --------")
        mark_path_7 = self.target_path + "UmaMark7/"
        self._proc_create_um_mark_file(self.agari_df, mark_path_7)
        print("---- 穴マーク --------")
        mark_path_8 = self.target_path + "UmaMark8/"
        self._proc_create_um_mark_file(self.ana_df, mark_path_8)

    def _proc_create_um_mark_file(self, df, folder_path):
        """ ランクを印にして馬印ファイルを作成 """
        df.loc[:, "RACEUMA_ID"] = df.apply(
            lambda x: mu.convert_jrdb_id(x["RACE_KEY"], x["target_date"]) + x["UMABAN"], axis=1)
        #df.loc[:, "predict_std"] = df["predict_std"].round(2)
        df.loc[:, "predict_rank"] = df["predict_rank"].astype(int)
        df = pd.merge(self.race_base_df[["RACE_KEY", "file_id", "nichiji", "race_no"]], df, on="RACE_KEY", how="left")
        # file_list = df["file_id"].drop_duplicates()
        for file in self.file_list:
            print(file)
            file_text = ""
            temp_df = df.query(f"file_id == '{file}'")
            nichiji_list = sorted(temp_df["nichiji"].drop_duplicates())
            for nichiji in nichiji_list:
                temp2_df = temp_df.query(f"nichiji == '{nichiji}'")
                race_list = sorted(temp2_df["RACE_KEY"].drop_duplicates())
                for race in race_list:
                    line_text = "      "
                    temp3_df = temp2_df.query(f"RACE_KEY == '{race}'").sort_values("UMABAN")
                    i = 0
                    for idx, val in temp3_df.iterrows():
                        line_text += self._return_mark(val["predict_rank"])
                        i += 1
                    if i != 18:
                        for j in range(i, 18):
                            line_text += "  "
                    file_text += line_text + '\r\n'
            with open(folder_path + "UM" + file + ".DAT", mode='w', encoding="shift-jis") as f:
                f.write(file_text.replace('\r', ''))

    def create_target_mark_df(self):
        base_df = pd.concat([self.win_df, self.jiku_df, self.ana_df])
        mark_base_df = base_df[["RACE_KEY", "UMABAN", "target", "predict_std"]].copy()
        mark_base_df = mark_base_df.set_index(["RACE_KEY", "UMABAN", "target"]).unstack("target")
        mark_base_df.columns = ["ANA_FLAG", "JIKU_FLAG", "WIN_FLAG"]
        mark_base_df = mark_base_df.reset_index()
        mark_df = mark_base_df.copy()
        mark_df.loc[:, "SCORE"] = mark_df["WIN_FLAG"] * self.win_rate / 100  + mark_base_df["JIKU_FLAG"] / 100  * self.jiku_rate + mark_base_df["ANA_FLAG"] * self.ana_rate / 100
        mark_df.loc[:, "RANK"] = mark_df.groupby("RACE_KEY")["SCORE"].rank(ascending=False)
        mark_prob_df = base_df[["RACE_KEY", "UMABAN", "target", "prob"]].copy()
        mark_prob_df = mark_prob_df.set_index(["RACE_KEY", "UMABAN", "target"]).unstack("target")
        mark_prob_df.columns = ["ana_prob", "jiku_prob", "win_prob"]
        mark_prob_df = mark_prob_df.reset_index()
        self.target_mark_df = pd.merge(mark_df, mark_prob_df, on=["RACE_KEY", "UMABAN"])

        race_df = self.ext.get_race_before_table_base()
        base_term_df = race_df.query(f"NENGAPPI >= '{self.term_start_date}' and NENGAPPI <= '{self.term_end_date}'")[["RACE_KEY"]].copy()
        self.race_df = race_df[["RACE_KEY", "距離", "芝ダ障害コード", "種別", "条件", "天候コード", "芝馬場状態コード", "ダ馬場状態コード", "target_date"]].copy()
        self.race_df.loc[:, "年月"] = self.race_df["target_date"].str[0:6]
        self.race_df = pd.merge(self.race_df, base_term_df, on="RACE_KEY")

        res_raceuma_df = self.ext.get_raceuma_table_base()[["RACE_KEY", "UMABAN", "着順", "確定単勝オッズ", "確定単勝人気順位", "レース脚質", "単勝", "複勝", "テン指数結果順位", "上がり指数結果順位"]].copy()
        self.res_raceuma_df = res_raceuma_df

    def create_vote_file(self):
        target_df = self.target_mark_df.copy()
        print(target_df.shape)
        print(target_df.iloc[0])
        sim = Simulation(self.start_date, self.end_date, False, target_df)
        tansho_target_bet_df = pd.DataFrame(); fukusho_target_bet_df = pd.DataFrame(); umaren_target_bet_df = pd.DataFrame()
        umatan_target_bet_df = pd.DataFrame(); wide_target_bet_df = pd.DataFrame(); sanrenpuku_target_bet_df = pd.DataFrame()
        print("--- tansho ----")
        if self.tansho_flag:
            print(f" 条件: {self.tansho_cond}  オッズ:{self.tansho_odds_cond}")
            tansho_kaime_df = sim.create_tansho_base_df(self.tansho_cond)
            print(tansho_kaime_df.shape)
            tansho_target_bet_df = self._get_tansho_bet_df(tansho_kaime_df)
            print(tansho_target_bet_df.shape)
        print("--- fukusho ----")
        if self.fukusho_flag:
            print(f" 条件: {self.fukusho_cond}  オッズ:{self.fukusho_odds_cond}")
            fukusho_kaime_df = sim.create_fukusho_base_df(self.fukusho_cond)
            print(fukusho_kaime_df.shape)
            fukusho_target_bet_df = self._get_fukusho_bet_df(fukusho_kaime_df)
            print(fukusho_target_bet_df.shape)
        print("--- umaren ----")
        if self.umaren_flag:
            print(f" 条件: {self.umaren1_cond}/ {self.umaren2_cond}  オッズ:{self.umaren_odds_cond}")
            umaren_kaime_df = sim.create_umaren_base_df(self.umaren1_cond, self.umaren2_cond)
            print(umaren_kaime_df.shape)
            umaren_target_bet_df = self._get_umaren_bet_df(umaren_kaime_df)
            print(umaren_target_bet_df.shape)
        print("--- umatan ----")
        if self.umatan_flag:
            print(f" 条件: {self.umatan1_cond}/ {self.umatan2_cond}  オッズ:{self.umatan_odds_cond}")
            umatan_kaime_df = sim.create_umatan_base_df(self.umatan1_cond, self.umatan2_cond)
            print(umatan_kaime_df.shape)
            umatan_target_bet_df = self._get_umatan_bet_df(umatan_kaime_df)
            print(umatan_target_bet_df.shape)
        print("--- wide ----")
        if self.wide_flag:
            print(f" 条件: {self.wide1_cond}/ {self.wide2_cond}  オッズ:{self.wide_odds_cond}")
            wide_kaime_df = sim.create_wide_base_df(self.wide1_cond, self.wide2_cond)
            print(wide_kaime_df.shape)
            wide_target_bet_df = self._get_wide_bet_df(wide_kaime_df)
            print(wide_target_bet_df.shape)
        print("--- sanrenpuku ----")
        if self.sanrenpuku_flag:
            print(f" 条件: {self.sanrenpuku1_cond}/ {self.sanrenpuku2_cond}/ {self.sanrenpuku3_cond}  オッズ:{self.sanrenpuku_odds_cond}")
            sanrenpuku_kaime_df = sim.create_sanrenpuku_base_df(self.sanrenpuku1_cond, self.sanrenpuku2_cond, self.sanrenpuku3_cond)
            sanrenpuku_target_bet_df = self._get_sanrenpuku_bet_df(sanrenpuku_kaime_df)
            print(sanrenpuku_target_bet_df.shape)
        target_bet_df = pd.concat([tansho_target_bet_df, fukusho_target_bet_df, umaren_target_bet_df, umatan_target_bet_df, wide_target_bet_df, sanrenpuku_target_bet_df])
        target_bet_df = target_bet_df.sort_values(["RACE_ID", "エリア", "券種", "購入金額", "目１", "目２", "目３"])
        target_bet_df = target_bet_df.drop_duplicates(subset=["RACE_ID", "券種", "目１", "目２", "目３"])
        target_bet_df.to_csv(self.auto_bet_path + "target_bet.csv", index=False, header=False)
        self.target_bet_df = target_bet_df

    def _get_tansho_bet_df(self, kaime_df):
        if len(kaime_df.index) == 0:
            print("_get_tansho_bet_df -- no data--")
            return pd.DataFrame(columns=self.kaime_columns)
        # tansho_base_df = self.get_tansho_target_df(target_df)
        bet_df = kaime_df.query(self.tansho_odds_cond).copy()
        if len(bet_df.index) == 0:
            return pd.DataFrame(columns=self.kaime_columns)
        else:
            bet_df.loc[:, "目１"] =bet_df["UMABAN"]
            bet_df.loc[:, "目２"] =""
            bet_df.loc[:, "目３"] =""
            bet_df.loc[:, "券種"] = '0'
            # bet_df.loc[:, "オッズ"] = bet_df["単勝オッズ"]
            target_bet_df = self._get_target_bet_df(bet_df)
            return target_bet_df

    def _get_fukusho_bet_df(self, kaime_df):
        if len(kaime_df.index) == 0:
            print("_get_fukusho_bet_df -- no data--")
            return pd.DataFrame(columns=self.kaime_columns)
        # tansho_base_df = self.get_fukusho_target_df(target_df)
        bet_df = kaime_df.query(self.fukusho_odds_cond).copy()
        if len(bet_df.index) == 0:
            return pd.DataFrame(columns=self.kaime_columns)
        else:
            bet_df.loc[:, "目１"] =bet_df["UMABAN"]
            bet_df.loc[:, "目２"] =""
            bet_df.loc[:, "目３"] =""
            bet_df.loc[:, "券種"] = '1'
            # bet_df.loc[:, "オッズ"] = bet_df["複勝オッズ"]
            target_bet_df = self._get_target_bet_df(bet_df)
            return target_bet_df

    def _get_umaren_bet_df(self, kaime_df):
        if len(kaime_df.index) == 0:
            print("_get_umaren_bet_df -- no data--")
            return pd.DataFrame(columns=self.kaime_columns)
        #umaren_base_df = self.get_umaren_target_df(uma1_df, uma2_df)
        bet_df = kaime_df.query(self.umaren_odds_cond).copy()
        if len(bet_df.index) == 0:
            return pd.DataFrame(columns=self.kaime_columns)
        else:
            bet_df.loc[:, "目１"] =bet_df["UMABAN"].apply(lambda x: x[0])
            bet_df.loc[:, "目２"] =bet_df["UMABAN"].apply(lambda x: x[1])
            bet_df.loc[:, "目３"] =""
            bet_df.loc[:, "券種"] = '3'
            target_bet_df = self._get_target_bet_df(bet_df)
            return target_bet_df

    def _get_umatan_bet_df(self, kaime_df):
        if len(kaime_df.index) == 0:
            print("_get_umatan_bet_df -- no data--")
            return pd.DataFrame(columns=self.kaime_columns)
        # umatan_base_df = self.get_umatan_target_df(uma1_df, uma2_df)
        bet_df = kaime_df.query(self.umatan_odds_cond).copy()
        if len(bet_df.index) == 0:
            return pd.DataFrame(columns=self.kaime_columns)
        else:
            bet_df.loc[:, "目１"] =bet_df["UMABAN"].apply(lambda x: x[0])
            bet_df.loc[:, "目２"] =bet_df["UMABAN"].apply(lambda x: x[1])
            bet_df.loc[:, "目３"] =""
            bet_df.loc[:, "券種"] = '5'
            target_bet_df = self._get_target_bet_df(bet_df)
            return target_bet_df

    def _get_wide_bet_df(self, kaime_df):
        if len(kaime_df.index) == 0:
            print("_get_wide_bet_df -- no data--")
            return pd.DataFrame(columns=self.kaime_columns)
        #wide_base_df = self.get_wide_target_df(uma1_df, uma2_df)
        bet_df = kaime_df.query(self.wide_odds_cond).copy()
        if len(bet_df.index) == 0:
            return pd.DataFrame(columns=self.kaime_columns)
        else:
            bet_df.loc[:, "目１"] =bet_df["UMABAN"].apply(lambda x: x[0])
            bet_df.loc[:, "目２"] =bet_df["UMABAN"].apply(lambda x: x[1])
            bet_df.loc[:, "目３"] =""
            bet_df.loc[:, "券種"] = '4'
            target_bet_df = self._get_target_bet_df(bet_df)
            return target_bet_df

    def _get_sanrenpuku_bet_df(self, kaime_df):
        if len(kaime_df.index) == 0:
            print("_get_sanrenpuku_bet_df -- no data--")
            return pd.DataFrame(columns=self.kaime_columns)
        #sanrenpuku_base_df = self.get_sanrenpuku_target_df(uma1_df, uma2_df, uma3_df)
        bet_df = kaime_df.query(self.sanrenpuku_odds_cond).copy()
        if len(bet_df.index) == 0:
            return pd.DataFrame(columns=self.kaime_columns)
        else:
            bet_df.loc[:, "目１"] =bet_df["UMABAN"].apply(lambda x: x[0])
            bet_df.loc[:, "目２"] =bet_df["UMABAN"].apply(lambda x: x[1])
            bet_df.loc[:, "目３"] =bet_df["UMABAN"].apply(lambda x: x[2])
            bet_df.loc[:, "券種"] = '6'
            target_bet_df = self._get_target_bet_df(bet_df)
            return target_bet_df

    def get_tansho_target_df(self, uma1_df):
        add_uma1_df = pd.merge(uma1_df, self.res_raceuma_df, on=["RACE_KEY", "UMABAN"])
        odds_df = self.ext.get_odds_df("単勝")
        odds_df = odds_df[["RACE_KEY", "UMABAN", "単勝オッズ"]]
        target_df = pd.merge(add_uma1_df, odds_df, on=["RACE_KEY", "UMABAN"])
        return target_df

    def get_fukusho_target_df(self, uma1_df):
        add_uma1_df = pd.merge(uma1_df, self.res_raceuma_df, on=["RACE_KEY", "UMABAN"])
        odds_df = self.ext.get_odds_df("複勝")
        odds_df = odds_df[["RACE_KEY", "UMABAN", "複勝オッズ"]]
        target_df = pd.merge(add_uma1_df, odds_df, on=["RACE_KEY", "UMABAN"])
        return target_df

    def get_umaren_target_df(self, uma1_df, uma2_df):
        add_uma1_df = pd.merge(uma1_df, self.res_raceuma_df, on=["RACE_KEY", "UMABAN"]).add_suffix("_1").rename(columns={"RACE_KEY_1":"RACE_KEY"})
        add_uma2_df = pd.merge(uma2_df, self.res_raceuma_df, on=["RACE_KEY", "UMABAN"]).add_suffix("_2").rename(columns={"RACE_KEY_2":"RACE_KEY"})
        odds_df = self.ext.get_odds_df("馬連")
        base_uma1_df = pd.merge(uma1_df[["RACE_KEY", "UMABAN"]], odds_df, on=["RACE_KEY", "UMABAN"]).set_index(["RACE_KEY", "UMABAN"])
        umaren_uma1_df = base_uma1_df[['馬連オッズ０１', '馬連オッズ０２', '馬連オッズ０３', '馬連オッズ０４', '馬連オッズ０５', '馬連オッズ０６', '馬連オッズ０７', '馬連オッズ０８', '馬連オッズ０９',
                                       '馬連オッズ１０', '馬連オッズ１１', '馬連オッズ１２', '馬連オッズ１３', '馬連オッズ１４', '馬連オッズ１５', '馬連オッズ１６', '馬連オッズ１７', '馬連オッズ１８']].copy()
        umaren_uma1_df.columns = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12", "13", "14", "15", "16", "17", "18"]
        umaren_uma1_df = umaren_uma1_df.stack().reset_index()
        umaren_uma1_df.columns = ["RACE_KEY", "UMABAN_1", "UMABAN_2", "オッズ"]
        target_df = pd.merge(umaren_uma1_df, self.race_df, on="RACE_KEY")
        target_df = pd.merge(target_df, add_uma1_df, on=["RACE_KEY", "UMABAN_1"])
        target_df = pd.merge(target_df, add_uma2_df, on=["RACE_KEY", "UMABAN_2"])
        target_df = target_df.query("UMABAN_1 != UMABAN_2")
        target_df = target_df.drop_duplicates(subset=["RACE_KEY", "UMABAN_1", "UMABAN_2"])
        return target_df

    def get_umatan_target_df(self, uma1_df, uma2_df):
        add_uma1_df = pd.merge(uma1_df, self.res_raceuma_df, on=["RACE_KEY", "UMABAN"]).add_suffix("_1").rename(
            columns={"RACE_KEY_1": "RACE_KEY"})
        add_uma2_df = pd.merge(uma2_df, self.res_raceuma_df, on=["RACE_KEY", "UMABAN"]).add_suffix("_2").rename(
            columns={"RACE_KEY_2": "RACE_KEY"})
        base_df = pd.merge(add_uma1_df, add_uma2_df, on="RACE_KEY")
        base_df = base_df.query("UMABAN_1 != UMABAN_2")
        odds_df = self.ext.get_odds_df("馬単")
        target_df = pd.merge(base_df, odds_df, on=["RACE_KEY", "UMABAN_1", "UMABAN_2"])
        target_df = pd.merge(target_df, self.race_df, on=["RACE_KEY", "target_date"])
        target_df = target_df.rename(columns={"馬単オッズ": "オッズ"})
        return target_df

    def get_wide_target_df(self, uma1_df, uma2_df):
        add_uma1_df = pd.merge(uma1_df, self.res_raceuma_df, on=["RACE_KEY", "UMABAN"]).add_suffix("1").rename(
            columns={"RACE_KEY1": "RACE_KEY"})
        add_uma2_df = pd.merge(uma2_df, self.res_raceuma_df, on=["RACE_KEY", "UMABAN"]).add_suffix("2").rename(
            columns={"RACE_KEY2": "RACE_KEY"})
        base_df = pd.merge(add_uma1_df, add_uma2_df, on="RACE_KEY")
        base_df = base_df.query("UMABAN1 != UMABAN2")
        base_df.loc[:, "UMABAN_bet"] = base_df.apply(lambda x: sorted([x["UMABAN1"], x["UMABAN2"]]), axis=1)
        base_df.loc[:, "UMABAN_1"] = base_df["UMABAN_bet"].apply(lambda x: x[0])
        base_df.loc[:, "UMABAN_2"] = base_df["UMABAN_bet"].apply(lambda x: x[1])
        odds_df = self.ext.get_odds_df("ワイド")
        target_df = pd.merge(base_df, odds_df, on=["RACE_KEY", "UMABAN_1", "UMABAN_2"])
        target_df = pd.merge(target_df, self.race_df, on=["RACE_KEY", "target_date"])
        target_df = target_df.drop_duplicates(subset=["RACE_KEY", "UMABAN_1", "UMABAN_2"])
        target_df = target_df.rename(columns={"ワイドオッズ": "オッズ"})
        return target_df

    def get_sanrenpuku_target_df(self, uma1_df, uma2_df, uma3_df):
        add_uma1_df = pd.merge(uma1_df, self.res_raceuma_df, on=["RACE_KEY", "UMABAN"]).add_suffix("1").rename(columns={"RACE_KEY1":"RACE_KEY"})
        add_uma2_df = pd.merge(uma2_df, self.res_raceuma_df, on=["RACE_KEY", "UMABAN"]).add_suffix("2").rename(columns={"RACE_KEY2":"RACE_KEY"})
        add_uma3_df = pd.merge(uma3_df, self.res_raceuma_df, on=["RACE_KEY", "UMABAN"]).add_suffix("3").rename(columns={"RACE_KEY3":"RACE_KEY"})
        base_df = pd.merge(add_uma1_df, add_uma2_df, on="RACE_KEY")
        base_df = pd.merge(base_df, add_uma3_df, on="RACE_KEY")
        base_df = base_df.query("(UMABAN1 != UMABAN2) and (UMABAN2 != UMABAN3) and (UMABAN3 != UMABAN1)")
        base_df.loc[:, "UMABAN_bet"] = base_df.apply(lambda x: sorted([x["UMABAN1"], x["UMABAN2"], x["UMABAN3"]]), axis=1)
        base_df.loc[:, "UMABAN_1"] = base_df["UMABAN_bet"].apply(lambda x: x[0])
        base_df.loc[:, "UMABAN_2"] = base_df["UMABAN_bet"].apply(lambda x: x[1])
        base_df.loc[:, "UMABAN_3"] = base_df["UMABAN_bet"].apply(lambda x: x[2])
        odds_df = self.ext.get_odds_df("三連複")
        target_df = pd.merge(base_df, odds_df, on=["RACE_KEY", "UMABAN_1", "UMABAN_2", "UMABAN_3"])
        target_df = pd.merge(target_df, self.race_df, on=["RACE_KEY", "target_date"])
        target_df = target_df.drop_duplicates(subset=["RACE_KEY", "UMABAN_1", "UMABAN_2", "UMABAN_3"])
        target_df = target_df.rename(columns={"３連複オッズ": "オッズ"})
        return target_df

    def create_result_raceuma_mark_file(self):
        ru_cluster_path = self.target_path + "UmaMark8/"
        for file in self.file_list:
            print(file)
            file_text = ""
            temp_df = self.raceuma_result_df.query(f"file_id == '{file}'")
            nichiji_list = sorted(temp_df["nichiji"].drop_duplicates())
            for nichiji in nichiji_list:
                temp2_df = temp_df.query(f"nichiji == '{nichiji}'")
                race_list = sorted(temp2_df["RACE_KEY"].drop_duplicates())
                for race in race_list:
                    line_text = "      "
                    temp3_df = temp2_df.query(f"RACE_KEY == '{race}'").sort_values("UMABAN")
                    i = 0
                    for idx, val in temp3_df.iterrows():
                        if len(str(val["ru_cluster"])) == 1:
                            line_text += ' ' + str(val["ru_cluster"])
                        else:
                            line_text += '  '
                        i += 1
                    if i != 18:
                        for j in range(i, 18):
                            line_text += "  "
                    file_text += line_text + '\r\n'
            with open(ru_cluster_path + "UM" + file + ".DAT", mode='w', encoding="shift-jis") as f:
                f.write(file_text.replace('\r', ''))

    def create_result_race_comment_file(self):
        for file in self.rc_file_list:
            print(file)
            race_comment_df = self.race_result_df.query(f"rc_file_id == '{file}'")[["RACE_KEY", "レースコメント"]].sort_values(
                "RACE_KEY")
            folder_path = self.target_path + "RACE_COM/20" + file[4:6]
            filename = file + ".dat"
            self._export_file(race_comment_df, folder_path, filename, False)
#            race_comment_df.to_csv(self.target_path + "RACE_COM/20" + file[4:6] + "/" + file + ".dat", header=False,
#                                   index=False, encoding="cp932")

    def create_result_raceuma_comment_file(self):
        for file in self.kc_file_list:
            print(file)
            race_comment_df = self.raceuma_result_df.query(f"kc_file_id == '{file}'")[["RACE_KEY", "UMABAN", "レース馬コメント"]]
            race_comment_df.loc[:, "RACE_UMA_KEY"] = race_comment_df["RACE_KEY"] + race_comment_df["UMABAN"]
            race_comment_df = race_comment_df[["RACE_UMA_KEY", "レース馬コメント"]].sort_values("RACE_UMA_KEY")
            folder_path = self.target_path + "KEK_COM/20" + file[4:6]
            filename = file + ".dat"
            self._export_file(race_comment_df, folder_path, filename, False)
#            race_comment_df.to_csv(self.target_path + "KEK_COM/20" + file[4:6] + "/" + file + ".dat", header=False,
#                                   index=False, encoding="cp932")

    def create_pbi_file(self):
        ## race
        race_df = self.ext.get_race_before_table_base()
        race_df.loc[:, "レースNo"] = race_df["RACE_KEY"].str[6:8]
        race_df.loc[:, "種別"] = race_df["種別"].apply(lambda x: mu.convert_shubetsu(x))
        race_df.loc[:, "芝ダ"] = race_df["芝ダ障害コード"].apply(lambda x: mu.convert_shida(x))
        race_df.loc[:, "コース名"] = race_df.apply(lambda x: self._get_course_name(x), axis=1)
        race_df = race_df[["RACE_KEY", "場名", "レースNo", "距離", "芝ダ", "種別", "条件", "target_date", "コース名",
                           "発走時間", "レース名９文字", "WIN5フラグ"]].copy()
        race_df.loc[:, "年月"] = race_df["target_date"].str[0:6]
        yearmonth_list = race_df["年月"].drop_duplicates()
        for ym in yearmonth_list:
            temp_df = race_df.query(f"年月 == '{ym}'")
            self._export_file(temp_df, self.for_pbi_path + '/race/', ym + ".csv", True)
        ## raceuma
        raceuma_df = self.ext.get_raceuma_before_table_base()
        raceuma_df = raceuma_df[["RACE_KEY", "UMABAN", "基準オッズ", "基準人気順位", "騎手名", "調教師名", "馬名", "血統登録番号",
                                 "IDM", "負担重量", "枠番", "脚質", "距離適性", "芝適性コード", "ダ適性コード", "テン指数",
                                 "ペース指数", "上がり指数", "位置指数", "性別コード", "馬主会コード", "走法", "芝ダ障害フラグ", "距離フラグ",
                                 "クラスフラグ", "転厩フラグ", "去勢フラグ", "乗替フラグ", "放牧先ランク", "厩舎ランク", "調教コースコード",
                                 "追切種類", "追い状態", "調教タイプ", "調教距離", "調教重点", "仕上指数", "調教量評価", "仕上指数変化", "target_date"]].copy()
        raceuma_df = pd.merge(raceuma_df, self.target_mark_df, on=["RACE_KEY", "UMABAN"])
        win5_df = self.win5_df[["RACE_KEY", "UMABAN", "predict_rank"]].copy()
        win5_df.columns = ["RACE_KEY", "UMABAN", "win5_rank"]
        win_df = self.win_df[["RACE_KEY", "UMABAN", "predict_rank"]].copy()
        win_df.columns = ["RACE_KEY", "UMABAN", "win_rank"]
        jiku_df = self.jiku_df[["RACE_KEY", "UMABAN", "predict_rank"]].copy()
        jiku_df.columns = ["RACE_KEY", "UMABAN", "jiku_rank"]
        ana_df = self.ana_df[["RACE_KEY", "UMABAN", "predict_rank"]].copy()
        ana_df.columns = ["RACE_KEY", "UMABAN", "ana_rank"]
        nigeuma_df = self.nigeuma_df[["RACE_KEY", "UMABAN", "predict_rank"]].copy()
        nigeuma_df.columns = ["RACE_KEY", "UMABAN", "nige_rank"]
        agari_df = self.agari_df[["RACE_KEY", "UMABAN", "predict_rank"]].copy()
        agari_df.columns = ["RACE_KEY", "UMABAN", "agari_rank"]
        ten_df = self.ten_df[["RACE_KEY", "UMABAN", "predict_rank"]].copy()
        ten_df.columns = ["RACE_KEY", "UMABAN", "ten_rank"]
        total_df = self.total_df[["RACE_KEY", "UMABAN", "predict_rank"]].copy()
        total_df.columns = ["RACE_KEY", "UMABAN", "total_rank"]
        point_df = self.point_df[["RACE_KEY", "UMABAN", "predict_rank"]].copy()
        point_df.columns = ["RACE_KEY", "UMABAN", "point_rank"]
        raceuma_df = pd.merge(raceuma_df, win5_df, on=["RACE_KEY", "UMABAN"], how='left')
        raceuma_df = pd.merge(raceuma_df, win_df, on=["RACE_KEY", "UMABAN"])
        raceuma_df = pd.merge(raceuma_df, jiku_df, on=["RACE_KEY", "UMABAN"])
        raceuma_df = pd.merge(raceuma_df, ana_df, on=["RACE_KEY", "UMABAN"])
        raceuma_df = pd.merge(raceuma_df, nigeuma_df, on=["RACE_KEY", "UMABAN"])
        raceuma_df = pd.merge(raceuma_df, agari_df, on=["RACE_KEY", "UMABAN"])
        raceuma_df = pd.merge(raceuma_df, ten_df, on=["RACE_KEY", "UMABAN"])
        raceuma_df = pd.merge(raceuma_df, total_df, on=["RACE_KEY", "UMABAN"])
        raceuma_df = pd.merge(raceuma_df, point_df, on=["RACE_KEY", "UMABAN"])
        raceuma_df.loc[:, "RACE_UMA_ID"] = raceuma_df["RACE_KEY"].str.cat(raceuma_df["UMABAN"])
        raceuma_df.loc[:, "年月"] = raceuma_df["target_date"].str[0:6]
        yearmonth_list = raceuma_df["年月"].drop_duplicates()
        for ym in yearmonth_list:
            temp_df = raceuma_df.query(f"年月 == '{ym}'")
            self._export_file(temp_df, self.for_pbi_path + '/raceuma/', ym + ".csv", True)
        ## bet
        bet_df = pd.merge(self.target_bet_df, self.race_base_df[["NENGAPPI", "RACE_ID"]], on="RACE_ID")
        bet_df.loc[:, "年月"] = bet_df["NENGAPPI"].str[0:6]
        yearmonth_list = bet_df["年月"].drop_duplicates()
        for ym in yearmonth_list:
            temp_df = bet_df.query(f"年月 == '{ym}'")
            self._export_file(temp_df, self.for_pbi_path + '/bet/', ym + ".csv", True)


    def create_pbi_result_file(self):
        ## race_result
        race_result_df = self.ext.get_race_table_base()
        race_result_df.loc[:, "種別"] = race_result_df["種別"].apply(lambda x: mu.convert_shubetsu(x))
        race_result_df.loc[:, "芝ダ"] = race_result_df["芝ダ障害コード"].apply(lambda x: mu.convert_shida(x))
        race_result_df.loc[:, "コース名"] = race_result_df.apply(lambda x: self._get_course_name(x), axis=1)
        race_result_df = race_result_df[["RACE_KEY", "距離", "芝ダ", "右左", "内外", "種別", "条件", "グレード", "レース名９文字", "コース名",
                                         "WIN5フラグ", "場名", "芝馬場状態コード", "ダ馬場状態コード", "芝種類", "草丈", "転圧", "凍結防止剤",
                                         "中間降水量", "レースコメント", "ハロンタイム０１", "ハロンタイム０２", "ハロンタイム０３", "ハロンタイム０４",
                                         "ハロンタイム０５", "ハロンタイム０６", "ハロンタイム０７", "ハロンタイム０８", "ハロンタイム０９", "ハロンタイム１０",
                                         "ハロンタイム１１", "ハロンタイム１２", "ハロンタイム１３", "ハロンタイム１４", "ハロンタイム１５", "ハロンタイム１６",
                                         "ハロンタイム１７", "ハロンタイム１８", "ラスト５ハロン", "ラスト４ハロン", "ラスト３ハロン", "ラスト２ハロン",
                                         "ラスト１ハロン", "ラップ差４ハロン", "ラップ差３ハロン", "ラップ差２ハロン", "ラップ差１ハロン", "RAP_TYPE",
                                         "TRACK_BIAS_ZENGO", "TRACK_BIAS_UCHISOTO", "target_date"]].copy()
        race_result_df = pd.merge(race_result_df, self.race_base_df[["RACE_KEY", "RACE_ID"]], on="RACE_KEY")
        race_result_df.loc[:, "年月"] = race_result_df["target_date"].str[0:6]
        yearmonth_list = race_result_df["年月"].drop_duplicates()
        for ym in yearmonth_list:
            temp_df = race_result_df.query(f"年月 == '{ym}'")
            self._export_file(temp_df, self.for_pbi_path + '/race_result/', ym + ".csv", True)
        ## raceuma_result
        raceuma_result_df = self.ext.get_raceuma_table_base()
        raceuma_result_df = raceuma_result_df[["RACE_KEY", "UMABAN", "基準オッズ", "基準人気順位", "騎手名", "調教師名", "馬名", "血統登録番号", "着順",
                                               "IDM", "負担重量", "枠番", "脚質", "距離適性", "芝適性コード", "ダ適性コード", "テン指数",
                                               "ペース指数", "上がり指数", "位置指数", "性別コード", "馬主会コード", "走法", "芝ダ障害フラグ", "距離フラグ",
                                               "クラスフラグ", "転厩フラグ", "去勢フラグ", "乗替フラグ", "放牧先ランク", "厩舎ランク", "調教コースコード",
                                               "追切種類", "追い状態", "調教タイプ", "調教距離", "調教重点", "仕上指数", "調教量評価", "仕上指数変化",
                                               "タイム", "確定単勝オッズ", "確定単勝人気順位", "ＩＤＭ結果", "テン指数結果", "上がり指数結果", "ペース指数結果",
                                               "前３Ｆタイム", "後３Ｆタイム", "コーナー順位１", "コーナー順位２", "コーナー順位３", "コーナー順位４", "馬体重",
                                               "レース脚質", "単勝", "複勝", "レース馬コメント", "馬具(その他)コメント", "パドックコメント", "脚元コメント", "target_date"]].copy()
        raceuma_result_df = pd.merge(raceuma_result_df, self.race_base_df[["RACE_KEY", "RACE_ID"]], on="RACE_KEY")
        raceuma_result_df.loc[:, "年月"] = raceuma_result_df["target_date"].str[0:6]
        raceuma_result_df.loc[:, "RACE_UMA_ID"] = raceuma_result_df["RACE_KEY"].str.cat(raceuma_result_df["UMABAN"])
        yearmonth_list = raceuma_result_df["年月"].drop_duplicates()
        for ym in yearmonth_list:
            temp_df = raceuma_result_df.query(f"年月 == '{ym}'")
            self._export_file(temp_df, self.for_pbi_path + '/raceuma_result/', ym + ".csv", True)
        ## haraimodoshi
        haraimodoshi_df = self.ext.get_haraimodoshi_table_base()
        tansho_df = self.ext.get_tansho_df(haraimodoshi_df)
        tansho_df.loc[:, "券種"] = "単勝"
        fukusho_df = self.ext.get_fukusho_df(haraimodoshi_df)
        fukusho_df.loc[:, "券種"] = "複勝"
        umaren_df = self.ext.get_umaren_df(haraimodoshi_df)
        umaren_df.loc[:, "券種"] = "馬連"
        umatan_df = self.ext.get_umatan_df(haraimodoshi_df)
        umatan_df.loc[:, "券種"] = "馬単"
        wide_df = self.ext.get_wide_df(haraimodoshi_df)
        wide_df.loc[:, "券種"] = "ワイド"
        sanrenpuku_df = self.ext.get_sanrenpuku_df(haraimodoshi_df)
        sanrenpuku_df.loc[:, "券種"] = "三連複"
        return_df = pd.concat([tansho_df, fukusho_df, umaren_df, umatan_df, wide_df, sanrenpuku_df])
        return_df = pd.merge(return_df, self.race_base_df[["NENGAPPI", "RACE_KEY"]], on="RACE_KEY")
        return_df.loc[:, "年月"] = return_df["NENGAPPI"].str[0:6]
        yearmonth_list = return_df["年月"].drop_duplicates()
        for ym in yearmonth_list:
            temp_df = return_df.query(f"年月 == '{ym}'")
            self._export_file(temp_df, self.for_pbi_path + '/return/', ym + ".csv", True)

    def _export_file(self, df, folder_path, filename, header):
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        df.to_csv(folder_path + "/" + filename, header=header, index=False, encoding="cp932")

    def _return_mark(self, num):
        if num == 1: return "◎"
        if num == 2: return "○"
        if num == 3: return "▲"
        if num == 4: return "△"
        if num == 5:
            return "×"
        else:
            return "  "

    def _get_target_bet_df(self, bet_df):
        bet_df = pd.merge(bet_df, self.race_base_df, on="RACE_KEY")
        bet_df.loc[:, "変換フラグ"] = 0
        bet_df.loc[:, "購入金額"] = 100
        bet_df.loc[:, "的中時の配当"] = 0
        bet_df.loc[:, "エリア"] = "F"
        bet_df.loc[:, "マーク"] = ""
        bet_df = bet_df[["RACE_ID", "変換フラグ", "券種", "目１", "目２", "目３", "購入金額", "オッズ", "的中時の配当", "エリア", "マーク"]].copy()
        return bet_df

    def _get_course_name(self, sr):
        soto = "外" if sr["内外"] == "2" else ""
        return sr["場名"] + sr["芝ダ"] + str(sr["距離"]) +"m" + soto

    @classmethod
    def post_slack_text(cls, post_text):
        slack = slackweb.Slack(url=cls.slack_operation_url)
        #slack.notify(text=post_text)



