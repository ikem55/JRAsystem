from modules.extract import Extract
from modules.transform import Transform as Tf
import modules.util as mu
import my_config as mc

from datetime import datetime as dt
from datetime import timedelta
import pandas as pd

class Load(object):
    """
    データロードに関する共通処理を定義する。
    race,raceuma,prev_raceといった塊ごとのデータを作成する。learning_df等の最終データの作成はsk_proc側に任せる

    """
    dict_folder = ""
    """ 辞書フォルダのパス """
    mock_flag = False
    """ mockデータを利用する場合はTrueにする """
    race_df = ""
    raceuma_df = ""
    horse_df = ""
    prev_raceuma_df = ""
    result_df = ""

    def __init__(self, version_str, start_date, end_date, mock_flag, test_flag):
        self.start_date = start_date
        self.end_date = end_date
        self.mock_flag = mock_flag
        self.dict_path = mc.return_base_path(test_flag)
        self._set_folder_path(version_str)
        self.ext = self._get_extract_object(start_date, end_date, mock_flag)
        self.tf = self._get_transform_object(start_date, end_date)

    def _set_folder_path(self, version_str):
        self.dict_folder = self.dict_path + 'dict/' + version_str + '/'
        print("self.dict_folder:", self.dict_folder)

    def _get_extract_object(self, start_date, end_date, mock_flag):
        """ 利用するExtクラスを指定する """
        ext = Extract(start_date, end_date, mock_flag)
        return ext

    def _get_transform_object(self, start_date, end_date):
        """ 利用するTransformクラスを指定する """
        tf = Tf(start_date, end_date)
        return tf

    def set_race_df(self):
        """  race_dfを作成するための処理。race_dfに処理がされたデータをセットする """
        race_base_df = self.ext.get_race_before_table_base()
        self.race_df = self._proc_race_df(race_base_df)
        print("set_race_df: race_df", self.race_df.shape)

    def _proc_race_df(self, race_df):
        """ race_dfの前処理、encode -> normalize -> standardize -> feature_create -> drop_columnsの順で処理 """
        race_df = self.tf.cluster_course_df(race_df, self.dict_path) #参照フォルダの位置の問題がある(dict_path/dict_foloder)
        race_df = self.tf.encode_race_df(race_df)
        race_df = self.tf.normalize_race_df(race_df)
        race_df = self.tf.standardize_race_df(race_df)
        race_df = self.tf.create_feature_race_df(race_df)
        race_df = self.tf.choose_race_df_columns(race_df)
        return race_df

    def set_raceuma_df(self):
        """ raceuma_dfを作成するための処理。raceuma_dfに処理がされたデータをセットする """
        raceuma_base_df = self.ext.get_raceuma_before_table_base()
        self.raceuma_df = self._proc_raceuma_df(raceuma_base_df)
        print("set_raceuma_df: raceuma_df", self.raceuma_df.shape)

    def _proc_raceuma_df(self, raceuma_df):
        raceuma_df = self.tf.encode_raceuma_df(raceuma_df, self.dict_folder)
        raceuma_df = self.tf.normalize_raceuma_df(raceuma_df)
        raceuma_df = self.tf.standardize_raceuma_df(raceuma_df)
        raceuma_df = self.tf.create_feature_raceuma_df(raceuma_df)
        raceuma_df = self.tf.choose_raceuma_df_columns(raceuma_df)
        return raceuma_df.copy()

    def set_horse_df(self):
        """  horse_dfを作成するための処理。horse_dfに処理がされたデータをセットする """
        horse_base_df = self.ext.get_horse_table_base()
        self.horse_df = self._proc_horse_df(horse_base_df)
        print("set_horse_df: horse_df", self.horse_df.shape)

    def _proc_horse_df(self, horse_df):
        horse_df = self.tf.encode_horse_df(horse_df, self.dict_folder)
        horse_df = self.tf.normalize_horse_df(horse_df)
        horse_df = self.tf.standardize_horse_df(horse_df)
        horse_df = self.tf.create_feature_horse_df(horse_df)
        horse_df = self.tf.choose_horse_df_column(horse_df)
        return horse_df.copy()

    def set_prev_df(self):
        """  prev_dfを作成するための処理。prev1_raceuma_df,prev2_raceuma_dfに処理がされたデータをセットする。過去２走のデータと過去走を集計したデータをセットする  """
        raceuma_5_prev_df = self.ext.get_raceuma_zenso_table_base()
        self._proc_prev_df(raceuma_5_prev_df)
        print("set_prev_df: raceuma_5_prev_df", self.prev_raceuma_df.shape)
        print("set_prev_df: prev_feature_raceuma_df", self.prev_feature_raceuma_df.shape)

    def _proc_prev_df(self, raceuma_5_prev_df):
        raceuma_5_prev_df = self.tf.cluster_course_df(raceuma_5_prev_df, self.dict_path)
        raceuma_5_prev_df = self.tf.cluster_raceuma_result_df(raceuma_5_prev_df, self.dict_path)
        raceuma_5_prev_df = self.tf.factory_analyze_race_result_df(raceuma_5_prev_df, self.dict_path)
        prev5_raceuma_df = self._get_prev_df(5, raceuma_5_prev_df)
        prev5_raceuma_df.rename(columns=lambda x: x + "_5", inplace=True)
        prev5_raceuma_df.rename(columns={"RACE_KEY_5": "RACE_KEY", "UMABAN_5": "UMABAN", "target_date_5": "target_date"}, inplace=True)
        prev4_raceuma_df = self._get_prev_df(4, raceuma_5_prev_df)
        prev4_raceuma_df.rename(columns=lambda x: x + "_4", inplace=True)
        prev4_raceuma_df.rename(columns={"RACE_KEY_4": "RACE_KEY", "UMABAN_4": "UMABAN", "target_date_4": "target_date"}, inplace=True)
        prev3_raceuma_df = self._get_prev_df(3, raceuma_5_prev_df)
        prev3_raceuma_df.rename(columns=lambda x: x + "_3", inplace=True)
        prev3_raceuma_df.rename(columns={"RACE_KEY_3": "RACE_KEY", "UMABAN_3": "UMABAN", "target_date_3": "target_date"}, inplace=True)
        prev2_raceuma_df = self._get_prev_df(2, raceuma_5_prev_df)
        prev2_raceuma_df.rename(columns=lambda x: x + "_2", inplace=True)
        prev2_raceuma_df.rename(columns={"RACE_KEY_2": "RACE_KEY", "UMABAN_2": "UMABAN", "target_date_2": "target_date"}, inplace=True)
        prev1_raceuma_df = self._get_prev_df(1, raceuma_5_prev_df)
        prev1_raceuma_df.rename(columns=lambda x: x + "_1", inplace=True)
        prev1_raceuma_df.rename(columns={"RACE_KEY_1": "RACE_KEY", "UMABAN_1": "UMABAN", "target_date_1": "target_date"}, inplace=True)
        prev_raceuma_df = pd.merge(prev1_raceuma_df, prev2_raceuma_df, on=["RACE_KEY", "UMABAN", "target_date"], how="outer")
        prev_raceuma_df = pd.merge(prev_raceuma_df, prev3_raceuma_df, on=["RACE_KEY", "UMABAN", "target_date"], how="outer")
        prev_raceuma_df = pd.merge(prev_raceuma_df, prev4_raceuma_df, on=["RACE_KEY", "UMABAN", "target_date"], how="outer")
        prev_raceuma_df = pd.merge(prev_raceuma_df, prev5_raceuma_df, on=["RACE_KEY", "UMABAN", "target_date"], how="outer")
        self.prev_raceuma_df = prev_raceuma_df
        self.prev_feature_raceuma_df = self._get_prev_feature_df(raceuma_5_prev_df)

    def _get_prev_df(self, num, raceuma_5_prev_df):
        prev_race_key = f"ZENSO{num}_KYOSO_RESULT"
        raceuma_base_df = self.raceuma_df[["RACE_KEY", "UMABAN", "target_date", prev_race_key]].rename(columns={prev_race_key: "KYOSO_RESULT_KEY"})
        this_raceuma_df = pd.merge(raceuma_base_df, raceuma_5_prev_df.drop(["RACE_KEY", "UMABAN"], axis=1), on=["KYOSO_RESULT_KEY", "target_date"])
        this_raceuma_df = self.tf.encode_raceuma_result_df(this_raceuma_df, self.dict_folder)
        this_raceuma_df = self.tf.normalize_raceuma_result_df(this_raceuma_df)
        this_raceuma_df = self.tf.create_feature_raceuma_result_df(this_raceuma_df)
        this_raceuma_df = self.tf.choose_raceuma_result_df_columns(this_raceuma_df)
        print("_proc_raceuma_result_df: raceuma_df", this_raceuma_df.shape)
        return this_raceuma_df


    def _get_prev_feature_df(self, raceuma_5_prev_df):
        max_columns = ['血統登録番号', 'target_date', 'fa_1', 'fa_2', 'fa_3', 'fa_4', 'fa_5', 'ＩＤＭ結果', 'テン指数結果', '上がり指数結果', 'ペース指数結果']
        min_columns = ['血統登録番号', 'target_date', 'fa_4', 'テン指数結果順位', '上がり指数結果順位', 'ペース指数結果順位']
        max_score_df = raceuma_5_prev_df[max_columns].groupby(['血統登録番号', 'target_date']).max().add_prefix("max_").reset_index()
        min_score_df = raceuma_5_prev_df[min_columns].groupby(['血統登録番号', 'target_date']).min().add_prefix("min_").reset_index()
        feature_df = pd.merge(max_score_df, min_score_df, on=["血統登録番号", "target_date"])
        race_df = self.race_df[["RACE_KEY", "course_cluster"]].copy()
        raceuma_df = self.raceuma_df[["RACE_KEY", "UMABAN", "血統登録番号", "target_date"]].copy()
        raceuma_df = pd.merge(race_df, raceuma_df, on="RACE_KEY")
        filtered_df = pd.merge(raceuma_df, raceuma_5_prev_df.drop(["RACE_KEY", "UMABAN"], axis=1), on=["血統登録番号", "target_date", "course_cluster"])[["RACE_KEY", "UMABAN", "ru_cluster"]]
        filtered_df_c1 = filtered_df.query("ru_cluster == '1'").groupby(["RACE_KEY", "UMABAN"]).count().reset_index()
        filtered_df_c1.columns = ["RACE_KEY", "UMABAN", "c1_cnt"]
        filtered_df_c2 = filtered_df.query("ru_cluster == '2'").groupby(["RACE_KEY", "UMABAN"]).count().reset_index()
        filtered_df_c2.columns = ["RACE_KEY", "UMABAN", "c2_cnt"]
        filtered_df_c3 = filtered_df.query("ru_cluster == '3'").groupby(["RACE_KEY", "UMABAN"]).count().reset_index()
        filtered_df_c3.columns = ["RACE_KEY", "UMABAN", "c3_cnt"]
        filtered_df_c4 = filtered_df.query("ru_cluster == '4'").groupby(["RACE_KEY", "UMABAN"]).count().reset_index()
        filtered_df_c4.columns = ["RACE_KEY", "UMABAN", "c4_cnt"]
        filtered_df_c7 = filtered_df.query("ru_cluster == '7'").groupby(["RACE_KEY", "UMABAN"]).count().reset_index()
        filtered_df_c7.columns = ["RACE_KEY", "UMABAN", "c7_cnt"]
        raceuma_df = pd.merge(raceuma_df, filtered_df_c1, on=["RACE_KEY", "UMABAN"], how="left")
        raceuma_df = pd.merge(raceuma_df, filtered_df_c2, on=["RACE_KEY", "UMABAN"], how="left")
        raceuma_df = pd.merge(raceuma_df, filtered_df_c3, on=["RACE_KEY", "UMABAN"], how="left")
        raceuma_df = pd.merge(raceuma_df, filtered_df_c4, on=["RACE_KEY", "UMABAN"], how="left")
        raceuma_df = pd.merge(raceuma_df, filtered_df_c7, on=["RACE_KEY", "UMABAN"], how="left")
        raceuma_df = raceuma_df.fillna(0)
        raceuma_df = pd.merge(raceuma_df, feature_df, on=["血統登録番号", "target_date"], how="left").drop(["course_cluster", "血統登録番号", "target_date"], axis=1)
        return raceuma_df


    def set_result_df(self):
        """ result_dfを作成するための処理。result_dfに処理がされたデータをセットする """
        result_race_df = self.ext.get_race_table_base()
        result_raceuma_df = self.ext.get_raceuma_table_base()
        result_haraimodoshi_df = self.ext.get_haraimodoshi_table_base()
        self.result_df = self._proc_result_df(result_race_df, result_raceuma_df, result_haraimodoshi_df)

    def _proc_result_df(self, result_race_df, result_raceuma_df, result_haraimodoshi_df):
        result_race_df = result_race_df[["RACE_KEY", "NENGAPPI", "RAP_TYPE", "TRACK_BIAS_UCHISOTO", "TRACK_BIAS_ZENGO", "target_date"]].copy()
        result_raceuma_df = result_raceuma_df[["RACE_KEY" ,"UMABAN", "着順", "複勝", "レース脚質", "レースペース流れ", "上がり指数結果順位", "テン指数結果順位"]].copy()
        umaren_df = result_haraimodoshi_df[["RACE_KEY", "馬連払戻１", "馬単払戻１", "３連複払戻１"]].copy()
        result_df = pd.merge(result_race_df, result_raceuma_df, on="RACE_KEY")
        result_df = pd.merge(result_df, umaren_df, on="RACE_KEY")
        return result_df[["RACE_KEY", "UMABAN", "target_date", "NENGAPPI", "着順", "複勝", "馬連払戻１", "馬単払戻１", "３連複払戻１", "RAP_TYPE",
                          "TRACK_BIAS_UCHISOTO", "TRACK_BIAS_ZENGO", "レースペース流れ", "レース脚質", "上がり指数結果順位", "テン指数結果順位"]].copy()
