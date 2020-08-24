from modules.load import Load
from modules.sk_model import SkModel
import modules.util as mu
import my_config as mc

import pandas as pd
import os
import numpy as np
import pickle
import category_encoders as ce

from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import BorderlineSMOTE

import optuna.integration.lightgbm as lgb
import lightgbm as lgb_original

class SkProc(object):
    """
    機械学習を行うのに必要なプロセスを定義する。learning/predictデータの作成プロセスや
    """
    learning_df = ""
    categ_columns = ""
    target_enc_columns = ""
    target_flag = ""
    x_df = ""
    y_df = ""
    X_train = ""
    X_test = ""
    y_train = ""
    y_test = ""
    label_list = ""

    def __init__(self, version_str, start_date, end_date, model_name, mock_flag, test_flag):
        self.start_date = start_date
        self.end_date = end_date
        self.version_str = version_str # type:haito, win etc.
        self.model_name = model_name #race or raceuma
        self._set_folder_path(version_str, model_name, test_flag)
        self._set_index_list(model_name)
        self._set_obj_column_list(version_str)
        mu.create_folder(self.model_folder)
        self.ld = self._get_load_object(version_str, start_date, end_date, mock_flag, test_flag)
        self.skmodel = self._get_skmodel_object(model_name, version_str, start_date, end_date, test_flag)
        self.test_flag = test_flag

    def _set_folder_path(self, version_str, model_name, test_flag):
        self.dict_path = mc.return_base_path(test_flag)
        self.dict_folder = self.dict_path + 'dict/' + version_str + '/'
        self.model_folder = self.dict_path + 'model/' + version_str + '/'
        self.pred_folder =self.dict_path + 'pred/' + version_str + '/'
        mu.create_folder(self.dict_folder)
        mu.create_folder(self.model_folder)
        mu.create_folder(self.pred_folder)

    def _set_index_list(self, model_name):
        if model_name == "race":
            self.index_list = ["RACE_KEY", "NENGAPPI"]
        elif model_name == "raceuma":
            self.index_list = ["RACE_KEY", "UMABAN", "NENGAPPI"]

    def _set_obj_column_list(self, version_str):
        if version_str == "win":
            self.obj_column_list = ['WIN_FLAG', 'JIKU_FLAG', 'ANA_FLAG']
            self.lgbm_params = {
                'WIN_FLAG': {'objective': 'binary'},
                'JIKU_FLAG': {'objective': 'binary'},
                'ANA_FLAG': {'objective': 'binary'},
            }
        elif version_str == "win5":
            self.obj_column_list = ['WIN5_FLAG']
            self.lgbm_params = {
                'WIN5_FLAG': {'objective': 'binary'},
            }
        elif version_str == "haito":
            self.obj_column_list = ["UMAREN_ARE", "UMATAN_ARE", "SANRENPUKU_ARE"]
            self.lgbm_params = {
                'UMAREN_ARE': {'objective': 'binary'},
                'UMATAN_ARE': {'objective': 'binary'},
                'SANRENPUKU_ARE': {'objective': 'binary'},
            }
        elif version_str == "raptype":
            self.obj_column_list = ['RAP_TYPE', 'TRACK_BIAS_ZENGO', 'TRACK_BIAS_UCHISOTO', 'PRED_PACE']
            self.lgbm_params = {
                'RAP_TYPE': {'objective': 'multiclass', 'metric': 'multi_logloss', 'num_class': 10},
                'TRACK_BIAS_ZENGO': {'objective': 'multiclass', 'metric': 'multi_logloss', 'num_class': 5},
                'TRACK_BIAS_UCHISOTO': {'objective': 'multiclass', 'metric': 'multi_logloss', 'num_class': 5},
                'PRED_PACE': {'objective': 'multiclass', 'metric': 'multi_logloss', 'num_class': 10},
            }
        elif version_str == "nigeuma":
            self.obj_column_list = ['NIGEUMA', 'AGARI_SAISOKU', 'TEN_SAISOKU']
            self.lgbm_params = {
                'NIGEUMA': {'objective': 'binary'},
                'AGARI_SAISOKU': {'objective': 'binary'},
                'TEN_SAISOKU': {'objective': 'binary'},
            }

    def _get_load_object(self, version_str, start_date, end_date, mock_flag, test_flag):
        ld = Load(version_str, start_date, end_date, mock_flag, test_flag)
        return ld

    def _get_skmodel_object(self, model_name, version_str, start_date, end_date, test_flag):
        skmodel = SkModel(model_name, version_str, start_date, end_date, test_flag)
        return skmodel


    def set_learning_df(self):
        base_df = self._get_base_df()
        self.learning_df = base_df
        #result_df = self._get_result_df()
        #self.learning_df = pd.merge(base_df, result_df, on = self.index_list)
        self.categ_columns = self.skmodel.categ_columns
        self.target_enc_columns = self.skmodel.target_enc_columns

    def merge_learning_df(self, learning_df):
        result_df = self._get_result_df()
        learning_df = pd.merge(learning_df, result_df, on = self.index_list)
        return learning_df

    def create_predict_data(self):
        base_df = self._get_base_df()
        self.categ_columns = self.skmodel.categ_columns
        self.target_enc_columns = self.skmodel.target_enc_columns
        return base_df

    def _get_base_df(self):
        self._set_ld_data()
        base_df = self.skmodel.get_merge_df(self.ld.race_df, self.ld.raceuma_df, self.ld.horse_df, self.ld.prev_raceuma_df, self.ld.prev_feature_raceuma_df)
        base_df = self.skmodel.get_create_feature_raceuma_df(base_df)
        base_df = self.skmodel.get_droped_columns_raceuma_df(base_df)
        # 特徴生成時にエンコードしたデータだと不都合が発生したのでエンコードを後で実施
        base_df = self.ld.tf.encode_race_df(base_df)
        base_df = self.ld.tf.encode_raceuma_df(base_df, self.dict_folder)
        base_df = self.ld.tf.encode_horse_df(base_df, self.dict_folder)
        base_df = self.skmodel.get_label_encoding_raceuma_df(base_df, self.index_list)
        if self.model_name == 'race':
            base_df = self._get_base_df_for_race(base_df)
        return base_df

    def _get_base_df_for_race(self, base_df):
        base_df = self.skmodel.get_create_feature_race_df(base_df, self.ld.race_df)
        base_df = self.skmodel.get_droped_columns_race_df(base_df)
        base_df = self.skmodel.get_label_encoding_race_df(base_df, self.index_list)
        return base_df

    def _set_ld_data(self):
        """  Loadオブジェクトにデータをセットする処理をまとめたもの。Race,Raceuma,Horse,Prevのデータフレームをセットする

        :param object ld: データロードオブジェクト(ex.LocalBaozLoad)
        """
        self.ld.set_race_df()  # データ取得
        self.ld.set_raceuma_df()
        self.ld.set_horse_df()
        self.ld.set_prev_df()

    def _get_result_df(self):
        self.ld.set_result_df()
        result_df = self.skmodel.get_target_variable_df(self.ld.result_df, self.version_str)
        return result_df

    def create_featrue_select_data(self, learning_df):
        self.learning_df = learning_df
        for target_flag in self.obj_column_list:
            print(target_flag)
            self._set_target_flag(target_flag)
            self._create_feature_select_data(target_flag)

    def _set_target_flag(self, target_flag):
        """ 目的変数となるターゲットフラグの値をセットする

        :param str target_flag: ターゲットフラグ名(WIN_FLAG etc.)
        """
        self.target_flag = target_flag

    def _create_feature_select_data(self, target_flag):
        """  指定した説明変数に対しての特徴量作成の処理を行う。TargetEncodingや、Borutaによる特徴選択（DeepLearning以外）を行う

        :param str target_flag:
        """
        print("create_feature_select_data")
        self._set_learning_data(self.learning_df, target_flag)
        self._divide_learning_data()
        self._create_learning_target_encoding()

    def _set_learning_data(self, df, target_column):
        """ 与えられたdfからexp_data,obj_dataを作成する。目的変数が複数ある場合は除外する

        :param dataframe df: dataframe
        :param str target_column: 目的変数のカラム名
        """
        df = df.drop(self.index_list, axis=1)
        # 重みづけ用の列を追加
        df_weight = df[target_column].value_counts(normalize=True)
        df.loc[:, "weight"] = df[target_column].apply(lambda x: 1 / (df_weight[x] * 100))
        self.y_df = df[target_column].copy()
        self.x_df = df.drop(self.obj_column_list, axis=1).copy()
        self._set_label_list(self.x_df)


    def _set_label_list(self, df):
        """ label_listの値にわたされたdataframeのデータ型がobjectのカラムのリストをセットする

        :param dataframe df: dataframe
        """
        self.label_list = df.select_dtypes(include=object).columns.tolist()

    def _divide_learning_data(self):
        """ 学習データをtrainとtestに分ける。オブジェクトのx_df,y_dfからX_train,X_test,y_train,y_testに分けてセットする """
        print(self.x_df.shape)
        print(self.y_df.shape)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.x_df, self.y_df, test_size=0.25, random_state=None)

    def _create_learning_target_encoding(self):
        """ TargetEncodeの計算を行い、計算結果のエンコード値をセットする """
        self.X_train = self._set_target_encoding(self.X_train, True, self.y_train).copy()
        self.X_test = self._set_target_encoding(self.X_test, False, self.y_test).copy()

    def _set_target_encoding(self, x_df, fit, y_df):
        """ TargetEncodingの処理をコントロールする処理。label_list（dataframeでデータ型がobjectとなったカラムのリスト）に対してtarget_encodingの処理を行う。
        fitの値がTrueの場合はエンコーダーを作り、Falseの場合は作成されたエンコーダーを適用し、エンコードした値のdataframeを返す。

        :param dataframe x_df: dataframe
        :param bool fit: bool
        :param dataframe y_df: dataframe
        :return: dataframe
        """
        if self.target_enc_columns == "":
            self.target_enc_columns = self.skmodel.target_enc_columns
        if self.version_str != 'raptype':
            target_encoding_columns = list(set(x_df.columns.tolist()) & set(self.target_enc_columns))
            for label in target_encoding_columns:
                #x_df.loc[:, "tr_" + label] = self._target_encoding(x_df[label], label, self.target_flag + '_tr_' + label, fit, y_df)
                x_df["tr_" + label] = self._target_encoding(x_df[label], label, self.target_flag + '_tr_' + label, fit, y_df)
        return x_df

    def _target_encoding(self, sr, label, dict_name, fit, y):
        """ srに対してtarget encodingした結果を返す。fitがTrueの場合はエンコーディングする。ただし、すでに辞書がある場合はそれを使う。

        :param series sr: エンコード対象のSeries
        :param str label: エンコードするラベル名
        :param str dict_name: エンコード辞書名
        :param bool fit: エンコード実施 or 実施済み辞書から変換
        :param series y: 目的変数のSeries
        :return:
        """
        # print("---- target encoding: " + label)
        tr = ce.TargetEncoder(cols=label)
        dict_file = self.dict_folder + '/' + dict_name + '.pkl'
        if fit and not os.path.exists(dict_file):
            tr = tr.fit(sr, y)
            mu.save_dict(tr, dict_name, self.dict_folder)
        else:
            tr = mu.load_dict(dict_name, self.dict_folder)
        sr_tr = tr.transform(sr)
        return sr_tr

    def proc_learning_sk_model(self, df):
        """  説明変数ごとに、指定された場所の学習を行う

        :param dataframe df: dataframe
        :param str basho: str
        """
#        if not df.dropna().empty:
        if len(df.index) >= 30:
            print("proc_learning_sk_model: df", df.shape)
            for target in self.obj_column_list:
                print(target)
                self._learning_sk_model(df, target)
        else:
            print("---- 少数レコードのため学習スキップ -- " + str(len(df.index)))
#        else:
#            print("---- NaNデータが含まれているため学習をスキップ")


    def _learning_sk_model(self, df, target):
        """ 指定された場所・ターゲットに対しての学習処理を行う

        :param dataframe df: dataframe
        :param str target: str
        """
        this_model_name = self.model_name + "_" + target
        if os.path.exists(self.model_folder + this_model_name + '.pickle'):
            print("\r\n -- skip create learning model -- \r\n")
        else:
            self._set_target_flag(target)
            print("learning_sk_model: df", df.shape)
            if df.empty:
                print("--------- alert !!! no data")
            else:
                self._set_learning_data(df, target)
                self._divide_learning_data()
                temp_col = df.dtypes[df.dtypes != "object"].index.tolist()
                print(temp_col)
                if self.y_train.sum() == 0:
                    print("---- wrong data --- skip learning")
                else:
                    self._load_learning_target_encoding()
                    self.X_train = self._change_obj_to_int(self.X_train)
                    self.X_test = self._change_obj_to_int(self.X_test)
                    self.categ_columns = list(set(self.categ_columns) & set(self.X_train.columns.tolist()))
                    imp_features = self._learning_base_race_lgb(this_model_name, target)
                    imp_features.append("weight")
                    # x_dfにTRの値を持っていないのでTR前の値に切り替え
                    #imp_features = [w.replace('tr_','') for w in imp_features]
                    #imp_features = list(set(imp_features))
                    # 抽出した説明変数でもう一度Ｌｅａｒｎｉｎｇを実施
                    self.x_df = self.x_df[imp_features]
                    self.categ_columns = list(set(self.categ_columns) & set(imp_features))
                    self.target_enc_columns = list(set(self.target_enc_columns) & set(imp_features))
                    self._divide_learning_data()
                    self._set_label_list(self.x_df) # 項目削除されているから再度ターゲットエンコーディングの対象リストを更新する
                    self._load_learning_target_encoding()
                    self.X_train = self._change_obj_to_int(self.X_train)
                    self.X_test = self._change_obj_to_int(self.X_test)
                    self._learning_race_lgb(this_model_name, target)

    def _load_learning_target_encoding(self):
        """ TargetEncodeを行う。すでに作成済みのエンコーダーから値をセットする  """
        print("load_learning_target_encoding")
        self.X_train = self._set_target_encoding(self.X_train, False, self.y_train).copy()
        self.X_test = self._set_target_encoding(self.X_test, False, self.y_test).copy()


    def _change_obj_to_int(self, df):
        """ objのデータ項目をint型に変更する """
        label_list = df.select_dtypes(include=object).columns.tolist()
        df[label_list] = df[label_list].astype(float) #NaNがあるとintにできない
        return df

    def _learning_base_race_lgb(self, this_model_name, target):
        """ null importanceにより有効な説明変数を抽出する """
        print("learning_base_race_lgb")
        # テスト用のデータを評価用と検証用に分ける
        X_eval, X_valid, y_eval, y_valid = train_test_split(self.X_test, self.y_test, random_state=42)
        X_eval_weight = X_eval["weight"]
        X_eval = X_eval.drop("weight", axis=1)

        if self.test_flag:
            num_boost_round = 5
            n_rum = 3
            threshold = 5
            ram_imp_num_boost_round = 5
            early_stopping_rounds = 3
        else:
            num_boost_round = 100
            n_rum = 15
            threshold = 30
            ram_imp_num_boost_round = 100
            early_stopping_rounds = 50

        # データセットを生成する
        # カテゴリ変数を指定する　https://upura.hatenablog.com/entry/2019/10/29/184617
        lgb_train = lgb.Dataset(self.X_train.drop("weight", axis=1), self.y_train, categorical_feature=self.categ_columns, weight=self.X_train["weight"])
        lgb_eval = lgb.Dataset(X_eval, y_eval, reference=lgb_train, categorical_feature=self.categ_columns, weight=X_eval_weight)

        # 上記のパラメータでモデルを学習する
        this_param = self.lgbm_params[target]
        model = lgb_original.train(this_param, lgb_train,
                          # モデルの評価用データを渡す
                          valid_sets=lgb_eval,
                          # 最大で 1000 ラウンドまで学習する
                          num_boost_round=num_boost_round,
                          # 10 ラウンド経過しても性能が向上しないときは学習を打ち切る
                          early_stopping_rounds=early_stopping_rounds)

        # 特徴量の重要度を含むデータフレームを作成
        imp_df = pd.DataFrame()
        imp_df["feature"] = X_eval.columns
        imp_df["importance"] = model.feature_importance()
        imp_df = imp_df.sort_values("importance")

        # 比較用のランダム化したモデルを学習する
        null_imp_df = pd.DataFrame()
        for i in range(n_rum):
            print(i)
            ram_lgb_train = lgb.Dataset(self.X_train.drop("weight", axis=1), np.random.permutation(self.y_train), weight=self.X_train["weight"])
            ram_lgb_eval = lgb.Dataset(X_eval, np.random.permutation(y_eval), reference=lgb_train, weight=X_eval_weight)
            ram_model = lgb_original.train(this_param, ram_lgb_train,
                              # モデルの評価用データを渡す
                              valid_sets=ram_lgb_eval,
                              # 最大で 1000 ラウンドまで学習する
                              num_boost_round=ram_imp_num_boost_round,
                              # 10 ラウンド経過しても性能が向上しないときは学習を打ち切る
                              early_stopping_rounds=early_stopping_rounds)
            ram_imp_df = pd.DataFrame()
            ram_imp_df["feature"] = X_eval.columns
            ram_imp_df["importance"] = ram_model.feature_importance()
            ram_imp_df = ram_imp_df.sort_values("importance")
            ram_imp_df["run"] = i + 1
            null_imp_df = pd.concat([null_imp_df, ram_imp_df])

        # 閾値を超える特徴量を取得
        imp_features = []
        for feature in imp_df["feature"]:
            actual_value = imp_df.query(f"feature=='{feature}'")["importance"].values
            null_value = null_imp_df.query(f"feature=='{feature}'")["importance"].values
            percentage = (null_value < actual_value).sum() / null_value.size * 100
            if percentage >= threshold:
                imp_features.append(feature)
        print(len(imp_features))

        self._save_learning_model(imp_features, this_model_name + "_feat_columns")
        return imp_features


    def _learning_race_lgb(self, this_model_name, target):
        # テスト用のデータを評価用と検証用に分ける
        X_eval, X_valid, y_eval, y_valid = train_test_split(self.X_test, self.y_test, random_state=42)
        X_eval_weight = X_eval["weight"]
        X_eval = X_eval.drop("weight", axis=1)

        # データセットを生成する
        lgb_train = lgb.Dataset(self.X_train.drop("weight", axis=1), self.y_train, weight=self.X_train["weight"])#, categorical_feature=self.categ_columns)
        lgb_eval = lgb.Dataset(X_eval, y_eval, reference=lgb_train, weight=X_eval_weight)#, categorical_feature=self.categ_columns)

        if self.test_flag:
            num_boost_round=5
            early_stopping_rounds = 3
        else:
            num_boost_round=1000
            early_stopping_rounds = 50

        # 上記のパラメータでモデルを学習する
        best_params, history = {}, []
        this_param = self.lgbm_params[target]
        model = lgb.train(this_param, lgb_train,
                          valid_sets=lgb_eval,
                          verbose_eval=False,
                          num_boost_round=num_boost_round,
                          early_stopping_rounds=early_stopping_rounds,
                          best_params=best_params,
                          tuning_history=history)
        print("Bset Paramss:", best_params)
        print('Tuning history:', history)

        self._save_learning_model(model, this_model_name)



    def del_set_smote_data(self):
        """ 学習データのSMOTE処理を行い学習データを更新する  """
        # 対象数が少ない場合はサンプリングレートを下げる
        positive_count_train = self.y_train.sum()
        negative_count_train = len(self.y_train) - positive_count_train
        print("check y_train value 0:" + str(negative_count_train) + " 1:" + str(positive_count_train))
        if positive_count_train >= 6:
            smote = BorderlineSMOTE()
            self.X_train, self.y_train = smote.fit_sample(self.X_train, self.y_train)
        else:
            print("----- RandomOverSampler ----- ")
            ros = RandomOverSampler(
                # ratio={1: self.X_train.shape[0], 0: self.X_train.shape[0] // 3}, random_state=71)
                ratio={1: negative_count_train, 0: negative_count_train}, random_state=71)
            # 学習用データに反映
            self.X_train, self.y_train = ros.fit_sample(self.X_train, self.y_train)
        print("-- after sampling: " + str(np.unique(self.y_train, return_counts=True)))

    def _save_learning_model(self, model, model_name):
        """ 渡されたmodelをmodel_nameで保存する。

        :param object model: 学習モデル
        :param str model_name: モデル名
        """
        with open(self.model_folder + model_name + '.pickle', 'wb') as f:
            pickle.dump(model, f)


    def proc_predict_sk_model(self, df):
        """ predictする処理をまとめたもの。指定されたbashoのターゲットフラグ事の予測値を作成して連結したものをdataframeとして返す

        :param dataframe df: dataframe
        :param str val: str
        :return: dataframe
        """
        all_df = pd.DataFrame()
        if not df.empty:
            for target in self.obj_column_list:
                pred_df = self._predict_sk_model(df, target)
                if not pred_df.empty:
                    grouped_df = pred_df  #self._calc_grouped_data(pred_df)
                    grouped_df["target"] = target
                    grouped_df["model_name"] = self.model_name
                    all_df = pd.concat([all_df, grouped_df]).round(3)
        return all_df

    def _predict_sk_model(self, df, target):
        """ 指定された場所・ターゲットに対しての予測データの作成を行う

        :param dataframe  df: dataframe
        :param str val: str
        :param str target: str
        :return: dataframe
        """
        self._set_target_flag(target)
        all_pred_df = pd.DataFrame()
        this_model_name = self.model_name + "_" + target
        print("======= this_model_name: " + this_model_name + " ==========")
        with open(self.model_folder + this_model_name + '_feat_columns.pickle', 'rb') as f:
            imp_features = pickle.load(f)
        # x_dfにTRの値を持っていないのでTR前の値に切り替え
        #imp_features = [w.replace('tr_', '') for w in imp_features]
        #imp_features = list(set(imp_features))
        exp_df = df.drop(self.index_list, axis=1)
        exp_df = exp_df[imp_features]
        exp_df = self._set_predict_target_encoding(exp_df)
        exp_df = exp_df.to_numpy()
        print(exp_df.shape)
        if os.path.exists(self.model_folder + this_model_name + '.pickle'):
            with open(self.model_folder + this_model_name + '.pickle', 'rb') as f:
                model = pickle.load(f)
            y_pred = model.predict(exp_df)
            pred_df = self._sub_create_pred_df(df, y_pred)
            all_pred_df = pd.concat([all_pred_df, pred_df])
        return all_pred_df

    def _sub_create_pred_df(self, temp_df, y_pred):
        pred_df = temp_df[self.index_list].copy()
        pred_df.loc[:, "target_date"] = temp_df["NENGAPPI"]#.apply(lambda x: x[0:4] + "/" + x[4:6] + "/" + x[6:8])
        if self.version_str == 'raptype': #目的変数が多値なので横に結合してからunstack
            y_pred = pd.DataFrame(y_pred)
            pred_df = pd.concat([pred_df, y_pred], axis=1).set_index(["RACE_KEY", "target_date"])#, "NENGAPPI"
            pred_df = pred_df.stack().reset_index()
            pred_df.columns = ["RACE_KEY", "target_date", "pred_type", "prob"]#, "NENGAPPI"
        else:
            pred_df.loc[:, "prob"] = y_pred
            pred_df.loc[:, "pred"] = pred_df.apply(lambda x: 1 if x["prob"] >= 0.5 else 0, axis=1)
        return pred_df

    def create_import_data(self, pred_df):
        pred_df.dropna(inplace=True)
        if self.model_name == 'raceuma':
            grouped_all_df = pred_df.groupby(["RACE_KEY", "UMABAN", "target"], as_index=False).mean()
            date_df = pred_df[["RACE_KEY", "target_date"]].drop_duplicates()
            temp_grouped_df = pd.merge(grouped_all_df, date_df, on="RACE_KEY")
            grouped = temp_grouped_df.groupby(["RACE_KEY", "target"])
            grouped_df = grouped.describe()['prob'].reset_index()
            merge_df = pd.merge(temp_grouped_df, grouped_df, on=["RACE_KEY", "target"])
            merge_df['predict_std'] = (merge_df['prob'] - merge_df['mean']) / merge_df['std'] * 10 + 50
            temp_grouped_df['predict_rank'] = grouped['prob'].rank(method='min', ascending=False)
            merge_df = pd.merge(merge_df, temp_grouped_df[["RACE_KEY", "UMABAN", "predict_rank", "target"]], on=["RACE_KEY", "UMABAN", "target"])
            #return_df = merge_df[['RACE_KEY', 'UMABAN', 'target_date', 'prob', 'predict_std', 'predict_rank']]
            import_df = merge_df[["RACE_KEY", "UMABAN", "pred", "prob", "predict_std", "predict_rank", "target", "target_date"]].round(3)
        else:
            import_df = pred_df
        return import_df

    def import_data(self, import_df):
        date_list = sorted(import_df["target_date"].drop_duplicates().tolist())
        for date in date_list:
            print(date)
            target_df = import_df.query(f"target_date == '{date}'")
            if len(target_df["RACE_KEY"].drop_duplicates().tolist()) <= 9:
                print("数が少ないのでskip")
            else:
                target_df.to_pickle(self.pred_folder + self.version_str + "_" + date + ".pkl")
                #target_df.to_pickle(self.pred_folder + self.version_str + "_" + mu.convert_date_to_str(date) + ".pkl")

    def _set_predict_target_encoding(self, df):
        """ 渡されたdataframeに対してTargetEncodeを行いエンコードした値をセットしたdataframeを返す

        :param dataframe df: dataframe
        :return: dataframe
        """
        self._set_label_list(df)
        df_temp = self._set_target_encoding(df, False, "").copy()
        return df_temp

    def eval_pred_data(self, df):
        """ 予測されたデータの精度をチェック(raceumaのみ） """
        this_index_list = self.index_list
        this_index_list.remove("NENGAPPI")
        result_df = self._get_result_df()
        check_df = pd.merge(df, result_df, on=this_index_list)
        all_analyze_df = pd.DataFrame()
        for target in self.obj_column_list:
            print(target)
            target_df = check_df[check_df["target"] == target].copy()
            target_df.loc[:, "的中"] = target_df.apply(lambda x: 1 if x[target] == 1 else 0, axis=1)
            prob_analyze_df = target_df.copy()
            prob_analyze_df.loc[:, "bin"] = pd.cut(prob_analyze_df["prob"], 10)
            prob_analyze_df = prob_analyze_df[["bin", "的中", "target"]].copy()
            prob_analyze_df = prob_analyze_df.groupby(["bin", "target"]).describe().reset_index()
            prob_analyze_df.columns = ["bin", "target", "count", "mean", "std", "min", "25%", "50%", "75%", "max"]
            prob_analyze_df = prob_analyze_df[["bin", "target", "count", "mean"]].copy()
            prob_analyze_df.loc[:, "type"] = "prob"
            all_analyze_df = pd.concat([all_analyze_df, prob_analyze_df])

            std_analyze_df = target_df.copy()
            std_analyze_df.loc[:, "bin"] = pd.cut(std_analyze_df["predict_std"], 10)
            std_analyze_df = std_analyze_df[["bin", "的中", "target"]].copy()
            std_analyze_df = std_analyze_df.groupby(["bin", "target"]).describe().reset_index()
            std_analyze_df.columns = ["bin", "target", "count", "mean", "std", "min", "25%", "50%", "75%", "max"]
            std_analyze_df = std_analyze_df[["bin", "target", "count", "mean"]].copy()
            std_analyze_df.loc[:, "type"] = "std"
            all_analyze_df = pd.concat([all_analyze_df, std_analyze_df])
        return all_analyze_df

    def create_mark_std_df(self, import_df):
        """ 各予測値のFLAG値（偏差値）を横持変換する """
        mark_std_df = import_df[["RACE_KEY", "UMABAN", "target", "predict_std"]].copy()
        mark_std_df = mark_std_df.set_index(["RACE_KEY", "UMABAN", "target"]).unstack("target")
        mark_std_df.columns = ["ANA_FLAG", "JIKU_FLAG", "WIN_FLAG"]
        mark_std_df = mark_std_df.reset_index()
        return mark_std_df

    def create_mark_prob_df(self, import_df):
        """ 各予測値のFLAG値（偏差値）を横持変換する """
        mark_prob_df = import_df[["RACE_KEY", "UMABAN", "target", "prob"]].copy()
        mark_prob_df = mark_prob_df.set_index(["RACE_KEY", "UMABAN", "target"]).unstack("target")
        mark_prob_df.columns = ["ANA_PROB", "JIKU_PROB", "WIN_PROB"]
        mark_prob_df = mark_prob_df.reset_index()
        return mark_prob_df

    def create_mark_base_df(self, import_df):
        """ 各予測値を横持変換する """
        win_df = import_df.query("target == 'WIN_FLAG'").copy()
        win_df = win_df[["RACE_KEY", "UMABAN", "predict_std", "predict_rank", "prob", "target_date"]].rename(
            columns={"predict_std": "win_std", "predict_rank": "win_rank", "prob": "win_prob"})
        jiku_df = import_df.query("target == 'JIKU_FLAG'").copy()
        jiku_df = jiku_df[["RACE_KEY", "UMABAN", "predict_std", "predict_rank", "prob"]].rename(
            columns={"predict_std": "jiku_std", "predict_rank": "jiku_rank", "prob": "jiku_prob"})
        ana_df = import_df.query("target == 'ANA_FLAG'").copy()
        ana_df = ana_df[["RACE_KEY", "UMABAN", "predict_std", "predict_rank", "prob"]].rename(
            columns={"predict_std": "ana_std", "predict_rank": "ana_rank", "prob": "ana_prob"})
        mark_base_df = pd.merge(win_df, jiku_df, on=["RACE_KEY", "UMABAN"])
        mark_base_df = pd.merge(mark_base_df, ana_df, on=["RACE_KEY", "UMABAN"])
        mark_base_df.loc[:, "年月"] = mark_base_df["target_date"].str[0:6]
        return mark_base_df

    def create_mark_df(self, mark_base_df, win_rate, jiku_rate, ana_rate):
        """ rateをかけてScoreと順位を計算したデータフレームを作成する。rateは２５％とかパーセント値。その後各予測値のprob値を横変換して追加する """
        mark_df = mark_base_df.copy()
        mark_df.loc[:, "SCORE"] = mark_df["win_std"] * win_rate / 100 + mark_df["jiku_std"] * jiku_rate / 100 + mark_df[
            "ana_std"] * ana_rate / 100
        mark_df.loc[:, "RANK"] = mark_df.groupby("RACE_KEY")["SCORE"].rank(ascending=False)
        return mark_df

    def get_result_df(self):
        result_df = self.ld.ext.get_raceuma_table_base()[["RACE_KEY", "UMABAN", "着順", "単勝", "複勝"]].copy()
        return result_df

    def calc_monthly_tanpuku_df(self, mark_df, result_df):
        ym_list = mark_df["年月"].drop_duplicates().tolist()
        summary_df = pd.DataFrame()
        for ym in ym_list:
            temp_mark_df = mark_df.query(f"年月 == '{ym}'").copy()
            temp_mark_df = pd.merge(temp_mark_df, result_df, on=["RACE_KEY", "UMABAN"])
            for cond in ["win_rank == 1", "jiku_rank == 1", "ana_rank == 1"]:
                temp_df = temp_mark_df.query(cond).copy()
                cond_text = f"{cond} 年月 == {ym}"
                temp_sr = self._calc_tanpuku_summary(temp_df, cond_text)
                summary_df = summary_df.append(temp_sr, ignore_index=True)
        return summary_df

    def calc_monthly_mark_df(self, mark_df ,result_df):
        ym_list = mark_df["年月"].drop_duplicates().tolist()
        summary_df = pd.DataFrame()
        for ym in ym_list:
            temp_mark_df = mark_df.query(f"年月 == '{ym}'").copy()
            temp_mark_df = pd.merge(temp_mark_df, result_df, on=["RACE_KEY", "UMABAN"])
            for rank in [1,2,3,4,5]:
                temp_df = temp_mark_df.query(f"RANK == {rank}").copy()
                cond_text = f"RANK == {rank} and 年月 == {ym}"
                temp_sr = self._calc_tanpuku_summary(temp_df, cond_text)
                summary_df = summary_df.append(temp_sr, ignore_index=True)
        return summary_df


    def calc_rank_mark_df(self, mark_df ,result_df):
        summary_df = pd.DataFrame()
        temp_mark_df = pd.merge(mark_df, result_df, on=["RACE_KEY", "UMABAN"])
        for rank in [1,2,3,4,5]:
            temp_df = temp_mark_df.query(f"RANK == {rank}").copy()
            cond_text = f"RANK == {rank}"
            temp_sr = self._calc_tanpuku_summary(temp_df, cond_text)
            summary_df = summary_df.append(temp_sr, ignore_index=True)
        return summary_df

    def simulate_mark_rate(self, mark_base_df, result_df, fukusho_base):
        win_range = [35, 40, 45, 50]
        jiku_range = [35, 40, 45, 50]
        ana_range = [10, 15, 20, 25, 30]
        mark_base_df = pd.merge(mark_base_df, result_df, on=["RACE_KEY", "UMABAN"])
        summary_df = pd.DataFrame()
        for win in win_range:
            for jiku in jiku_range:
                for ana in ana_range:
                    if win + jiku + ana == 100:
                        cond = f"WIN:{win}% JIKU:{jiku}% ANA:{ana}% and RANK <= 3"
                        mark_df = self.create_mark_df(mark_base_df, win, jiku, ana)
                        mark_df = mark_df.query("RANK <= 3")
                        temp_sr = self._calc_tanpuku_summary(mark_df, cond)
                        temp_sr["win_rate"] = win
                        temp_sr["jiku_rate"] = jiku
                        temp_sr["ana_rate"] = ana
                        summary_df = summary_df.append(temp_sr, ignore_index=True)
        summary_df = summary_df.query(f"複勝的中率 >= {fukusho_base}").sort_values("複勝回収率", ascending=False).reset_index()
        return summary_df


    def _calc_tanpuku_summary(self, df, cond_text):
        all_count = len(df)
        race_count = len(df["RACE_KEY"].drop_duplicates())
        tansho_hit_df = df[df["単勝"] != 0]
        tansho_hit_count = len(tansho_hit_df)
        tansho_race_hit_count = len(tansho_hit_df["RACE_KEY"].drop_duplicates())
        tansho_avg_return = round(tansho_hit_df["単勝"].mean(), 0)
        tansho_std_return = round(tansho_hit_df["単勝"].std(), 0)
        tansho_max_return = tansho_hit_df["単勝"].max()
        tansho_avg = round(df["単勝"].mean() , 1)
        tansho_hit_rate = round(tansho_hit_count / all_count * 100 , 1) if all_count !=0 else 0
        tansho_race_hit_rate = round(tansho_race_hit_count / race_count * 100 , 1) if race_count !=0 else 0

        fukusho_hit_df = df[df["複勝"] != 0]
        fukusho_hit_count = len(fukusho_hit_df)
        fukusho_race_hit_count = len(fukusho_hit_df["RACE_KEY"].drop_duplicates())
        fukusho_avg_return = round(fukusho_hit_df["複勝"].mean(), 0)
        fukusho_std_return = round(fukusho_hit_df["複勝"].std(), 0)
        fukusho_max_return = fukusho_hit_df["複勝"].max()
        fukusho_avg = round(df["複勝"].mean() , 1)
        fukusho_hit_rate = round(fukusho_hit_count / all_count * 100 , 1) if all_count !=0 else 0
        fukusho_race_hit_rate = round(fukusho_race_hit_count / race_count * 100 , 1) if race_count !=0 else 0

        sr = pd.Series(data=[cond_text, all_count, race_count, tansho_hit_count, tansho_avg, tansho_hit_rate, tansho_race_hit_rate,
                             tansho_avg_return, tansho_std_return, tansho_max_return,
                             fukusho_hit_count, fukusho_avg, fukusho_hit_rate, fukusho_race_hit_rate,
                             fukusho_avg_return, fukusho_std_return, fukusho_max_return]
                       , index=["条件", "件数", "レース数", "単勝的中数", "単勝回収率", "単勝的中率", "単勝R的中率", "単勝払戻平均", "単勝払戻偏差", "単勝最大払戻",
                                "複勝的中数", "複勝回収率", "複勝的中率", "複勝R的中率", "複勝払戻平均", "複勝払戻偏差", "複勝最大払戻"])
        return sr.fillna(0)

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
        sr = pd.Series(data=[cond_text, all_count, hit_count, race_count, avg, hit_rate, race_hit_rate, avg_return, std_return, max_return, all_count * 100 , sum_return]
                       , index=["条件", "件数", "的中数", "レース数", "回収率", "的中率", "R的中率", "払戻平均", "払戻偏差", "最大払戻", "購入総額", "払戻総額"])
        return sr.fillna(0)