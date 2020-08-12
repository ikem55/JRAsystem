import my_config as mc
import modules.util as mu

import pandas as pd
import featuretools as ft

class SkModel(object):
    """
    モデルに関する情報を定義する。データの加工、特徴量生成を行う。SkProcオブジェクトから呼び出す
    """
    categ_columns = []
    target_enc_columns = []
    class_dict = [{"name": "芝", "code": "1", "except_list": ["芝ダ障害コード", "転圧", "凍結防止剤"]},
                  {"name": "ダ", "code": "2", "except_list": ["芝ダ障害コード", "芝馬場状態コード", "芝種類", "草丈"]}]

    def __init__(self, model_name, version_str, start_date, end_date, test_flag):
        self.model_name = model_name
        self.version_str = version_str
        self.start_date = start_date
        self.end_date = end_date
        self.dict_path = mc.return_base_path(test_flag)
        self.dict_folder = self.dict_path + 'dict/' + self.version_str + '/'
        self._set_columns()

    def _set_columns(self):
        if self.model_name == 'raceuma':
            self.categ_columns = ['芝ダ障害コード', '右左', '内外', '種別', '条件', '記号', '重量', 'コース', '曜日', '天候コード', '芝馬場状態コード', '芝馬場状態内', '芝馬場状態中', '芝馬場状態外',
                                  '直線馬場差最内', '直線馬場差内', '直線馬場差中', '直線馬場差外', '直線馬場差大外', 'ダ馬場状態コード', 'ダ馬場状態内', 'ダ馬場状態中', 'ダ馬場状態外', '芝種類',
                                  '転圧', '凍結防止剤', '非根幹', '距離グループ', 'course_cluster', '脚質', '距離適性', '上昇度', '調教矢印コード', '厩舎評価コード', '蹄コード', '重適正コード',
                                  'クラスコード', 'ブリンカー', '見習い区分', '総合印', 'ＩＤＭ印', '情報印', '騎手印', '厩舎印', '調教印', '激走印', '芝適性コード', 'ダ適性コード', '条件クラス',
                                  'ペース予想', '道中内外', '後３Ｆ内外', 'ゴール内外', '展開記号', '距離適性２', '性別コード', '馬主名', '馬主会コード', '馬記号コード', '輸送区分', '万券印',
                                  '激走タイプ', '休養理由分類コード', '芝ダ障害フラグ', '距離フラグ', 'クラスフラグ', '転厩フラグ', '去勢フラグ', '乗替フラグ', '放牧先', '放牧先ランク',
                                  '厩舎ランク', 'EM', '厩舎ＢＢ印', '騎手ＢＢ印', '調教曜日', '調教コースコード', '追切種類', '追い状態', '乗り役', '併せ結果', '併せ追切種類', '調教タイプ',
                                  '調教コース種別', '調教距離', '調教重点', '調教量評価', '仕上指数変化', '調教評価',
                                  #'raceuma_before_taikei_0', 'raceuma_before_taikei_1', 'raceuma_before_taikei_2','raceuma_before_tokki_0', 'raceuma_before_tokki_1',
                                  '馬番グループ', '基準人気グループ', '毛色コード', '父馬名', '母父馬名', '生産者名', '産地名', '父系統コード', '母父系統コード',

                                  '芝ダ障害コード_1', '右左_1', '内外_1', '馬場状態_1', '種別_1', '条件_1', '記号_1', '重量_1', '不利_1',
                                  '４角コース取り_1', 'RAP_TYPE_1', '芝種類_1', '転圧_1', '凍結防止剤_1',
                                  '芝_1', '外_1', '重_1', '軽_1', '場コード_1', 'course_cluster_1', 'ru_cluster_1', '平坦コース_1',
                                  '急坂コース_1', '好走_1', '内枠_1', '外枠_1', '中山_1', '東京_1',
                                  '上がりかかる_1', '上がり速い_1', 'ダート道悪_1', '前崩れレース_1', '先行惜敗_1', '前残りレース_1', '差し損ね_1',
                                  '上がり幅小さいレース_1', '内枠砂被り_1',

                                  '芝ダ障害コード_2', '右左_2', '内外_2', '馬場状態_2', '種別_2', '条件_2', '記号_2', '重量_2', '不利_2',
                                  '４角コース取り_2', 'RAP_TYPE_2', '芝種類_2', '転圧_2', '凍結防止剤_2',
                                  '芝_2', '外_2', '重_2', '軽_2', '場コード_2', 'course_cluster_2', 'ru_cluster_2', '平坦コース_2',
                                  '急坂コース_2', '好走_2', '内枠_2', '外枠_2', '中山_2', '東京_2',
                                  '上がりかかる_2', '上がり速い_2', 'ダート道悪_2', '前崩れレース_2', '先行惜敗_2', '前残りレース_2', '差し損ね_2',
                                  '上がり幅小さいレース_2', '内枠砂被り_2',

                                  '芝ダ障害コード_3', '右左_3', '内外_3', '馬場状態_3', '種別_3', '条件_3', '記号_3', '重量_3', '不利_3',
                                  '４角コース取り_3', 'RAP_TYPE_3', '芝種類_3', '転圧_3', '凍結防止剤_3',
                                  '芝_3', '外_3', '重_3', '軽_3', '場コード_3', 'course_cluster_3', 'ru_cluster_3', '平坦コース_3',
                                  '急坂コース_3', '好走_3', '内枠_3', '外枠_3', '中山_3', '東京_3',
                                  '上がりかかる_3', '上がり速い_3', 'ダート道悪_3', '前崩れレース_3', '先行惜敗_3', '前残りレース_3', '差し損ね_3',
                                  '上がり幅小さいレース_3', '内枠砂被り_3',

                                  '芝ダ障害コード_4', '右左_4', '内外_4', '馬場状態_4', '種別_4', '条件_4', '記号_4', '重量_4', '不利_4',
                                  '４角コース取り_4', 'RAP_TYPE_4', '芝種類_4', '転圧_4', '凍結防止剤_4',
                                  '芝_4', '外_4', '重_4', '軽_4', '場コード_4', 'course_cluster_4', 'ru_cluster_4', '平坦コース_4',
                                  '急坂コース_4', '好走_4', '内枠_4', '外枠_4', '中山_4', '東京_4',
                                  '上がりかかる_4', '上がり速い_4', 'ダート道悪_4', '前崩れレース_4', '先行惜敗_4', '前残りレース_4', '差し損ね_4',
                                  '上がり幅小さいレース_4', '内枠砂被り_4',

                                  '芝ダ障害コード_5', '右左_5', '内外_5', '馬場状態_5', '種別_5', '条件_5', '記号_5', '重量_5', '不利_5',
                                  '４角コース取り_5', 'RAP_TYPE_5', '芝種類_5', '転圧_5', '凍結防止剤_5',
                                  '芝_5', '外_5', '重_5', '軽_5', '場コード_5', 'course_cluster_5', 'ru_cluster_5', '平坦コース_5',
                                  '急坂コース_5', '好走_5', '内枠_5', '外枠_5', '中山_5', '東京_5',
                                  '上がりかかる_5', '上がり速い_5', 'ダート道悪_5', '前崩れレース_5', '先行惜敗_5', '前残りレース_5', '差し損ね_5',
                                  '上がり幅小さいレース_5', '内枠砂被り_5',

                                  '継続騎乗', '同根幹', '同距離グループ', '前走凡走', '前走激走', '前走逃げそびれ', '前走加速ラップ', '平坦', '急坂', '内枠', '外枠', '枠', '中山', '東京', '上がり遅',
                                  '上がり速', 'ダート道悪', '突然バテた馬の距離短縮', '短距離からの延長', '中距離からの延長', '前崩れレースで先行惜敗', '前残りレースで差し損ね', '上がり幅小さいレースで差し損ね',
                                  'ダート短距離血統１', '内枠短縮', '外枠短縮', '内枠延長', '外枠延長', '延長得意父', '延長得意母父', '砂被り苦手父', '砂被り苦手母父', '逆ショッカー', '前走砂被り外枠',
                                  '母父サンデー短縮', 'ボールドルーラー系', 'ダーレーパイロ', 'ダ1200ｍ好走父米国型', 'ダマスカス系', '中山ダ１８００血統', 'ナスルーラ系', 'ダート系サンデー', 'マイル芝血統',
                                  'ノーザンＦ', 'ニジンスキー系', 'Ｐサンデー系', '母父トニービン', 'グレイソヴリン系', 'マッチェム系', '父ルーラーシップ', '父ネオユニ短縮', '父ロードカナロア',
                                  '母父ディープ', '母父キンカメ'
                                  ]

            self.target_enc_columns = ['脚質', '距離適性', '上昇度', '調教矢印コード', '厩舎評価コード', '蹄コード', '重適正コード', 'クラスコード', 'ブリンカー', '見習い区分', '総合印', 'ＩＤＭ印', '情報印',
                                       '騎手印', '厩舎印', '調教印', '激走印', '展開記号', '性別コード', '馬主名', '馬主会コード', '馬記号コード', '輸送区分', '万券印', '激走タイプ', '休養理由分類コード',
                                       '芝ダ障害フラグ', '距離フラグ', 'クラスフラグ', '転厩フラグ', '去勢フラグ', '乗替フラグ', '放牧先', '放牧先ランク', '厩舎ランク', 'EM', '厩舎ＢＢ印', '騎手ＢＢ印',
                                       '追切種類', '追い状態', '乗り役', '併せ結果', '併せ追切種類', '調教タイプ', '調教コース種別', '調教量評価', '仕上指数変化', '調教評価',
                                       #'raceuma_before_taikei_0','raceuma_before_taikei_1', 'raceuma_before_taikei_2', 'raceuma_before_tokki_0', 'raceuma_before_tokki_1',
                                       '基準人気グループ', '父馬名', '母父馬名', '生産者名',
                                       '産地名', '父系統コード', '母父系統コード', '継続騎乗', '同根幹', '同距離グループ', '前走凡走', '前走激走', '前走逃げそびれ', '前走加速ラップ', '平坦', '急坂', '内枠', '外枠',
                                       '枠', '中山', '東京', '上がり遅', '上がり速', 'ダート道悪', '突然バテた馬の距離短縮', '短距離からの延長', '中距離からの延長', '前崩れレースで先行惜敗', '前残りレースで差し損ね',
                                       '上がり幅小さいレースで差し損ね', 'ダート短距離血統１', '内枠短縮', '外枠短縮', '内枠延長', '外枠延長', '延長得意父', '延長得意母父', '砂被り苦手父', '砂被り苦手母父', '逆ショッカー',
                                       '前走砂被り外枠', '母父サンデー短縮', 'ボールドルーラー系', 'ダーレーパイロ', 'ダ1200ｍ好走父米国型', 'ダマスカス系', '中山ダ１８００血統', 'ナスルーラ系', 'ダート系サンデー',
                                       'マイル芝血統', 'ノーザンＦ', 'ニジンスキー系', 'Ｐサンデー系', '母父トニービン', 'グレイソヴリン系', 'マッチェム系', '父ルーラーシップ', '父ネオユニ短縮', '父ロードカナロア',
                                       '母父ディープ', '母父キンカメ']
        elif self.model_name == 'race':
            self.categ_columns = ['芝ダ障害コード', '右左', '内外', '種別', '条件', '記号', '重量', 'コース', '曜日', '天候コード', '芝馬場状態コード', '芝馬場状態内', '芝馬場状態中', '芝馬場状態外', '直線馬場差最内',
                                  '直線馬場差内', '直線馬場差中', '直線馬場差外', '直線馬場差大外', 'ダ馬場状態コード', 'ダ馬場状態内', 'ダ馬場状態中', 'ダ馬場状態外', '芝種類', '転圧', '凍結防止剤', '中間降水量', '非根幹',
                                  '距離グループ', 'course_cluster',
                                  '逃げ_脚質', '逃げ_距離適性', '逃げ_上昇度', '逃げ_蹄コード', '逃げ_見習い区分', '逃げ_総合印', '逃げ_ＩＤＭ印', '逃げ_輸送区分', '逃げ_激走タイプ', '逃げ_休養理由分類コード', '逃げ_芝ダ障害フラグ',
                                  '逃げ_距離フラグ', '逃げ_クラスフラグ', '逃げ_転厩フラグ', '逃げ_去勢フラグ', '逃げ_乗替フラグ',
                                  '上り_脚質', '上り_距離適性', '上り_上昇度', '上り_蹄コード', '上り_見習い区分', '上り_総合印', '上り_ＩＤＭ印', '上り_輸送区分', '上り_激走タイプ', '上り_休養理由分類コード', '上り_芝ダ障害フラグ',
                                  '上り_距離フラグ', '上り_クラスフラグ', '上り_転厩フラグ', '上り_去勢フラグ', '上り_乗替フラグ',
                                  '人気_脚質', '人気_距離適性', '人気_上昇度', '人気_蹄コード', '人気_見習い区分', '人気_総合印', '人気_ＩＤＭ印', '人気_情報印', '人気_騎手印', '人気_厩舎印', '人気_調教印', '人気_激走印', '人気_展開記号',
                                  '人気_輸送区分', '人気_激走タイプ', '人気_休養理由分類コード', '人気_芝ダ障害フラグ', '人気_距離フラグ', '人気_クラスフラグ', '人気_転厩フラグ', '人気_去勢フラグ', '人気_乗替フラグ', '人気_放牧先ランク', '人気_厩舎ランク',
                                  '人気_調教量評価', '人気_仕上指数変化', '人気_調教評価'
                                  ]

            self.target_enc_columns = ['芝ダ障害コード', '右左', '内外', '種別', '条件', '記号', '重量', 'コース', '曜日', '天候コード', '芝馬場状態コード', '芝馬場状態内', '芝馬場状態中', '芝馬場状態外', '直線馬場差最内',
                                  '直線馬場差内', '直線馬場差中', '直線馬場差外', '直線馬場差大外', 'ダ馬場状態コード', 'ダ馬場状態内', 'ダ馬場状態中', 'ダ馬場状態外', '芝種類', '転圧', '凍結防止剤', '中間降水量', '非根幹',
                                  '距離グループ', 'course_cluster',
                                  '逃げ_脚質', '逃げ_距離適性', '逃げ_上昇度', '逃げ_蹄コード', '逃げ_見習い区分', '逃げ_総合印', '逃げ_ＩＤＭ印', '逃げ_輸送区分', '逃げ_激走タイプ', '逃げ_休養理由分類コード', '逃げ_芝ダ障害フラグ',
                                  '逃げ_距離フラグ', '逃げ_クラスフラグ', '逃げ_転厩フラグ', '逃げ_去勢フラグ', '逃げ_乗替フラグ',
                                  '上り_脚質', '上り_距離適性', '上り_上昇度', '上り_蹄コード', '上り_見習い区分', '上り_総合印', '上り_ＩＤＭ印', '上り_輸送区分', '上り_激走タイプ', '上り_休養理由分類コード', '上り_芝ダ障害フラグ',
                                  '上り_距離フラグ', '上り_クラスフラグ', '上り_転厩フラグ', '上り_去勢フラグ', '上り_乗替フラグ',
                                  '人気_脚質', '人気_距離適性', '人気_上昇度', '人気_蹄コード', '人気_見習い区分', '人気_総合印', '人気_ＩＤＭ印', '人気_情報印', '人気_騎手印', '人気_厩舎印', '人気_調教印', '人気_激走印', '人気_展開記号',
                                  '人気_輸送区分', '人気_激走タイプ', '人気_休養理由分類コード', '人気_芝ダ障害フラグ', '人気_距離フラグ', '人気_クラスフラグ', '人気_転厩フラグ', '人気_去勢フラグ', '人気_乗替フラグ', '人気_放牧先ランク', '人気_厩舎ランク',
                                  '人気_調教量評価', '人気_仕上指数変化', '人気_調教評価'
                                  ]

    def get_merge_df(self, race_df, raceuma_df, horse_df, prev_raceuma_df, prev_feature_raceuma_df):
        base_df = pd.merge(race_df, raceuma_df, on=["RACE_KEY", "target_date", "NENGAPPI"])
        base_df = pd.merge(base_df, horse_df, on=["血統登録番号", "target_date"])
        base_df = pd.merge(base_df, prev_raceuma_df, on=["RACE_KEY", "UMABAN", "target_date"], how='left')
        base_df = pd.merge(base_df, prev_feature_raceuma_df, on=["RACE_KEY", "UMABAN"], how='left')
        return base_df

    ### raceuma
    def get_create_feature_raceuma_df(self, base_df):
        """ マージしたデータから特徴量を生成する """
        base_df.fillna(({'平坦コース_1':0, '平坦コース_2':0, '平坦コース_3':0, '平坦コース_4':0, '平坦コース_5':0, '急坂コース_1':0, '急坂コース_2':0, '急坂コース_3':0, '急坂コース_4':0, '急坂コース_5':0, '好走_1':0, '好走_2':0, '好走_3':0, '好走_4':0, '好走_5':0,
                              '内枠_1': 0, '内枠_2': 0, '内枠_3': 0, '内枠_4': 0, '内枠_5': 0, '外枠_1': 0, '外枠_2': 0, '外枠_3': 0, '外枠_4': 0, '外枠_5': 0, '上がりかかる_1': 0, '上がりかかる_2': 0, '上がりかかる_3': 0, '上がりかかる_4': 0, '上がりかかる_5': 0,
                              '上がり速い_1': 0, '上がり速い_2': 0, '上がり速い_3': 0, '上がり速い_4': 0, '上がり速い_5': 0, 'ダート道悪_1': 0, 'ダート道悪_2': 0, 'ダート道悪_3': 0, 'ダート道悪_4': 0, 'ダート道悪_5': 0,
                              '前崩れレース_1': 0, '前崩れレース_2': 0, '前崩れレース_3': 0, '前崩れレース_4': 0, '前崩れレース_5': 0, '先行惜敗_1': 0, '先行惜敗_2': 0, '先行惜敗_3': 0, '先行惜敗_4': 0, '先行惜敗_5': 0,
                              '前残りレース_1': 0, '前残りレース_2': 0, '前残りレース_3': 0, '前残りレース_4': 0, '前残りレース_5': 0, '差し損ね_1': 0, '差し損ね_2': 0, '差し損ね_3': 0, '差し損ね_4': 0, '差し損ね_5': 0,
                              '中山_1': 0, '中山_2': 0, '中山_3': 0, '中山_4': 0, '中山_5': 0, '東京_1': 0, '東京_2': 0, '東京_3': 0, '東京_4': 0, '東京_5': 0,
                              '上がり幅小さいレース_1': 0, '上がり幅小さいレース_2': 0, '上がり幅小さいレース_3': 0, '上がり幅小さいレース_4': 0, '上がり幅小さいレース_5': 0}), inplace=True)
        base_df.loc[:, "継続騎乗"] = (base_df["騎手コード"] == base_df["騎手コード_1"]).astype(int)
        base_df.loc[:, "距離増減"] = base_df["距離"] - base_df["距離_1"]
        base_df.loc[:, "頭数増減"] = base_df["頭数"] - base_df["頭数_1"]
        base_df.loc[:, "同根幹"] = (base_df["非根幹"] == base_df["非根幹_1"]).astype(int)
        base_df.loc[:, "同距離グループ"] = (base_df["距離グループ"] == base_df["距離グループ_1"]).astype(int)
        base_df.loc[:, "前走凡走"] = base_df.apply(lambda x: 1 if (x["人気率_1"] < 0.3 and x["着順率_1"] > 0.5) else 0, axis=1)
        base_df.loc[:, "前走激走"] = base_df.apply(lambda x: 1 if (x["人気率_1"] > 0.5 and x["着順率_1"] < 0.3) else 0, axis=1)
        base_df.loc[:, "前走逃げそびれ"] = base_df.apply(lambda x: 1 if (x["展開記号"] == '1' and x["先行率_1"] > 0.5) else 0, axis=1)
        base_df.loc[:, "前走加速ラップ"] = base_df["ラップ差１ハロン_1"].apply(lambda x: 1 if x > 0 else 0)
        base_df.loc[:, "平坦"] = base_df.apply(lambda x: x["平坦コース_1"] * x["好走_1"] + x["平坦コース_2"] * x["好走_2"] + x["平坦コース_3"] * x["好走_3"] + x["平坦コース_4"] * x["好走_4"] + x["平坦コース_5"] * x["好走_5"], axis=1)
        base_df.loc[:, "急坂"] = base_df.apply(lambda x: x["急坂コース_1"] * x["好走_1"] + x["急坂コース_2"] * x["好走_2"] + x["急坂コース_3"] * x["好走_3"] + x["急坂コース_4"] * x["好走_4"] + x["急坂コース_5"] * x["好走_5"], axis=1)
        base_df.loc[:, "内枠"] = base_df.apply(lambda x: x["内枠_1"] * x["好走_1"] + x["内枠_2"] * x["好走_2"] + x["内枠_3"] * x["好走_3"] + x["内枠_4"] * x["好走_4"] + x["内枠_5"] * x["好走_5"], axis=1)
        base_df.loc[:, "外枠"] = base_df.apply(lambda x: x["外枠_1"] * x["好走_1"] + x["外枠_2"] * x["好走_2"] + x["外枠_3"] * x["好走_3"] + x["外枠_4"] * x["好走_4"] + x["外枠_5"] * x["好走_5"], axis=1)
        base_df.loc[:, "枠"] = base_df.apply(lambda x: x["内枠"] if int(x["UMABAN"]) / x["頭数"] <= 0.5 else x["外枠"], axis=1)
        base_df.loc[:, "中山"] = base_df.apply(lambda x: x["中山_1"] * x["好走_1"] + x["中山_2"] * x["好走_2"] + x["中山_3"] * x["好走_3"] + x["中山_4"] * x["好走_4"] + x["中山_5"] * x["好走_5"], axis=1)
        base_df.loc[:, "東京"] = base_df.apply(lambda x: x["東京_1"] * x["好走_1"] + x["東京_2"] * x["好走_2"] + x["東京_3"] * x["好走_3"] + x["東京_4"] * x["好走_4"] + x["東京_5"] * x["好走_5"], axis=1)
        base_df.loc[:, "上がり遅"] = base_df.apply(lambda x: x["上がりかかる_1"] * x["好走_1"] + x["上がりかかる_2"] * x["好走_2"] + x["上がりかかる_3"] * x["好走_3"] + x["上がりかかる_4"] * x["好走_4"] + x["上がりかかる_5"] * x["好走_5"], axis=1)
        base_df.loc[:, "上がり速"] = base_df.apply(lambda x: x["上がり速い_1"] * x["好走_1"] + x["上がり速い_2"] * x["好走_2"] + x["上がり速い_3"] * x["好走_3"] + x["上がり速い_4"] * x["好走_4"] + x["上がり速い_5"] * x["好走_5"], axis=1)
        base_df.loc[:, "ダート道悪"] = base_df.apply(lambda x: x["ダート道悪_1"] * x["好走_1"] + x["ダート道悪_2"] * x["好走_2"] + x["ダート道悪_3"] * x["好走_3"] + x["ダート道悪_4"] * x["好走_4"] + x["ダート道悪_5"] * x["好走_5"], axis=1)
        base_df.loc[:, "突然バテた馬の距離短縮"] = base_df.apply(lambda x: 1 if x["ru_cluster_1"] == 5 and x["距離増減"] <= -200 else 0, axis=1)
        base_df.loc[:, "短距離からの延長"] = base_df.apply(lambda x: 1 if x["距離_1"] <= 1600 and x["距離増減"] >= 200 and x["ru_cluster_1"] in (1, 6) else 0, axis=1)
        base_df.loc[:, "中距離からの延長"] = base_df.apply(lambda x: 1 if x["距離_1"] > 1600 and x["距離増減"] >= 200 and x["ru_cluster_1"] in (0, 6) else 0, axis=1)
        base_df.loc[:, "前崩れレースで先行惜敗"] = base_df.apply(lambda x: x["前崩れレース_1"] * x["先行惜敗_1"] + x["前崩れレース_2"] * x["先行惜敗_2"] + x["前崩れレース_3"] * x["先行惜敗_3"] + x["前崩れレース_4"] * x["先行惜敗_4"] + x["前崩れレース_5"] * x["先行惜敗_5"], axis=1)
        base_df.loc[:, "前残りレースで差し損ね"] = base_df.apply(lambda x: x["前残りレース_1"] * x["差し損ね_1"] + x["前残りレース_2"] * x["差し損ね_2"] + x["前残りレース_3"] * x["差し損ね_3"] + x["前残りレース_4"] * x["差し損ね_4"] + x["前残りレース_5"] * x["差し損ね_5"], axis=1)
        base_df.loc[:, "上がり幅小さいレースで差し損ね"] = base_df.apply(lambda x: x["上がり幅小さいレース_1"] * x["差し損ね_1"] + x["上がり幅小さいレース_2"] * x["差し損ね_2"] + x["上がり幅小さいレース_3"] * x["差し損ね_3"] + x["上がり幅小さいレース_4"] * x["差し損ね_4"] + x["上がり幅小さいレース_5"] * x["差し損ね_5"], axis=1)
        base_df.loc[:, "ダート短距離血統１"] = base_df["父馬名"].apply(lambda x: 1 if x in ('サウスヴィグラス', 'エンパイアメーカー', 'エーピーインディ', 'カジノドライヴ', 'パイロ') else 0)
        base_df.loc[:, "内枠短縮"] = base_df.apply(lambda x: 1 if int(x["UMABAN"]) / x["頭数"] <= 0.3 and x["距離増減"] <= -200 else 0, axis=1)
        base_df.loc[:, "外枠短縮"] = base_df.apply(lambda x: 1 if int(x["UMABAN"]) / x["頭数"] >= 0.7 and x["距離増減"] <= -200 else 0, axis=1)
        base_df.loc[:, "内枠延長"] = base_df.apply(lambda x: 1 if int(x["UMABAN"]) / x["頭数"] <= 0.3 and x["距離増減"] >= 200 else 0, axis=1)
        base_df.loc[:, "外枠延長"] = base_df.apply(lambda x: 1 if int(x["UMABAN"]) / x["頭数"] >= 0.7 and x["距離増減"] >= 200 else 0, axis=1)
        base_df.loc[:, "延長得意父"] = base_df["父馬名"].apply(lambda x:1 if x in ('ウォーエンブレム', 'オレハマッテルゼ', 'キャプテンスティーヴ', 'コンデュイット', 'スズカマンボ', 'チーフベアハート', 'チチカステナンゴ', 'ディープスカイ', 'ネオユニヴァース', 'ハーツクライ', 'ハービンジャー', 'フォーティナイナーズサン', 'マリエンバード', 'メイショウサムソン', 'ワークフォース') else 0 )
        base_df.loc[:, "延長得意母父"] = base_df["母父馬名"].apply(lambda x: 1 if x in ('アサティス', 'エルコレドール', 'エルコンドルパサー', 'オジジアン', 'クロコルージュ', 'コマンダーインチーフ', 'スキャターザゴールド', 'フォレストリー', 'フサイチペガサス', 'ホワイトマズル', 'マーケトリー', 'ミシル', 'モンズン', 'メジロマックイーン') else 0)
        base_df.loc[:, "砂被り苦手父"] = base_df["父馬名"].apply(lambda x: 1 if x in ('アドマイヤオーラ', 'アドマイヤマックス', 'コンデュイット', 'ステイゴールド', 'タイキシャトル', 'ダノンシャンティ', 'ナカヤマフェスタ', 'ハービンジャー', 'ファルブラヴ', 'マイネルラヴ', 'ローエングリン') else 0)
        base_df.loc[:, "砂被り苦手母父"] = base_df["母父馬名"].apply(lambda x: 1 if x in ('アンバーシャダイ', 'エリシオ', 'カーリアン', 'サッカーボーイ', 'タマモクロス', 'ニホンピロヴィナー', 'メジロライアン', 'ロックオブジブランルタル', 'ロドリゴデトリアーノ') else 0)
        base_df.loc[:, "逆ショッカー"] = base_df.apply(lambda x: 1 if x["距離増減"] <= -200 and x["コーナー順位３_1"] >= 5 and x["道中順位"] <= 8 else 0, axis=1)
        base_df.loc[:, "前走砂被り外枠"] = base_df.apply(lambda x: 1 if x["芝ダ障害コード_1"] == '2' and x["内枠砂被り_1"] == 1 else 0, axis=1)
        base_df.loc[:, "母父サンデー短縮"] = base_df.apply(lambda x: 1 if x["母父馬名"] == "サンデーサイレンス" and x["距離増減"] <= -200 else 0, axis=1)
        base_df.loc[:, "ボールドルーラー系"] = base_df.apply(lambda x: 1 if x["父系統コード"] == '1305' or x["母父系統コード"] == '1305' else 0, axis=1)
        base_df.loc[:, "ダーレーパイロ"] = base_df.apply(lambda x: 1 if x["生産者名"] == "ダーレー・ジャパン・ファーム" and x["父馬名"] == "パイロ" else 0, axis=1)
        base_df.loc[:, "ダ1200ｍ好走父米国型"] = base_df.apply(lambda x: 1 if x["ダート短距離血統１"] == 1 and x["芝ダ障害コード_1"] == '2' and x["距離_1"] <= 1200 and x["着順_1"] in (1,2,3) else 0, axis=1)
        base_df.loc[:, "ダマスカス系"] = base_df.apply(lambda x: 1 if x["父系統コード"] == '1701' or x["母父系統コード"] == '1701' else 0, axis=1)
        base_df.loc[:, "中山ダ１８００血統"] = base_df.apply(lambda x: 1 if x["母父系統コード"] != "1206" and x["父馬名"] in ('キングカメハメハ', 'ロージズインメイ', 'アイルハヴアナザー') else 0, axis=1)
        base_df.loc[:, "ナスルーラ系"] = base_df.apply(lambda x: 1 if x["父系統コード"] == '1301' or x["母父系統コード"] == '1301' else 0, axis=1)
        base_df.loc[:, "ダート系サンデー"] = base_df["父馬名"].apply(lambda x: 1 if x in ('ゴールドアリュール', 'キンシャサノキセキ') else 0)
        base_df.loc[:, "マイル芝血統"] = base_df["父馬名"].apply(lambda x:1 if x in ('ディープインパクト', 'キングカメハメハ', 'ハーツクライ', 'ステイゴールド') else 0)
        base_df.loc[:, "ノーザンＦ"] = base_df["生産者名"].apply(lambda x: 1 if x=="ノーザンファーム" else 0)
        base_df.loc[:, "ニジンスキー系"] = base_df.apply(lambda x: 1 if x["父系統コード"] == '1102' or x["父馬名"] == 'アドマイヤムーン' else 0, axis=1)
        base_df.loc[:, "Ｐサンデー系"] = base_df["父馬名"].apply(
            lambda x: 1 if x in ('ステイゴールド', 'マツリダゴッホ', 'アグネスタキオン', 'キンシャサノキセキ') else 0)
        base_df.loc[:, "母父トニービン"] = base_df["母父馬名"].apply(lambda x:1 if x =="トニービン" else 0)
        base_df.loc[:, "グレイソヴリン系"] = base_df.apply(lambda x: 1 if x["父系統コード"] == '1302' or x["母父系統コード"] == '1302' else 0, axis=1)
        base_df.loc[:, "マッチェム系"] = base_df.apply(lambda x: 1 if x["父系統コード"] in ('2101', '2102', '2103', '2104', '2105') or x["母父系統コード"] == '2101' else 0, axis=1)
        base_df.loc[:, "父ルーラーシップ"] = base_df["父馬名"].apply(lambda x:1 if x == "ルーラーシップ" else 0)
        base_df.loc[:, "父ネオユニ短縮"] = base_df.apply(lambda x:1 if x["父馬名"] == "ネオユニヴァース" and x["距離増減"] <= -200 else 0, axis=1)
        base_df.loc[:, "父ロードカナロア"] = base_df["父馬名"].apply(lambda x:1 if x == "ロードカナロア" else 0)
        base_df.loc[:, "母父ディープ"] = base_df["母父馬名"].apply(lambda x:1 if x =="ディープインパクト" else 0)
        base_df.loc[:, "母父キンカメ"] = base_df["母父馬名"].apply(lambda x:1 if x =="キングカメハメハ" else 0)
        return base_df

    def get_droped_columns_raceuma_df(self, base_df):
        base_df.drop(['発走時間', '開催区分', 'データ区分', 'WIN5フラグ', '場名', 'ZENSO1_KYOSO_RESULT', 'ZENSO2_KYOSO_RESULT', 'ZENSO3_KYOSO_RESULT', 'ZENSO4_KYOSO_RESULT', 'ZENSO5_KYOSO_RESULT',
                      'ZENSO1_RACE_KEY', 'ZENSO2_RACE_KEY', 'ZENSO3_RACE_KEY', 'ZENSO4_RACE_KEY', 'ZENSO5_RACE_KEY',
                      '騎手コード', '調教師コード', '入厩年月日', '併せクラス', '血統登録番号', 'KAISAI_KEY',
                       'KYOSO_RESULT_KEY_1', '血統登録番号_1', 'NENGAPPI_1', '騎手コード_1', '調教師コード_1',
                       'KYOSO_RESULT_KEY_2', '血統登録番号_2', 'NENGAPPI_2', '騎手コード_2', '調教師コード_2',
                       'KYOSO_RESULT_KEY_3', '血統登録番号_3', 'NENGAPPI_3', '騎手コード_3', '調教師コード_3',
                       'KYOSO_RESULT_KEY_4', '血統登録番号_4', 'NENGAPPI_4', '騎手コード_4', '調教師コード_4',
                       'KYOSO_RESULT_KEY_5', '血統登録番号_5', 'NENGAPPI_5', '騎手コード_5', '調教師コード_5'], axis=1, inplace=True)
        return base_df

    def get_label_encoding_raceuma_df(self, base_df, index_list):
        categorical_feats = base_df.dtypes[base_df.dtypes == "object"].index.tolist()
        for val in index_list:
            if val in categorical_feats:
                categorical_feats.remove(val)
        for categ in categorical_feats:
            base_df.loc[:, categ] = mu.label_encoding(base_df[categ], "ru_" + categ, self.dict_folder).astype(str)
        return base_df

    ### race_df
    def get_create_feature_race_df(self, base_df, race_df):
        """ マージしたデータから特徴量を生成する """
        print("_create_feature")
        raceuma_df = base_df[["RACE_KEY", "UMABAN", "脚質", "距離適性", "父馬産駒連対平均距離", "母父馬産駒連対平均距離", "IDM", "テン指数",
                                   "ペース指数", "上がり指数", "位置指数", "ＩＤＭ結果_1", "テン指数結果_1", "上がり指数結果_1", "ペース指数結果_1", "レースＰ指数結果_1",
                                   "先行率_1", "追込率_1", "fa_1_1", "fa_2_1", "fa_3_1", "fa_4_1", "fa_5_1"]]
        raceuma_df.loc[:, "RACE_UMA_KEY"] = raceuma_df["RACE_KEY"] + raceuma_df["UMABAN"]
        raceuma_df.drop("UMABAN", axis=1, inplace=True)
        # https://qiita.com/daigomiyoshi/items/d6799cc70b2c1d901fb5
        es = ft.EntitySet(id="race")
        es.entity_from_dataframe(entity_id='race', dataframe=race_df, index="RACE_KEY")
        es.entity_from_dataframe(entity_id='raceuma', dataframe=raceuma_df, index="RACE_UMA_KEY")
        relationship = ft.Relationship(es['race']["RACE_KEY"], es['raceuma']["RACE_KEY"])
        es = es.add_relationship(relationship)
        print(es)
        # 集約関数
        aggregation_list = ['min', 'max', 'mean', 'skew', 'percent_true']
        transform_list = []
        # run dfs
        print("un dfs")
        feature_matrix, features_dfs = ft.dfs(entityset=es, target_entity='race', agg_primitives=aggregation_list,
                                              trans_primitives=transform_list, max_depth=2)
        print("_create_feature: feature_matrix", feature_matrix.shape)

        # 予想１番人気のデータを取得
        ninki_df = base_df.query("基準人気順位==1")[["RACE_KEY", "脚質", "距離適性", "上昇度", "激走指数", "蹄コード", "見習い区分", "枠番", "総合印", "ＩＤＭ印", "情報印", "騎手印",
                                                  "厩舎印", "調教印", "激走印", "展開記号", "輸送区分", "騎手期待単勝率", "騎手期待３着内率", "激走タイプ", "休養理由分類コード", "芝ダ障害フラグ",
                                                    "距離フラグ", "クラスフラグ", "転厩フラグ", "去勢フラグ", "乗替フラグ", "放牧先ランク", "厩舎ランク", "調教量評価", "仕上指数変化", "調教評価",
                                                    "IDM", "騎手指数", "情報指数", "総合指数", "人気指数", "調教指数", "厩舎指数", "テン指数", "ペース指数", "上がり指数", "位置指数", "追切指数", "仕上指数",
                                                    "ＩＤＭ結果_1", "ＩＤＭ結果_2"]].add_prefix("人気_").rename(columns={"人気_RACE_KEY":"RACE_KEY"})
        # 逃げ予想馬のデータを取得
        nige_df = base_df.query("展開記号=='1'")[["RACE_KEY", "脚質", "距離適性", "上昇度", "激走指数", "蹄コード", "見習い区分", "枠番", "総合印", "ＩＤＭ印", "基準人気順位", "輸送区分", "激走タイプ", "休養理由分類コード", "芝ダ障害フラグ",
                                                    "距離フラグ", "クラスフラグ", "転厩フラグ", "去勢フラグ", "乗替フラグ", "IDM", "騎手指数", "テン指数", "ペース指数", "上がり指数", "位置指数", "追切指数", "仕上指数",
                                                    "斤量_1", "テン指数結果_1", "上がり指数結果_1", "ペース指数結果_1", "レースＰ指数結果_1", "斤量_2", "テン指数結果_2", "上がり指数結果_2", "ペース指数結果_2", "レースＰ指数結果_2",
                                                    "先行率_1", "先行率_2", "距離", "距離_1"]].add_prefix("逃げ_").rename(columns={"逃げ_RACE_KEY":"RACE_KEY"})
        nige_df.loc[:, "逃げ_距離増減"] = nige_df["逃げ_距離"] - nige_df["逃げ_距離_1"]
        nige_df.drop(["逃げ_距離", "逃げ_距離_1"], axis=1, inplace=True)
        nige_ddf = nige_df.groupby("RACE_KEY")
        nige_df2 = nige_df.loc[nige_ddf["逃げ_テン指数"].idxmax(),: ]
        # 上がり最速予想馬のデータを取得
        agari_df = base_df.query("展開記号=='2'")[["RACE_KEY", "脚質", "距離適性", "上昇度", "激走指数", "蹄コード", "見習い区分", "枠番", "総合印", "ＩＤＭ印", "基準人気順位", "輸送区分", "激走タイプ", "休養理由分類コード", "芝ダ障害フラグ",
                                                    "距離フラグ", "クラスフラグ", "転厩フラグ", "去勢フラグ", "乗替フラグ", "IDM", "騎手指数", "テン指数", "ペース指数", "上がり指数", "位置指数", "追切指数", "仕上指数",
                                                    "斤量_1", "テン指数結果_1", "上がり指数結果_1", "ペース指数結果_1", "レースＰ指数結果_1", "斤量_2", "テン指数結果_2", "上がり指数結果_2", "ペース指数結果_2", "レースＰ指数結果_2",
                                                    "先行率_1", "先行率_2"]].add_prefix("上り_").rename(columns={"上り_RACE_KEY":"RACE_KEY"})

        base_df = pd.merge(feature_matrix, nige_df2, on="RACE_KEY", how="left")
        base_df = pd.merge(base_df, agari_df, on="RACE_KEY", how="left")
        base_df = pd.merge(base_df, ninki_df, on="RACE_KEY")
        return base_df

    def get_droped_columns_race_df(self, base_df):
        print(base_df.iloc[0])
        #base_df = base_df.drop("KAISAI_KEY", axis=1)
        return base_df

    def get_label_encoding_race_df(self, base_df, index_list):
        categorical_feats = base_df.dtypes[base_df.dtypes == "object"].index.tolist()
        for val in index_list:
            if val in categorical_feats:
                categorical_feats.remove(val)
        for categ in categorical_feats:
            base_df.loc[:, categ] = mu.label_encoding(base_df[categ], "rc_" + categ, self.dict_folder).astype(str)
        return base_df

    def get_target_variable_df(self, result_df, version_str):
        if version_str == 'win':
            result_df['WIN_FLAG'] = result_df['着順'].apply(lambda x: 1 if x == 1 else 0)
            result_df['JIKU_FLAG'] = result_df.apply(lambda x: 1 if x['着順'] in (1, 2) and x['馬連払戻１'] >= 5000 else 0, axis=1)
            result_df['ANA_FLAG'] = result_df.apply(
                lambda x: 1 if x['着順'] in (1, 2, 3) and x['複勝'] >= 500 else 0, axis=1)
            result_df = result_df[["RACE_KEY", "UMABAN", "NENGAPPI", "WIN_FLAG", "JIKU_FLAG", "ANA_FLAG"]].copy()
            return result_df
        elif version_str == 'win5':
            result_df = result_df.query("WIN5フラグ in ('1','2','3','4','5')")
            result_df['WIN_FLAG'] = result_df['着順'].apply(lambda x: 1 if x == 1 else 0)
            result_df = result_df[["RACE_KEY", "UMABAN", "NENGAPPI", "WIN_FLAG"]].copy()
            return result_df
        elif version_str == 'haito':
            result_df['UMAREN_ARE'] = result_df['馬連払戻１'].apply(lambda x: 0 if x < 3000 else 1)
            result_df['UMATAN_ARE'] = result_df['馬連払戻１'].apply(lambda x: 0 if x < 5000 else 1)
            result_df['SANRENPUKU_ARE'] = result_df['馬連払戻１'].apply(lambda x: 0 if x < 5000 else 1)
            result_df = result_df[["RACE_KEY", "UMABAN", "NENGAPPI", "UMAREN_ARE", "UMATAN_ARE", "SANRENPUKU_ARE"]].copy()
            return result_df
        elif version_str == 'raptype':
            result_df['RAP_TYPE'] = result_df['RAP_TYPE'].apply(lambda x: mu.encode_rap_type(x))
            result_df['TRACK_BIAS_ZENGO'] = result_df['TRACK_BIAS_ZENGO'].apply(lambda x: mu.encode_zengo_bias(x))
            result_df['TRACK_BIAS_UCHISOTO'] = result_df['TRACK_BIAS_UCHISOTO'].apply(lambda x: mu.encode_uchisoto_bias(x))
            result_df['PRED_PACE'] = result_df['レースペース流れ'].apply(lambda x: mu.encode_race_pace(x))
            result_df = result_df[["RACE_KEY", "UMABAN", "NENGAPPI", "RAP_TYPE", "TRACK_BIAS_ZENGO", "TRACK_BIAS_UCHISOTO", "PRED_PACE"]].copy()
            return result_df
        elif version_str == 'nigeuma':
            result_df['NIGEUMA'] = result_df['レース脚質'].apply(lambda x: 1 if x == '1' else 0)
            result_df['AGARI_SAISOKU'] = result_df['上がり指数結果順位'].apply(lambda x: 1 if x == 1 else 0)
            result_df['TEN_SAISOKU'] = result_df['テン指数結果順位'].apply(lambda x: 1 if x == 1 else 0)
            result_df = result_df[["RACE_KEY", "UMABAN", "NENGAPPI", "NIGEUMA", "AGARI_SAISOKU", "TEN_SAISOKU"]].copy()
            return result_df
