from modules.luigi_tasks import Calc_predict_data, Create_target_file, Sub_download_jrdb_file
from modules.sk_proc import SkProc
from modules.download import JrdbDownload
import my_config as mc
import sys
from distutils.util import strtobool
import luigi
from datetime import datetime as dt
from datetime import timedelta

if __name__ == "__main__":
    args = sys.argv
    test_flag = strtobool(args[1])
    print("tst_flag:" + args[1])

    if test_flag:
        start_date = '2019/01/11'
        end_date = '2019/1/20'
        term_start_date = '20190101'
        term_end_date = '20190120'
    else:
        start_date = (dt.now() + timedelta(days=-90)).strftime('%Y/%m/%d')
        end_date = (dt.now() + timedelta(days=1)).strftime('%Y/%m/%d')
        term_start_date = (dt.now() + timedelta(days=-19)).strftime('%Y%m%d')
        term_end_date = (dt.now() + timedelta(days=1)).strftime('%Y%m%d')
    target_list = [{"version_str": "win", "model_name": "raceuma"},
                   {"version_str": "nigeuma", "model_name": "raceuma"},
                   {"version_str": "raptype", "model_name": "race"},
                   {"version_str": "haito", "model_name": "race"}]
    mock_flag = False
    export_mode = False
    dict_path = mc.return_base_path(test_flag)

    intermediate_folder = dict_path + 'intermediate/download_jrdb/'
    luigi.build([Sub_download_jrdb_file(end_date=end_date, intermediate_folder=intermediate_folder)],
                local_scheduler=True)

    for target in target_list:
        model_name = target["model_name"]
        version_str = target["version_str"]
        print("model_name:" + model_name , " version_str:" + version_str)
        intermediate_folder = dict_path + 'intermediate/' + model_name + '/'
        print(intermediate_folder)

        skproc = SkProc(version_str, start_date, end_date, model_name, mock_flag, test_flag)

        luigi.build([Calc_predict_data(start_date=start_date, end_date=end_date, skproc=skproc,intermediate_folder=intermediate_folder, export_mode=export_mode)],
                    local_scheduler=True)

    intermediate_folder = dict_path + 'intermediate/target_file/'
    luigi.build([Create_target_file(start_date=start_date, end_date=end_date, term_start_date=term_start_date, term_end_date=term_end_date,
                                    intermediate_folder=intermediate_folder, test_flag=test_flag)],local_scheduler=True)

    jrdb = JrdbDownload()
    jrdb.delete_temp_text_file()