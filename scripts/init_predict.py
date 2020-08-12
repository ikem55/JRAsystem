from modules.luigi_tasks import Calc_predict_data, Create_target_file, Sub_download_jrdb_file
from modules.sk_proc import SkProc
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
        start_date = '2020/01/01'
        end_date = '2020/01/11'
        term_start_date = '20200101'
        term_end_date = '20200111'
    else:
        start_date = '2020/01/01'
        end_date = (dt.now() + timedelta(days=1)).strftime('%Y/%m/%d')
        term_start_date = '20200101'
        term_end_date = (dt.now() + timedelta(days=1)).strftime('%Y%m%d')
#        start_date = '2020/07/24'
#        end_date = '2020/8/2'
#        term_start_date = '20200601'
#        term_end_date = '20200802'
    target_list = [{"version_str": "win", "model_name": "raceuma"},
                   {"version_str": "nigeuma", "model_name": "raceuma"},
                   {"version_str": "raptype", "model_name": "race"},
                   {"version_str": "haito", "model_name": "race"}]
    mock_flag = False
    export_mode = False
    dict_path = mc.return_base_path(test_flag)

    intermediate_folder = dict_path + 'intermediate/download_jrdb/'
    if not test_flag:
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

    if not test_flag:
        intermediate_folder = dict_path + 'intermediate/target_file/'
        luigi.build([Create_target_file(start_date=start_date, end_date=end_date, term_start_date=term_start_date, term_end_date=term_end_date,
                                        intermediate_folder=intermediate_folder, test_flag=test_flag)],local_scheduler=True)