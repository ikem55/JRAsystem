from modules.luigi_tasks import Calc_predict_data
from modules.sk_proc import SkProc
import my_config as mc
import sys
from distutils.util import strtobool
import luigi

if __name__ == "__main__":
    args = sys.argv
    test_flag = strtobool(args[1])
    print("tst_flag:" + args[1])

    if test_flag:
        start_date = '2019/01/30'
        end_date = '2019/02/10'
    else:
        start_date = '2020/01/01'
        end_date = '2020/6/30'
    target_list = [{"version_str": "win", "model_name": "raceuma"},
                   {"version_str": "nigeuma", "model_name": "raceuma"},
                   {"version_str": "raptype", "model_name": "race"},
                   {"version_str": "haito", "model_name": "race"}]
    mock_flag = False
    export_mode = False
    dict_path = mc.return_base_path(test_flag)

    for target in target_list:
        model_name = target["model_name"]
        version_str = target["version_str"]
        print("model_name:" + model_name , " version_str:" + version_str)
        intermediate_folder = dict_path + 'intermediate/' + model_name + '_' + version_str + '/'
        print(intermediate_folder)

        skproc = SkProc(version_str, start_date, end_date, model_name, mock_flag, test_flag)

        luigi.build([Calc_predict_data(start_date=start_date, end_date=end_date, skproc=skproc,intermediate_folder=intermediate_folder, export_mode=export_mode)],
                    local_scheduler=True)