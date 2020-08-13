from modules.luigi_tasks import Create_learning_model
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
        start_date = '2019/01/01'
        end_date = '2019/01/31'
        #start_date = '2019/01/30'
        #end_date = '2019/02/10'
    else:
        start_date = '2012/01/01'
        end_date = '2019/12/31'
    target_list = [{"version_str": "win", "model_name": "raceuma"},
                   {"version_str": "win5", "model_name": "raceuma"},
                   {"version_str": "nigeuma", "model_name": "raceuma"},
                   {"version_str": "raptype", "model_name": "race"},
                   {"version_str": "haito", "model_name": "race"}
                   ]
    mock_flag = False
    dict_path = mc.return_base_path(test_flag)

    for target in target_list:
        model_name = target["model_name"]
        version_str = target["version_str"]
        print("model_name:" + model_name , " version_str:" + version_str)
        intermediate_folder = dict_path + 'intermediate/' + model_name + '/'
        print(intermediate_folder)

        skproc = SkProc(version_str, start_date, end_date, model_name, mock_flag, test_flag)

        luigi.build([Create_learning_model(start_date=start_date, end_date=end_date, skproc=skproc,intermediate_folder=intermediate_folder)],
                    local_scheduler=True)