from create_peer import peer_train
from estimate_eyemove import peer_test
import os
import peer_func as pr
import pandas as pd
from scipy import stats

if __name__ == '__main__':
    os.chdir(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))
    test_path_dir = os.path.join(os.getcwd(), 'data')
    print(test_path_dir)
    data_list = [os.path.abspath(os.path.join(test_path_dir, x)) for x in os.listdir(test_path_dir) if
                 not x.startswith('.') and
                 os.path.isdir(os.path.join(test_path_dir, x))]
    data_list.sort()
    subj_list = ['subj1', 'subj2', 'subj3', 'subj4', 'subj5']
    x_corr = []
    y_corr = []
    for ind, data in enumerate(data_list):

        leave_out = subj_list[:ind] + subj_list[ind+1:]
        xmodel, ymodel, configs = peer_train(test_path_dir, 'peer/stim_vals.csv',
                                             leave_out)
        pr.save_model(xmodel, ymodel, configs['train_file'], configs['use_ms'], configs['use_gsr'], test_path_dir)
        xfix_list, yfix_list = peer_test([data], [test_path_dir])
        eyetracker = pd.read_csv(os.path.join(data, 'outputs', 'ds_eyetracker.csv'))
        r_x, coef = stats.pearsonr(eyetracker['X'], pd.Series(xfix_list[0][0:len(eyetracker.index)]))
        x_corr.append(r_x)
        print(r_x)
        r_y, coef = stats.pearsonr(eyetracker['Y'], pd.Series(yfix_list[0][0:len(eyetracker.index)]))
        y_corr.append(r_y)
        print(r_y)
    print(x_corr)
    print(y_corr)
    print("DONE")
