from create_peer import peer_train
from estimate_eyemove import peer_test
import peer_func as pr
import os
import pandas as pd

if __name__ == '__main__':
    test_path_dir = os.path.join(os.getcwd(), 'data')
    print(test_path_dir)
    os.chdir(os.path.abspath(os.path.join(os.getcwd(), os.pardir, os.pardir)))
    xmodel, ymodel, configs = peer_train(test_path_dir, 'peer/stim_vals.csv', ['subj1'])
    pr.save_model(xmodel, ymodel, configs['train_file'], configs['use_ms'], configs['use_gsr'], test_path_dir)
    data_list = [os.path.abspath(os.path.join(test_path_dir, x)) for x in os.listdir(test_path_dir) if not x.startswith('.') and
                                                            os.path.isdir(os.path.join(test_path_dir, x))]
    xfix_list, yfix_list = peer_test(data_list, [test_path_dir])
    test_fixation = os.path.join(test_path_dir, 'subj1', 'outputs', 'subj1_fixation.csv')
    test = pd.read_csv(test_fixation)

    if ((test['X'] - pd.Series(xfix_list[0])).abs() > 0.001).any():
        print("FAILED")
        exit(1)

    if ((test['Y'] - pd.Series(yfix_list[0])).abs() > 0.001).any():
        print("FAILED")
        exit(1)

    print("Done and PASSED!")