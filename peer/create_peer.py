#!/usr/bin/env python
"""
Script used on the command line to create SVR models for the PEER method

Authors:
    - Jake Son, 2017-2018  (jake.son@childmind.org)

"""

import os
import numpy as np
import nibabel as nib
import peer_func as pr

## TODO: refactor, combine top_data_dir and data_list into one parameter
def peer_train(top_data_dir, stimulus_path, data_list):
    configs = pr.load_config()
    all_processed_data = []
    xtarget_list = []
    ytarget_list = []
    for i, dataset in enumerate(data_list):
        data_dir = os.path.abspath(os.path.join(top_data_dir, dataset))
        print(('\nGenerating model for participant #{}').format(i + 1))
        print('====================================================')
        filepath = os.path.join(data_dir, configs['train_file'])
        data = pr.load_data(filepath)

        eye_mask = nib.load(configs['eye_mask_path']).get_data()
        for vol in range(data.shape[3]):
            output = np.multiply(eye_mask, data[:, :, :, vol])
            data[:, :, :, vol] = output



        if int(configs['use_gsr']):
            data = pr.global_signal_regression(data, configs['eye_mask_path'])
        if int(configs['use_ms']):
            removed_indices = pr.motion_scrub(configs['motion_scrub'], data_dir, configs['motion_threshold'])
        else:
            removed_indices = None

        processed_data, xtargets, ytargets = pr.prepare_data_for_svr(data, removed_indices, stimulus_path)

        print('\nTrain PEER')
        print('====================================================')
        all_processed_data.extend(processed_data)
        xtarget_list.extend(xtargets)
        ytarget_list.extend(ytargets)
    xmodel, ymodel = pr.train_model(all_processed_data, xtarget_list, ytarget_list)
    return xmodel, ymodel, configs


if __name__ == "__main__":
    project_dir, top_data_dir, stimulus_path = pr.scaffolding()
    os.chdir(project_dir)
    data_list = [x for x in os.listdir(top_data_dir) if not x.startswith('.')]
    xmodel, ymodel, configs = peer_train(top_data_dir, stimulus_path, data_list)
    pr.save_model(xmodel, ymodel, configs['train_file'], configs['use_ms'], configs['use_gsr'], top_data_dir)
