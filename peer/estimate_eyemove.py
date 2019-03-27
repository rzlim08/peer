#!/usr/bin/env python
"""
Script used on the command line to estimate eye movements

Authors:
    - Jake Son, 2017-2018  (jake.son@childmind.org)

"""

import os
import numpy as np
import nibabel as nib
import peer_func as pr


def peer_test(data_list, output_dirs):
    configs = pr.load_config()
    xfix_list = []
    yfix_list = []
    for i, data_dir in enumerate(data_list):

        print(('\nPredicting fixations for participant #{}').format(i + 1))
        print('====================================================')
        filepath = os.path.join(data_dir, configs['test_file'])

        data = pr.load_data(filepath)
        eye_mask = nib.load(configs['eye_mask_path']).get_data()
        for vol in range(data.shape[3]):
            output = np.multiply(eye_mask, data[:, :, :, vol])
            data[:, :, :, vol] = output

        data = pr.standardize_data(data)

        if int(configs['use_gsr']):
            data = pr.global_signal_regression(data, configs['eye_mask_path'])

        raveled_data = [data[:, :, :, vol].ravel() for vol in np.arange(data.shape[3])]
        del data
        xmodel, ymodel, xmodel_name, ymodel_name = pr.load_model(output_dirs[i])
        x_fix, y_fix = pr.predict_fixations(xmodel, ymodel, raveled_data)
        xfix_list.append(x_fix)
        yfix_list.append(y_fix)
        x_fixname, y_fixname = pr.save_fixations(x_fix, y_fix, xmodel_name, ymodel_name, output_dirs[i])

        print('\nEstimating Eye Movements')
        print('====================================================')
        pr.estimate_em(x_fix, y_fix, x_fixname, y_fixname, output_dirs[i])

        print('Eye movements saved to specified output directory.')

    return xfix_list, yfix_list


# TODO: add "output" list to this function so you can pass in a list  of outputs
if __name__ == "__main__":
    project_dir, top_data_dir, stimulus_path = pr.scaffolding()
    os.chdir(project_dir)
    data_list = [os.path.abspath(os.path.join(top_data_dir, x)) for x in os.listdir(top_data_dir) if not x.startswith('.') and
                                                            os.path.isdir(os.path.join(top_data_dir, x))]
    output_dirs = [os.path.abspath(os.path.join(data_dir, 'outputs')) for data_dir in data_list]
    x_fix, y_fix = peer_test(data_list, output_dirs)

    print('\n')
