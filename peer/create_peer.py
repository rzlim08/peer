#!/usr/bin/env python
"""
Script used on the command line to create SVR models for the PEER method

Authors:
    - Jake Son, 2017-2018  (jake.son@childmind.org)

"""

import os
import numpy as np
import nibabel as nib
import time
import peer_func as pr

if __name__ == "__main__":

    project_dir, top_data_dir, stimulus_path = pr.scaffolding()

    os.chdir(project_dir)
    all_processed_data = []
    all_removed_points = []
    for i, dataset in enumerate([x for x in os.listdir(top_data_dir) if not x.startswith('.')]):

        data_dir = os.path.abspath(os.path.join(top_data_dir, dataset))

        output_dir = os.path.abspath(os.path.join(data_dir, 'outputs'))

        print(('\nGenerating model for participant #{}').format(i+1))
        print('====================================================')

        configs = pr.load_config()

        filepath = os.path.join(data_dir, configs['train_file'])

        print('\nLoad Data')
        print('====================================================')

        eye_mask_path = configs['eye_mask_path']
        eye_mask = nib.load(eye_mask_path).get_data()

        data = pr.load_data(filepath)

        for vol in range(data.shape[3]):
            output = np.multiply(eye_mask, data[:, :, :, vol])
            data[:, :, :, vol] = output
        volumes = data.shape[3]
        start_time = time.time()
        mean_data = np.mean(data, axis=3)
        std_data = np.std(data, axis=3)
        std_data[std_data == 0] = 1

        for i in range(volumes):
            data[:, :, :, i] = (data[:, :, :, i]-mean_data)/std_data
        elapsed_time = time.time() - start_time
        print("Elapsed time: " + str(elapsed_time))

        if int(configs['use_gsr']):

            print('\nGlobal Signal Regression')
            print('====================================================')

            data = pr.global_signal_regression(data, eye_mask_path)

        if int(configs['use_ms']):

            thresh = configs['motion_threshold']

            print(str('\nMotion Scrubbing').format(thresh))
            print('====================================================')

            ms_filename = configs['motion_scrub']
            removed_indices = pr.motion_scrub(ms_filename, data_dir, thresh)
        else:
            removed_indices = None

        processed_data, calibration_points_removed = pr.prepare_data_for_svr(data, removed_indices, eye_mask_path)

        print('\nTrain PEER')
        print('====================================================')
        all_processed_data.append(processed_data)
        all_removed_points.append(calibration_points_removed)

    xmodel, ymodel = pr.train_model([ll for sublist in all_processed_data for ll in sublist], [ll for sublist in all_removed_points for ll in sublist], stimulus_path)

    pr.save_model(xmodel, ymodel, configs['train_file'], configs['use_ms'], configs['use_gsr'], output_dir)

    print('\n')
