import nibabel as nib
import numpy as np
import csv
import pandas as pd
import os
from sklearn.svm import SVR


class PEER:
    def __init__(self, eye_mask_path, use_gsr, monitor_width, monitor_height, output_dir):
        self.eye_mask_path = eye_mask_path
        self.use_gsr = use_gsr
        self.monitor_width = monitor_width
        self.monitor_height = monitor_height
        self.output_dir = output_dir
        self.X = []
        self.Y = []
        self.x_fixations = []
        self.y_fixations = []

    def load_data(self, _filepath):
        """
        Loads fMRI data

        Parameters
        ----------
        _filepath : string
            Pathname of the NIfTI file used to train a model or predict eye movements

        Returns
        -------
        _data : np.ndarray
            4D numpy array containing fMRI data

        """
        print('\nLoad Data')
        print('====================================================')
        nib_format = nib.load(_filepath)
        _data = nib_format.get_data()

        print('Training data Loaded')

        return _data

    def save_data(self, xfix, yfix, name="subj"):
        fixation_df = pd.DataFrame({'X': xfix, 'Y': yfix})
        fixation_df.to_csv(os.path.join(self.output_dir, name + '_fixation.csv'), index_label="scan_num")

    def standardize_data(self, data):

        import time
        volumes = data.shape[3]
        start_time = time.time()
        mean_data = np.mean(data, axis=3)
        std_data = np.std(data, axis=3)
        std_data[std_data == 0] = 1
        for i in range(volumes):
            data[:, :, :, i] = (data[:, :, :, i] - mean_data) / std_data
        elapsed_time = time.time() - start_time
        print("Elapsed time: " + str(elapsed_time))
        return data

    def global_signal_regression(self, _data, _eye_mask_path):

        """
        Performs global signal regression

        Parameters
        ----------
        _data : float
            Data from an fMRI scan as a 4D numpy array
        _eye_mask_path :
            Pathname for the eye mask NIfTI file (the standard MNI152 2mm FSL template is used for the linked preprint)
        Returns
        -------
        _data :
            4D numpy array containing fMRI data after global signal regression

        """
        print('\nGlobal Signal Regression')
        print('====================================================')
        eye_mask = nib.load(_eye_mask_path).get_data()

        global_mask = np.array(eye_mask, dtype=bool)

        regressor_map = {'constant': np.ones((_data.shape[3], 1))}
        regressor_map['global'] = _data[global_mask].mean(0)

        X = np.zeros((_data.shape[3], 1))
        csv_filename = ''

        for rname, rval in regressor_map.items():
            X = np.hstack((X, rval.reshape(rval.shape[0], -1)))
            csv_filename += '_' + rname

        X = X[:, 1:]

        Y = _data[global_mask].T
        B = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y)

        Y_res = Y - X.dot(B)

        _data[global_mask] = Y_res.T

        print('GSR completed.')

        return _data

    def motion_scrub(self, file_path, _motion_threshold):
        """
        Determines volumes with high motion artifact

        Parameters
        ----------
        _ms_filename : string
            Pathname of the CSV file containing the framewise displacement per time point for a given fMRI scan
        _data_dir : string
            Pathname of the directory containing data
        _motion_threshold  : float
            Threshold for high motion (framewise displacement, defined by Power et al. 2012)

        Returns
        -------
        _removed_indices : int
            List of volumes to remove for motion scrubbing

        """
        print(str('\nMotion Scrubbing').format(_motion_threshold))
        print('====================================================')
        with open(file_path, 'r') as f:
            reader = csv.reader(f)
            censor_pre = [x for x in reader]

        nuissance_vector = [float(x) for x in censor_pre[0]]

        _removed_indices = [i for i, x in enumerate(nuissance_vector) if x >= float(_motion_threshold)]

        return _removed_indices

    def default_algorithm(self):
        return SVR(kernel='linear', C=100, epsilon=0.01, verbose=2)

    def predict_fixations(self, _xmodel, _ymodel, _data):
        """
        Predict fixations

        Parameters
        ----------
        _xmodel :
            SVR model to estimate eye movements in the x-direction
        _ymodel :
            SVR model to estimate eye movements in the y-direction
        _data :
            4D numpy array containing fMRI data used to predict eye movements (e.g., movie data)

        Returns
        -------
        _x_fix : float
            List of predicted fixations in the x-direction
        _y_fix : float
            List of predicted fixations in the y-direction

        """

        print('\nPredicting Fixations')
        print('====================================================')

        print('Fixations saved to specified output directory.')
        _x_fix = _xmodel.predict(_data)
        _y_fix = _ymodel.predict(_data)

        return _x_fix, _y_fix

    def estimate_em(self, _x_fix, _y_fix):
        """
        Parameters
        ----------
        _x_fix : np.array
            List of predicted fixations in the x-direction
        _y_fix : np.array
            List of predicted fixations in the y-direction
        """
        x_em = []
        y_em = []

        for num in range(len(_x_fix) - 1):
            x_em.append(abs(_x_fix[num] - _x_fix[num + 1]))
            y_em.append(abs(_y_fix[num] - _y_fix[num + 1]))

        return x_em, y_em

    def prepare_data_for_svr(self, _data, _removed_time_points, _stimulus_path):
        """
        Preprocess fMRI data prior to SVR model generation

        Parameters
        ----------
        _data : float
            4D numpy array containing fMRI data after global signal regression
        _removed_time_points : int
            List of volumes to remove for motion scrubbing
        _stimulus_path : string
            Pathname of the PEER calibration scan stimuli
        monitor_width: int
            monitor width, in pixels
        monitor_height:int
            monitor_height, in pixels
        Returns
        -------
        _processed_data : float
            List of numpy arrays, where each array contains the averaged intensity values for each calibration point
        _calibration_points_removed : int
            List of calibration points removed if all volumes for a given calibration point were high motion

        """

        if _removed_time_points is not None:
            print(str('The {}th volume(s) were removed.').format(_removed_time_points))
        else:
            _removed_time_points = []

        _processed_data = []
        _calibration_points_removed = []

        for num in range(int(_data.shape[3] / 5)):
            vol_set = [x for x in np.arange(num * 5, (num + 1) * 5) if x not in _removed_time_points]
            if len(vol_set) != 0:
                _processed_data.append(np.average(_data[:, :, :, vol_set], axis=3).ravel())
            else:
                _calibration_points_removed.append(num)

        if (_calibration_points_removed) and (_removed_time_points):
            print(str('The {}th calibration point(s) were removed.').format(_calibration_points_removed))
        elif (not _calibration_points_removed) and (_removed_time_points):
            print(str('No calibration points were removed.'))

        fixations = pd.read_csv(_stimulus_path)
        x_targets = np.repeat(np.array(fixations['pos_x']), 1) * self.monitor_width / 2
        y_targets = np.repeat(np.array(fixations['pos_y']), 1) * self.monitor_height / 2

        x_targets = list(np.delete(np.array(x_targets), _calibration_points_removed))
        y_targets = list(np.delete(np.array(y_targets), _calibration_points_removed))

        return _processed_data, x_targets, y_targets

    def load_peer(self, data_paths, stimulus_paths, perform_motion_scrub=False,
                  motion_scrub_path="", motion_threshold=-1):
        processed_data = []
        xtarget_list = []
        ytarget_list = []
        for ind, data_path in enumerate(data_paths):
            print('\nGenerating model for participant #{}'.format(ind + 1))
            print('====================================================')
            data = self.preprocess_train_data(data_path)
            if perform_motion_scrub:
                removed_indices = self.motion_scrub(motion_scrub_path, motion_threshold)
            else:
                removed_indices = None
            ravelled_data, xtargets, ytargets = self.prepare_data_for_svr(data, removed_indices, stimulus_paths[ind])
            processed_data.extend(ravelled_data)
            xtarget_list.extend(xtargets)
            ytarget_list.extend(ytargets)

        self.X = processed_data
        self.Y = (xtarget_list, ytarget_list)
        return processed_data, xtarget_list, ytarget_list

    def preprocess_train_data(self, data_path):
        data = self.load_data(data_path)
        data = self.mask_data(data)
        data = self.standardize_data(data)
        if self.use_gsr:
            data = self.global_signal_regression(data, self.eye_mask_path)
        return data

    def mask_data(self, data):
        eye_mask = nib.load(self.eye_mask_path).get_data()
        for vol in range(data.shape[3]):
            output = np.multiply(eye_mask, data[:, :, :, vol])
            data[:, :, :, vol] = output

        return data

    # TODO:rename "peer_algorithm"
    def train_peer(self, x=None, y=None, peer_algorithm=None):
        if x is None:
            x = self.X
        if y is None:
            y = self.Y
        if peer_algorithm is None:
            peer_algorithm = self.default_algorithm

        _xmodel = peer_algorithm()
        _xmodel.fit(x, y[0])

        _ymodel = peer_algorithm()
        _ymodel.fit(x, y[1])

        return _xmodel, _ymodel

    def test_peer(self, data_list, xmodel, ymodel):
        xfix_list = []
        yfix_list = []
        for ind, data_dir in enumerate(data_list):
            print(('\nPredicting fixations for participant #{}').format(ind + 1))
            print('====================================================')
            data = self.preprocess_train_data(data_dir)
            raveled_data = [data[:, :, :, vol].ravel() for vol in np.arange(data.shape[3])]
            del data
            x_fix, y_fix = self.predict_fixations(xmodel, ymodel, raveled_data)
            xfix_list.append(x_fix)
            yfix_list.append(y_fix)
        self.x_fixations = xfix_list
        self.y_fixations = yfix_list
        return xfix_list, yfix_list


