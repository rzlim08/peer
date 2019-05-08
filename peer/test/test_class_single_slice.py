import os
from PEER_single_slice import PEER_single_slice
import pandas as pd
import numpy as np

def initialize_class(dir_path):
    peer_class = PEER_single_slice(
        eye_mask_path='/usr/share/fsl/5.0/data/standard/MNI152_T1_2mm_eye_mask.nii.gz',
        use_gsr=False,
        monitor_width=1680,
        monitor_height=1050,
        output_dir=os.path.join(dir_path, 'output')
    )
    return peer_class


def compare_fixations(fix, cmp_array):
        if len(fix) != len(cmp_array):
            return False
        diff = [abs(selfx-otherx) > 0.01 for selfx, otherx in zip(fix, cmp_array)]
        return not any(diff)


def test():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    data_path = os.path.join(dir_path, 'data')
    stim_vals = os.path.join(data_path, 'stim_vals.csv')
    peer_class = initialize_class(dir_path)
    # Get peer fMRI file
    fmri_path = os.path.join(data_path, 'subj1', 'PEER001.nii.gz')
    peer_class.load_peer([fmri_path], [stim_vals])
    xmodel, ymodel = peer_class.train_peer()

    # Get peer test file
    test_path = os.path.join(data_path, 'subj1', 'PEER002.nii.gz')

    peer_class.test_peer([test_path], xmodel, ymodel)
    test = pd.read_csv(os.path.join(data_path, 'subj1', 'outputs', 'subj1_fixation.csv'))
    if not compare_fixations(peer_class.x_fixations[0], test['X']):

        print(np.corrcoef(peer_class.x_fixations[0], test['X']))
        print("FAIL on X")
    if not compare_fixations(peer_class.y_fixations[0], test['Y']):
        print(np.corrcoef(peer_class.y_fixations[0], test['Y']))
        print("FAIL on Y")
    print("DONE")


if __name__ == '__main__':
    test()



