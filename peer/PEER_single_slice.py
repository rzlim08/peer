from PEER import PEER
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt


class PEER_single_slice(PEER):

    def mask_data(self, data):
        eye_mask = nib.load(self.eye_mask_path).get_data()
        lis = []

        for ind in range(eye_mask.shape[2]):
            if np.any(eye_mask[:,:,ind]):
                lis.append(ind)
        middle_slice = (max(lis) - min(lis))//2
        eye_mask[:, :, 0:middle_slice] = 0
        eye_mask[:, :, middle_slice+1:] = 0
        masked = []
        for vol in range(data.shape[3]):
            output = np.multiply(eye_mask, data[:, :, :, vol])
            #plt.imshow(output[:, :, 14])
            #plt.savefig('eye_mask' + str(vol) + '.png')
            data[:, :, :, vol] = output
            masked.append(data[:,:,:,vol][eye_mask==1])

        return data
