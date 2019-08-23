from PEER import PEER
import nibabel as nib
import numpy as np


class PEER_single_slice(PEER):
    def mask_data(self, data):
        eye_mask = nib.load(self.eye_mask_path).get_data()
        lis = []
        slice_order = self.get_slice_order(60, 6, 3)

        for ind in range(eye_mask.shape[2]):
            if np.any(eye_mask[:, :, ind]):
                lis.append(ind)
        # middle_slice = (max(lis) - min(lis))//2
        # eye_mask[:, :, 0:middle_slice] = 0
        # eye_mask[:, :, middle_slice+1:] = 0
        masked = []
        for vol in range(data.shape[3]):
            output = np.multiply(eye_mask, data[:, :, :, vol])
            # plt.imshow(output[:, :, 14])
            # plt.savefig('eye_mask' + str(vol) + '.png')
            data[:, :, :, vol] = output
            masked.append(data[:, :, :, vol][eye_mask == 1])
        arr = self.get_slice_timeseries(data, slice_order)

        return data

    def get_slice_order(self, slices, multiband, inca):
        axial_slice_indices = list(range(0, 60))
        stride = slices // multiband
        slice_order = []
        for i in range(stride):
            loc = (i * inca) % stride
            slices_at_time = list(range(loc, slices, stride))

            slice_indices = [axial_slice_indices[s] for s in slices_at_time]
            slice_order.append(slice_indices)

        return slice_order

    def get_slice_timeseries(self, data, slice_order, slice_timing=0.08):
        timepoints = []
        timings = []
        timing_ind = 0
        data = data.astype(float)
        # data[data < np.percentile(np.median(data, 3), 98)] = np.nan

        data[data <= 0] = np.nan
        for ind in range(data.shape[3]):
            slices = []
            for slice in slice_order:
                timings.append(timing_ind * slice_timing)
                timing_ind += 1
                timepoints.append(data[:, :, slice, ind].ravel())

            # timepoints.append(slices)

        arr = np.array(timepoints)
        return arr

    """
    #arr = arr[10:, :]
    import matplotlib.pyplot as plt
    import scipy.stats as stats
    arr = stats.zscore(arr)

    #plt.plot(arr)
    #plt.legend(list(range(9)))
    positions = []
    from scipy.interpolate import interp1d
    import scipy.signal
    #x = np.linspace(0, 134, num=1350, endpoint=True)
    #arr = scipy.signal.resample(arr, 1350)
    
    
    for i in range(arr.shape[1]):
        for j in range(arr.shape[1]):

            c = np.correlate(arr[:, i], arr[:, j], "full")
            x = np.argmax(c)
            pos = (np.argmax(c) - (len(c) // 2))
            positions.append(pos)
    

    #plt.plot(arr[100:, :])
    plt.plot(np.mean(arr[:, 0:5], 1)[100:])
    plt.plot(np.mean(arr[:, 5:], 1)[100:])
    plt.legend(list(range(10)))
    plt.show()
    print('pass')
    """
