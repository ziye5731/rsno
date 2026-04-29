import os
import torch
import random
import tifffile as tiff

from .utils import SSR_dataset

    

class AVIRIS_scenes_SSR(SSR_dataset):
    '''
    AVIRIS dataset for spectral super resolution, which is a collection of scenes and is made by myself.
    Args:
        data_dir: str, path to the dataset folder, inside which there are many .tif files.
        srf_dir: str, path to the camSpecSensitivity database folder, which is obtained from https://www.gujinwei.org/research/camspec/.
        wvlgth_dir: str, path to the .pt file, which contains the wavelength information.
        crop_size: int, size of the crop.
    '''
    def __init__(self, data_dir, srf_dir, crop_size, wvlgth_dir=None, absolute_scale=True):
        super().__init__(data_dir, srf_dir, crop_size)
        self.wvlgth_dir = wvlgth_dir
        if self.wvlgth_dir is not None:
            self.coord = torch.load(self.wvlgth_dir)/1000.0  # convert to micrometer
        else:
            self.coord = torch.tensor([
                365.9298, 375.5940, 385.2625, 394.9355, 404.6129, 414.2946, 423.9808,
                433.6713, 443.3662, 453.0655, 462.7692, 472.4773, 482.1898, 491.9066,
                501.6279, 511.3535, 521.0836, 530.8180, 540.5568, 550.3000, 560.0477,
                569.7996, 579.5560, 589.3168, 599.0819, 608.8515, 618.6254, 628.4037,
                638.1865, 647.9736, 657.7651, 667.5610, 655.2923, 665.0994, 674.9012,
                684.6979, 694.4894, 704.2756, 714.0566, 723.8325, 733.6031, 743.3685,
                753.1287, 762.8837, 772.6335, 782.3781, 792.1174, 801.8516, 811.5805,
                821.3043, 831.0228, 840.7361, 850.4442, 860.1471, 869.8448, 879.5372,
                889.2245, 898.9066, 908.5834, 918.2551, 927.9214, 937.5827, 947.2387,
                956.8895, 966.5351, 976.1755, 985.8106, 995.4406, 1005.065, 1014.685,
                1024.299, 1033.908, 1043.512, 1053.111, 1062.704, 1072.293, 1081.876,
                1091.454, 1101.026, 1110.594, 1120.156, 1129.713, 1139.265, 1148.811,
                1158.353, 1167.889, 1177.420, 1186.946, 1196.466, 1205.982, 1215.492,
                1224.997, 1234.496, 1243.991, 1253.480, 1262.964, 1253.373, 1263.346,
                1273.318, 1283.291, 1293.262, 1303.234, 1313.206, 1323.177, 1333.148,
                1343.119, 1353.089, 1363.060, 1373.030, 1383.000, 1392.969, 1402.939,
                1412.908, 1422.877, 1432.845, 1442.814, 1452.782, 1462.750, 1472.718,
                1482.685, 1492.652, 1502.619, 1512.586, 1522.552, 1532.518, 1542.484,
                1552.450, 1562.416, 1572.381, 1582.346, 1592.311, 1602.275, 1612.240,
                1622.204, 1632.167, 1642.131, 1652.094, 1662.057, 1672.020, 1681.983,
                1691.945, 1701.907, 1711.869, 1721.831, 1731.792, 1741.753, 1751.714,
                1761.675, 1771.635, 1781.596, 1791.556, 1801.515, 1811.475, 1821.434,
                1831.393, 1841.352, 1851.310, 1861.269, 1871.227, 1880.184, 1874.164,
                1884.225, 1894.285, 1904.342, 1914.396, 1924.448, 1934.499, 1944.546,
                1954.592, 1964.635, 1974.675, 1984.714, 1994.750, 2004.784, 2014.815,
                2024.845, 2034.872, 2044.896, 2054.919, 2064.939, 2074.956, 2084.972,
                2094.985, 2104.996, 2115.004, 2125.010, 2135.014, 2145.016, 2155.015,
                2165.012, 2175.007, 2184.999, 2194.989, 2204.977, 2214.962, 2224.945,
                2234.926, 2244.905, 2254.881, 2264.854, 2274.826, 2284.795, 2294.762,
                2304.727, 2314.689, 2324.649, 2334.607, 2344.562, 2354.516, 2364.467,
                2374.415, 2384.361, 2394.305, 2404.247, 2414.186, 2424.123, 2434.058,
                2443.990, 2453.920, 2463.848, 2473.773, 2483.696, 2493.617, 2503.536
            ])/1000.0  # from .hdr file
            
        self.absolute_scale = absolute_scale

    def load_data(self, datafile_dir):
        # load 'dc.tif'
        data = tiff.imread(datafile_dir)  # np.array[H, W, C] or [C, H, W], int16
        # convert to torch.tensor with dtype float32
        data = torch.tensor((data.astype('float32')))
        if data.shape[-1] == 224:  # if the last dimension is 224, it means the data is in [H, W, C] format
            data = data.permute(2, 0, 1)
        elif data.shape[0] == 224:  # if the first dimension is 224, it means the data is in [C, H, W] format
            pass
        else:
            raise ValueError(f'Invalid data shape: {data.shape}. Expected [H, W, 224] or [224, H, W].')

        if self.absolute_scale:
            data = data/32767.0
        else:
            data = self.minmax_scale(data)  # min-max scale the hyperspectral data to [0, 1]
        
        return data, self.coord

    
class AVIRIS_scenes_Interpolation(AVIRIS_scenes_SSR):
    def __init__(self, data_dir, srf_dir, crop_size, wvlgth_dir=None, absolute_scale=True,
                 scale=4):
        super().__init__(data_dir, srf_dir, crop_size, wvlgth_dir, absolute_scale)

        self.scale = scale
    
    def __getitem__(self, index):
        
        if os.path.isdir(self.data_dir):
            data_path = os.path.join(self.data_dir, self.data_dir_list[index])
            data, coord = self.load_data(data_path)
        elif os.path.isfile(self.data_dir):
            data, coord = self.data, self.coord
        else:
            raise ValueError(f'Invalid data_dir: {self.data_dir}. It should be a file or a folder.')
        

        coord = coord[::self.scale]  # downsample at self.scale

        # randomly crop the image
        hs, ws, h, w = self.randomcrop(data, self.crop_size)
        hsi = data[:, hs:hs+h, ws:ws+w]  # tensor[C, H, W] ([224, crop_size, crop_size])

        hsi = hsi[::self.scale, :, :]  # downsample at self.scale
        C = hsi.shape[-3]  # C: number of bands

        # randomly select a srf
        srf = random.choice(self.srf_list)
        # get the srf
        srf_b_at_coord = torch.where(
            (coord >= self.srf_coord.min()) & (coord <= self.srf_coord.max()),
            srf['b'](coord),
            torch.tensor(0.0, dtype=torch.float32)
        )
        srf_g_at_coord = torch.where(
            (coord >= self.srf_coord.min()) & (coord <= self.srf_coord.max()),
            srf['g'](coord),
            torch.tensor(0.0, dtype=torch.float32)
        )
        srf_r_at_coord = torch.where(
            (coord >= self.srf_coord.min()) & (coord <= self.srf_coord.max()),
            srf['r'](coord),
            torch.tensor(0.0, dtype=torch.float32)
        )
        srf = torch.stack((srf_r_at_coord, srf_g_at_coord, srf_b_at_coord), dim=0)  # [3, 224]
        srf = srf / torch.sum(srf, dim=1, keepdim=True)  # normalize srf to make it sum to 1 for each row, [3, 224]

        rgb = torch.matmul(srf, hsi.contiguous().view(C, -1)).view(-1, h, w)  # [3, crop_size, crop_size]
        
        return (rgb, coord, srf), (hsi, srf)
    