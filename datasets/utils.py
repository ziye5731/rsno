import torch
import torch.nn as nn
import os
import scipy.io as sio
import random
import torch.utils.data as tud

class Interpolator1D(nn.Module):
    def __init__(self, x_data, y_data):
        """
        1D linear interpolator.
        Args:
            x_data: Independent variable data, can be a list, numpy array, or tensor.
            y_data: Dependent variable data, corresponding to x_data, can be a list, numpy array, or tensor.
        """
        super(Interpolator1D, self).__init__()
        
        # Convert input to tensor
        x_data = torch.as_tensor(x_data, dtype=torch.float32)
        y_data = torch.as_tensor(y_data, dtype=torch.float32)
        
        # Ensure x_data is monotonically increasing
        sorted_indices = torch.argsort(x_data)
        x_data = x_data[sorted_indices]
        y_data = y_data[sorted_indices]
        
        # Save data
        self.register_buffer('x_data', x_data)
        self.register_buffer('y_data', y_data)
    
    def forward(self, x):
        """
        Perform linear interpolation for the input x.
        Args:
            x (torch.Tensor): Independent variable values to interpolate.
        Returns:
            torch.Tensor: Interpolated dependent variable values.
        """
        # Find the insertion positions of x in x_data
        indices = torch.searchsorted(self.x_data, x)
        
        # Handle boundary cases
        indices = torch.where(indices < 1, torch.tensor(1, device=indices.device), indices)
        indices = torch.where(indices > len(self.x_data) - 1, 
                             torch.tensor(len(self.x_data) - 1, device=indices.device), 
                             indices)
        
        # Get adjacent x and y values
        x_left = self.x_data[indices - 1]
        x_right = self.x_data[indices]
        y_left = self.y_data[indices - 1]
        y_right = self.y_data[indices]
        
        # Calculate interpolation
        t = (x - x_left) / (x_right - x_left)
        y = y_left + t * (y_right - y_left)
        
        return y



class SSR_dataset(tud.Dataset):
    '''
    Base class of spectral super resolution dataset, which uses SRF to obtain RGB images.
    Args:
        data_dir: str, path to the dataset folder/file.
        srf_dir: str, path to the camSpecSensitivity database folder, which is obtained from https://www.gujinwei.org/research/camspec/.
        crop_size: int, size of the crop. If false or None, the whole image will be used.
    '''
    def __init__(self, data_dir, srf_dir, crop_size):
        super().__init__()
        self.data_dir = data_dir
        self.srf_dir = srf_dir
        self.crop_size = crop_size

        # if data_dir is a file, treat it as a single hyperspectral image
        if os.path.isfile(self.data_dir):
            self.data, self.coord = self.load_data()  # load hyperspectral data
        # if data_dir is a folder, treat it as a folder containing multiple hyperspectral images
        elif os.path.isdir(self.data_dir):
            self.data_dir_list = os.listdir(self.data_dir)
        else:
            raise ValueError(f'Invalid data_dir: {self.data_dir}. It should be a file or a folder.')

        self.load_srf()

        #self.minmax_scale()  # min-max scale the hyperspectral data to [0, 1]

    def __len__(self):
        if os.path.isfile(self.data_dir):
            return 1
        elif os.path.isdir(self.data_dir):
            return len(self.data_dir_list)
        else:
            raise ValueError(f'Invalid data_dir: {self.data_dir}. It should be a file or a folder.')

    def load_data(self):
        '''
        Load hyperspectral data, 
        which is a 3D tensor with shape [C, H, W], 
        where C is the number of bands, H is the height, and W is the width.
        Should be implemented in the derived class.

        Args:
            None
        Returns:
            data: tensor[C, H, W], hyperspectral data.
            coord: tensor[C], wavelength of the hyperspectral data.
        '''
        pass

    def load_srf(self):
        '''
        Load spectral response function (SRF) data, 
        which is a 2D tensor with shape [M, C], 
        where M is the number of bands in the compressed image and C is the number of bands in the hyperspectral image.
        Should be implemented in the derived class.
        Args:
            None
        Returns:
            srf: tensor[M, C], SRF data.
        '''
        srf_dir_list = os.listdir(self.srf_dir)
        srf_dir_list = [f for f in srf_dir_list if f.endswith('.mat')]  # filter out files which are not .mat files
        self.srf_dir_list = srf_dir_list
        self.srf_coord = torch.linspace(0.4, 0.72, 33)

        self.srf_list = []
        for cam in self.srf_dir_list:
            cam_srf = sio.loadmat(os.path.join(self.srf_dir, cam))
            b = cam_srf['b']
            g = cam_srf['g']
            r = cam_srf['r']

            # b, g, r shape can be either [1,33] or [33,1], squeeze it to [33]
            b = b.squeeze()
            g = g.squeeze()
            r = r.squeeze()

            srf_b = Interpolator1D(self.srf_coord, b)
            srf_g = Interpolator1D(self.srf_coord, g)
            srf_r = Interpolator1D(self.srf_coord, r)

            cam_dict = {
                'name': os.path.splitext(cam)[0],
                'b': srf_b,
                'g': srf_g,
                'r': srf_r
            }
            self.srf_list.append(cam_dict)

    def randomcrop(self,img,crop_size):
        c,h,w= img.shape
        if type(crop_size) is int:
            th = tw = crop_size
        elif type(crop_size) is tuple:
            th, tw = crop_size
        elif crop_size is None or crop_size is False:
            th, tw = h, w
        if w == tw and h == th:
            return 0,0,h,w
        if w < tw or h < th:
            print('the size of the image is not enough for crop!')
            exit(-1)
        i = random.randint(0, h - th-10)
        j = random.randint(0, w - tw-10)
        return i,j,th,tw

    def __getitem__(self, index):
        
        if os.path.isdir(self.data_dir):
            data_path = os.path.join(self.data_dir, self.data_dir_list[index])
            data, coord = self.load_data(data_path)
        elif os.path.isfile(self.data_dir):
            data, coord = self.data, self.coord
        else:
            raise ValueError(f'Invalid data_dir: {self.data_dir}. It should be a file or a folder.')
        C = data.shape[-3]  # C: number of bands

        # randomly crop the image
        hs, ws, h, w = self.randomcrop(data, self.crop_size)
        hsi = data[:, hs:hs+h, ws:ws+w]  # tensor[C, H, W] ([224, crop_size, crop_size])
        

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
    
    def minmax_scale(self, data):
        '''
        Min-max scale the hyperspectral data to [0, 1].
        Args:
            None
        Returns:
            data: tensor[C, H, W], min-max scaled hyperspectral data.
        '''
        min_val = torch.min(data)
        max_val = torch.max(data)
        data = (data - min_val) / (max_val - min_val)
        return data
