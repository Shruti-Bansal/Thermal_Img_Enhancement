# Written by Ukcheol Shin (shinwc159[at]gmail.com)
import torch
import torch.utils.data as data
import numpy as np
import os.path as osp
import cv2
import random
from path import Path
from Datasets.MS2_utils import load_as_float_img, load_as_float_depth, get_transform, process_image


class DataLoader_MS2(data.Dataset):
    """A data loader where the files are arranged in this way:
    * Structure of "KAIST MS2 dataset"
    |--sync_data
        |-- <Seq name>
            |-- rgb, nir, thr
                |-- img_left
                |-- img_right
            |-- lidar
                |-- left
                |-- right
            |-- gps_imu
                |-- data
            |-- calib.npy
    |--proj_depth
        |-- <Seq name>
            |-- rgb, nir, thr
                |-- depth
                |-- intensity
                |-- depth_multi
                |-- intensity_multi
                |-- depth_filtered
    |--odom
        |-- <Seq name>
            |-- rgb, nir, thr, odom
    |-- train_list.txt
    |-- val_list.txt
    |-- test_list.txt
    """

    def __init__(
        self,
        root,
        datalist=None,
        seed=None,
        data_split="train",
        process="raw",
        resolution="640x256",
        sampling_step=3,
        set_length=3,
        set_interval=1,
        blur=None,  # blur = ksize (odd) : rate (0-1)
        opt=None,
    ):
        super(DataLoader_MS2, self).__init__()

        np.random.seed(seed)
        random.seed(seed)

        self.root = Path(root)
        # read (train/val/test) data list
        if datalist is not None:
            data_list_file = datalist
            #print("Data:",data_list_file)
        else:
            if data_split == "train":
                data_list_file = self.root / "train_list.txt"
            elif data_split == "val":
                data_list_file = self.root / "val_list.txt"
            elif data_split == "test":
                data_list_file = self.root / "test_list.txt"
            elif data_split == "test_day":
                data_list_file = self.root / "test_day_list.txt"
            elif data_split == "test_night":
                data_list_file = self.root / "test_night_list.txt"
            elif data_split == "test_rain":
                data_list_file = self.root / "test_rainy_list.txt"
            else:  # when data_split is a specific sequence name
                data_list_file = data_split

        # check if data_list_file has the .txt extension and create a list of folders
        if "txt" in data_list_file:
            self.seq_list = [seq_name[:-1] for seq_name in open(data_list_file)]
        else:
            self.seq_list = [data_list_file]

        self.modality = ["rgb", "thr"]
        #self.extrinsics = self.set_extrinsics()
            
        self.data_getter = self.get_RGB_thermal
        self.crawl_folders_img_enhnc(sampling_step, set_length, set_interval)

        # determine which data getter function to use
        # if data_format == "MonoDepth":  # Monocular depth estimation, dict: {'img', 'depth'}
        #     self.data_getter = self.get_data_MonoDepth
        #     self.crawl_folders_depth(sampling_step, set_length, set_interval)
        # elif data_format == "StereoMatch":
        #     self.data_getter = self.get_data_StereoMatching
        #     self.crawl_folders_depth(sampling_step, set_length, set_interval)
        # elif data_format == "MultiViewImg":
        #     self.data_getter = self.get_data_MultiViewImg
        #     self.crawl_folders_depth(sampling_step, set_length, set_interval)
        # elif data_format == "Odometry":
        #     self.data_getter = self.get_data_Odometry
        #     self.crawl_folders_pose(sampling_step, set_length, set_interval)
        # else:
        #     raise NotImplementedError(f"not supported type {data_format} in KAIST MS2 dataset.")

        self.process = process
        self.image_shape = tuple(map(int, resolution.split("x")))  # (width, height)

        

    def __getitem__(self, index):
        
        return self.data_getter(index)

    def __len__(self):
        #print("Len:", len(self.samples["rgb"]))
        return len(self.samples["rgb"])


    def crawl_folders_img_enhnc(self, sampling_step, set_length, set_interval):
        # define shifts to select reference frames
        demi_length = (set_length - 1) // 2
        shift_range = np.array([set_interval * i for i in range(-demi_length, demi_length + 1)]).reshape(1, -1) # check this reshape

        # crawl the request modality list
        sensor_list = []
        if isinstance(self.modality, list):
            for modal in self.modality:
                sensor_list.append(modal)
        else:
            sensor_list.append(self.modality)

        # iterate over each sensor modalitiy
        sample_sets = {}
        for sensor in sensor_list:
            # create an empty list to store samples for this sensor modality
            sample_set = []
            for seq in self.seq_list:  # iterate over each folder
                #print("Seq:",seq)
                img_list_left = sorted((self.root / "MS2" / "sync_data" / seq / sensor / "img_left").files("*.png"))
                #img_list_right = sorted((self.root / "sync_data" / seq / sensor / "img_right").files("*.png"))

                # construct N-snippet sequences (note: N=set_length)
                init_offset = demi_length * set_interval
                tgt_indices = np.arange(init_offset, len(img_list_left) - init_offset).reshape(-1, 1)
                snippet_indices = shift_range + tgt_indices

                for indices in snippet_indices:
                    sample = {"imgs": []}
                    for i in indices:
                        tgt_name = img_list_left[i].name[:-4]
                        #print(img_list_left[i])
                        sample["imgs"].append(img_list_left[i])
                        
                    sample_set.append(sample)

            # Subsampling the list of images according to the sampling step
            sample_set = sample_set[0:-1:sampling_step]
            sample_sets[sensor] = sample_set

        self.samples = sample_sets

    
    def get_RGB_thermal(self, index):
        sample_rgb = self.samples["rgb"][index]
        sample_thr = self.samples["thr"][index]

        #print("sample:", sample_rgb["imgs"][0])

        tgt_img_rgb = load_as_float_img(sample_rgb["imgs"][0])
        tgt_img_thr = load_as_float_img(sample_thr["imgs"][0])

        tgt_img_gray = cv2.cvtColor(tgt_img_rgb, cv2.COLOR_RGB2GRAY)

        w, h = self.image_shape
        tgt_img_gray = cv2.resize(tgt_img_gray, (w, h))
        tgt_img_gray = np.expand_dims(tgt_img_gray, axis=2)
        #print("Gray:", tgt_img_gray.shape)
        #print("Thr:", tgt_img_thr.shape)

        result = {}
        result["gray"] = tgt_img_gray
        result["thr"] = tgt_img_thr
        return result

    

   