# coding: utf-8

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import numpy as np
import pandas as pd
from PIL import Image
from skimage import color as ColorTran

import os
import sys
import time
import pickle
import threading
sys.path.append("/usr/local/lib/python2.7/dist-packages")
import cv2 as cv

import Filters as F



class MyThread(threading.Thread):
    def __init__(self,func,args=()):
        super(MyThread,self).__init__()
        self.func = func
        self.args = args
    
    def run(self):
        self.result = self.func(*self.args)
    
    def get_result(self):
        try:
            return self.result
        except Exception:
            print("Error in Thread")
            return None



class DataLoaderCASIA(object):
    """data loader for CASIA and other datasets
    """
    def __init__(self, root, txt_path) :
        """
            args:
                root: the root of Dataset
                txt_path: the absolute path of the txt caontaing all thg paths and labels of dataset,
                        
        """
        self.root = root
        self.txt_path = txt_path
        self._df = self._read_dataset_paths()
    
    def _read_dataset_paths(self) :
        """function: get the path from txt and distinguish the label of each path, shuffule the paths and labels together if needed 
                
            args:
                
            return:
                df:   dataframe, column_1: path of imgae, column_2: label of image
        """
        # read all the lines
        with open(self.txt_path, "r") as fp:
            lines = fp.readlines()
            print("number of lines: ", len(lines))
        
        # pick out the lines required
        print("the first line: ", lines[0])
        lines = [os.path.join(self.root, line).strip("\r\n").replace("\\", "/") for line in lines]
        print("the first line: ", lines[0])
        
        # split the paths and labels from each lines
        total_path = []
        total_label = []
        for line in lines:
            if line.find(" ") == -1:
                raise Exception("txt name: ", txt_name, "\r\n", line)
            path, label = line.split(" ")
            total_path.append( path )
            total_label.append(int(label))
        
        df = pd.DataFrame( {"path": total_path, "label": total_label} )
        
        return df
    
    def _get_cost_time(self, diff) :
        second = diff % 60
        minute = diff % 3600
        minute = minute // 60
        hour = diff % 86400
        hour = hour // 3600
        day = diff // 86400
        format_string="{}d {}h:{}m:{}s"
        return format_string.format(day, hour, minute, second)
    
    def _transform_color(self, img, imgType):
        """function: transform the color space

            args:
                img: PIL.Image
                imgType: string, "gray", "rgb", "hsv", "luv", "lab", "ycrcb", "hls", "private".

            return:
                image in target color space
        
        """
        if imgType == "gray" or imgType == "GRAY" :
            img = img.convert("L")
            img = np.array(img)
            img = img[..., np.newaxis]
            
        elif imgType == "rgb" or imgType == "RGB" :
            if img.mode != "RGB":
                img = np.array(img)
                img = np.dstack((img, img, img))
            else :
                img = np.array(img)
            
        elif imgType == "cmyk" or imgType == "CMYK" :
            img = img.convert("CMYK")
            img = np.array(img)
            
        elif imgType == "hsv" or imgType == "HSV" :
            img = np.array(img)
            img = cv.cvtColor(img, cv.COLOR_RGB2HSV)
            
        elif imgType == "luv" or imgType == "LUV" :
            img = np.array(img)
            img = cv.cvtColor(img, cv.COLOR_RGB2LUV)
            
        elif imgType == "lab" or imgType == "LAB" :
            img = np.array(img)
            img = cv.cvtColor(img, cv.COLOR_RGB2LAB)
            
        elif imgType == "ycrcb" or imgType == "YCrCb" :
            img = np.array(img)
            img = cv.cvtColor(img, cv.COLOR_RGB2YCrCb)
            
        elif imgType == "hls" or imgType == "HLS" :
            img = np.array(img)
            img = cv.cvtColor(img, cv.COLOR_RGB2HLS)
            
        elif imgType == "private" :
            img = np.array(img)
            temp_LAB = cv.cvtColor(img, cv.COLOR_RGB2LAB)
            temp_HSV = cv.cvtColor(img, cv.COLOR_RGB2HSV)
            img = np.concatenate((temp_HSV,temp_LAB), axis=2)
            
        else:
            raise Exception("wrong image type!")
            
        return img
    
    def _filter(self, img, filter_type, filter_paras):
        imgs = np.transpose(img, (2,0,1))
        imgs = F.Filter_list(imgs, Ftype=filter_type, N=filter_paras["N"], K=filter_paras["K"],
                             diffuse_function=filter_paras["diffuse_function"], gamma=filter_paras["gamma"])
        imgs = np.array(imgs)
        imgs = np.transpose(imgs, (1,2,0))
        return imgs
        
    
    def _process_pipeLine(self, path, imgSize, imgType, filter_type=None, filter_paras=None) :
        """function: process the subset of paths, mainly for the multi-thread function
            args:
                path: path of image
                imgType: target image color space, "gray", "rgb", "hsv", "luv", "lab", "ycrcb", "hls", "private"
                imgSize: target image size, (0,0) means no change
                filter_type: the type of filter, eg, "ND"
                filter_paras: dic, {"num":1, "tau":30, "h":1, "sigma":7, "lamda":17}
            return:
                img: the processed image
        """
        # read image
        img = Image.open(path)
        
        # resize image
        if img.size != imgSize and imgSize[0] != 0 :
            img = img.resize(imgSize)
        
        # color space transformation
        img = self._transform_color(img, imgType)
        
        # filter the image
        if filter_type is not None :
            img = self._filter(img, filter_type, filter_paras)
        
        return img
    
    
    def _multi_func(self, paths, imgSize, imgType, filter_type=None, filter_paras=None) :
        """function: process the subset of paths, mainly for the multi-thread function
            args:
                paths: subset of dataset's paths
                imgType: target image color space, "gray", "rgb", "hsv", "luv", "lab", "ycrcb", "hls", "private"
                imgSize: target image size, (0,0) means no change
                filter_type: the type of filter, eg, "ND"
                filter_paras: dic, {"num":1, "tau":30, "h":1, "sigma":7, "lamda":17}
            return:
                video_set: the processed image set
        """
        video_set = []
        path_set = []
        for i, path in enumerate(paths) :
            img = self._process_pipeLine(path, imgSize, imgType, filter_type=filter_type, filter_paras=filter_paras)
            if img is None :
                continue
            
            # append image
            if img.min() != img.max():
                video_set.append(img)
                path_set.append(path)
        
        return video_set, path_set
    
    def get_data(self, imgType="gray", imgSize=(64,64), if_alignment=False, num_threads=10, filter_type=None, filter_paras=None, separator="/"):
        """function: group the iamges and labels, each label is crossponding to a group containing a set of images
            
            args:
                imgType: target image color space, "gray", "rgb", "hsv", "luv", "lab", "ycrcb", "hls", "private"
                imgSize: target image size, (0,0) means no change
                if_alignment: whether to make the face vertical
                num_threads: number of threads to read the data
                filter_type: the type of filter, eg, "ND"
                filter_paras: dic, {"num":1, "tau":30, "h":1, "sigma":7, "lamda":17}
            
            return:
                video_set: frames of all video, (n_frames, n_row, n_column, n_channel)
                group_index_label: all index of each video's frames in the whole video_set and its label which is a dictionary
                        like, "real1": ([0,1,2,...,8], label), "fake1":([9,10,..,32],label), ...
        """
        start_time = time.time()
        self._df["prefix"] = self._df[["path"]].applymap(lambda ele: ele[:ele.rfind(separator)] )
        groups = self._df.groupby("prefix")
        
        count = 0
        video_set = []
        start = 0
        group_index_label = dict() # key: (index_array, label)
        temp_start_time = time.time()
        for key, group in groups:
            paths = group["path"]
            
            path_set = []
            threads = []
            step = int( np.floor( len(paths)/num_threads ) )
            for i in np.arange(num_threads) :
                start_pos = i*step
                end_pos = (i+1)*step
                if i == num_threads-1 :
                    end_pos = len(paths)
                
                sub_paths = paths.iloc[start_pos:end_pos]
                
                t = MyThread(self._multi_func, (sub_paths, imgSize, imgType, filter_type, filter_paras))
                threads.append(t)
            
            for t in threads :
                t.start()
            for t in threads :
                t.join()
            for t in threads :
                tmp_video_set, tmp_path_set = t.get_result()
                video_set = video_set + tmp_video_set
                path_set = path_set + tmp_path_set
            
            
            # append index and label
            group_index_label[key] = (range(start, len(video_set)), group["label"].iloc[0], path_set)
            start = len(video_set)
            
            # print the current information
            count += 1
            if count%50 == 0 :
                temp_end_time = time.time()
                cost_time = self._get_cost_time(temp_end_time-temp_start_time)
                print(count, " - cost time: ", cost_time)
                temp_start_time = time.time()
        print()
        
        video_set = np.array(video_set)
        print("There are {} groups.".format( len(group_index_label.keys()) ))
        end_time = time.time()
        cost_time = self._get_cost_time(end_time-start_time)
        print("{} - Complete the data loading, cost time: {}".format(time.asctime( time.localtime(time.time()) ), cost_time))
        
        return video_set, group_index_label
    


class Saver(object) :
    """function: the format of saved data:
            1: [images_set.shape, num_step]
            2: 1st part of images_set
            3: 2nd part of images_set
            ...
            num_step+1: num_step-th part of images_set
            num_step+2: group_index
    """
    def __init__(self, path, seed=17) :
        self.path = path
        np.random.seed(seed)
    
    def _get_cost_time(self, diff) :
        second = diff % 60
        minute = diff % 3600
        minute = minute // 60
        hour = diff % 86400
        hour = hour // 3600
        day = diff // 86400
        format_string="{}d {}h:{}m:{}s"
        return format_string.format(day, hour, minute, second)
    
    def save(self, images_set, group_index, num_step) :
        """function: sava the images_set and group_index
            args:
                images_set: (n_images, width, height)
                group_index: ()
                num_step: seperate the images_set as int(np.ceil(images_set.shape[0]/num_step))
        """
        start_time = time.time()
        with open(self.path, "wb") as fp_write:
            pickle.dump([images_set.shape, num_step], fp_write, protocol=2)
            print("shape of loaded data: {},   num_step:{}".format(images_set.shape, num_step))
#             print("shape of saved data: ", images_set.shape)
            
            step = int(np.ceil(images_set.shape[0]/num_step))
            for start in np.arange( 0, images_set.shape[0], step ) :
                pickle.dump( images_set[start:start+step, ...], fp_write, protocol=2 )
            
            pickle.dump(group_index, fp_write, protocol=2)
        end_time = time.time()
        cost_time = self._get_cost_time(end_time-start_time)
        print("save all the data and group, cost time: {}".format(cost_time))
    
    def load(self, leaveOut=None, pickOut=None, pro=None, subset_mode=None, mode_value=None) :
        """function: load the iamges_set and group_index
            args:
                leaveOut: split out the specific list, and only test the rest set, like [4,5]
                pickOut: pick out the specific data for ACER testing
                pro: the protocol now used
                subset_mode: if is not None, the we will get the subset under specific mode, "step","ratio","amount"
                mode_value: the value of each mode
        """
        start_time = time.time()
        with open(self.path, "rb") as fp_read:
            shape, num_step = pickle.load(fp_read, encoding="latin1")
            print("shape of loaded data: {},   num_step:{}".format(shape, num_step))
            
            # num_step = 10
            step = int(np.ceil(shape[0]/num_step))
            for start in np.arange( 0, shape[0], step ) :
                if start == 0 :
                    images_set = pickle.load(fp_read, encoding="latin1")
                else :
                    images_set = np.concatenate((images_set, pickle.load(fp_read, encoding="latin1")), axis=0)
            
            group_index = pickle.load(fp_read, encoding="latin1")
        
        # select some images if necessary
        if subset_mode is not None :
            if subset_mode == "step" :
                image_set, group_index = self._get_subset(images_set, group_index, step=mode_value)
            elif subset_mode == "ratio" :
                image_set, group_index = self._get_subset(images_set, group_index, ratio=mode_value)
            elif subset_mode == "amount" :
                image_set, group_index = self._get_subset(image_set, group_index, amount=mode_value)
            elif subset_mode == "one" :
                image_set, group_index = self._get_subset(image_set, group_index, one=subset_mode)
            else :
                raise Exception("Expect the spesific mode in ['step', 'ratio', 'amount'] but get ", subset_mode)
            
        end_time = time.time()
        cost_time = self._get_cost_time(end_time-start_time)
        print("Loading is complete: images_set.shape:{}, length of group_index:{}, cost time:{}".format(np.array(images_set).shape, len(group_index.keys()), cost_time))
        print("dtype of data: {},   size of data: {}MB".format(images_set.dtype, sys.getsizeof(images_set)/1024/1024))
        
        return images_set, group_index
    
    def _get_subset(self, image_set, group_index, step=None, ratio=None, amount=None, one=None) :
        """function: get the subset from the original image set with step 'step'
            args:
                step: if step is not None, then choose images every 'step' frames
                ratio: if ratio is not None, then choose a certain percentage of the collection from the original collection
                amount: if amount is not None, then choss subset images with a certain amount
                one: randomly pick out a image
            return:
                image_set:
                group_index:
        """
        def model(step, ratio, amount) :
            if step is not None :
                return "step"
            elif ratio is not None :
                return "ratio"
            elif amount is not None :
                return "amount"
            else :
                raise Exception("Please specific the mode")
        
        print("Note!!! we will get the image subset from original data under '{}' model".format( model(step, ratio, amount) ))
        group_index_new = dict()
        image_set_new = []
        start = 0
        
        keys = group_index.keys()
        for key in keys :
            if step is not None :
                index = group_index[key][0]
                index = index[::step]
                
            elif ratio is not None :
                index = group_index[key][0]
                size = int(np.floor( ratio*len(index) ))
                temp = np.random.randint(0, high=len(index), size=size)
                temp = np.sort(temp)
                index = index[temp]
                
            elif amount is not None :
                index = group_index[key][0]
                temp = np.random.randint(0, high=len(index), size=amount)
                temp = np.sort(temp)
                index = index[temp]
            
            elif one is not None:
                index = group_index[key][0]
                temp = np.random.randint(0, high=len(index), size=1)
                index = index[temp]
            
            # get the new image_set and group_index
            image_set_new += list( image_set[index] )
            group_index_new[key] = (range(start, len(image_set_new)), group_index[key][1], group_index[key][2])
            start = len(image_set_new)
            
        image_set_new = np.array(image_set_new)
            
        return image_set_new, group_index_new
            