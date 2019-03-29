# coding: utf-8

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import os
import sys
import time
import numpy as np
from scipy import ndimage as sciImg
from skimage import filters as skiFil

import multiprocessing
sys.path.append("/usr/local/lib/python2.7/dist-packages")
import cv2


def anisotropic_diffusion(img, N=10, K=-1.0, diffuse_function='exp', gamma=0.15):
    '''function: calc anisotropic diffusion of image <img> (4 neighbor)
        
        parameters:
            img：image
            N：number of iteration
            K：paramter of diffuse_function e.g (v/K)**2.0,
                if K<0, then will be calculated automatically
            diffuse_function：'exp' or 'rcp'
            mu：lamda e.g I = I + mu*delta
        
        return: image after anisotropic diffusion'''
    
    auto_K = K < 0
    
    # determine g(||dI(t)||)
    def g_exp(v): return 0 if K==0 else np.exp(-(v/K)**2.0)
    def g_rcp(v): return 0 if K==0 else 1.0/(1.0 + (v/K)**2.0)
    g_func = {'exp': g_exp, 'rcp': g_rcp}[diffuse_function]
    
    # calculate the gradience
    def gradient(m):
        gradx = np.hstack([m[:, 1:], m[:, -2:-1]]) - np.hstack([m[:, 0:1], m[:, :-1]])
        grady = np.vstack([m[1:, :], m[-2:-1, :]]) - np.vstack([m[0:1, :], m[:-1, :]])
        return gradx, grady
    
    def W_diff(m): return np.hstack([m[:, 0:1], m[:, :-1]])   - m
    def E_diff(m): return np.hstack([m[:, 1:],  m[:, -2:-1]]) - m
    def N_diff(m): return np.vstack([m[0:1, :], m[:-1, :]])   - m
    def S_diff(m): return np.vstack([m[1:, :],  m[-2:-1, :]]) - m
    
    # iteration
    I = np.array(img)
    for i in range(N):
        # calculate the K automatically
        if auto_K:
            gx, gy = gradient(I)
            K = np.sqrt(gx**2.0 + gy**2.0).mean()*0.9 # constatn 0.9 from the paper.
        
        # proceddure of anisotropic diffusion
        W, E, N, S = W_diff(I), E_diff(I), N_diff(I), S_diff(I)
        I_delta = W*g_func(np.abs(W)) + E*g_func(np.abs(E)) + N*g_func(np.abs(N)) + S*g_func(np.abs(S))
        I = I + gamma*I_delta
    
    return I
    


def Filter_list(images, Ftype="AD", N=5, K=-1, diffuse_function='exp', gamma=0.15) :
    start_time = time.time()
    diffed_images = []
    count = 0
    for img in images:
    #     temp = anisotropic_diffusion(img, N=ani_iter, K=ani_K, diffuse_function=ani_diffunc, gamma=ani_mu)
        if Ftype=="AD" :
            temp = anisotropic_diffusion(img, N=5, K=-1, diffuse_function='exp', gamma=0.15)
        elif Ftype=="Sobel" :
            temp = skiFil.sobel(img)
        elif Ftype=="Scharr" :
            temp = skiFil.scharr(img)
        elif Ftype=="Laplace" :
            temp = skiFil.laplace(img)
        elif Ftype=="Prewitt" :
            temp = skiFil.prewitt(img)
        elif Ftype=="Roberts" :
            temp = skiFil.roberts(img)
            
        elif Ftype=="Median" :
            temp = sciImg.median_filter(img, size=(3,3))
        elif Ftype=="Maximum" :
            temp = sciImg.maximum_filter(img, size=(3,3))
        elif Ftype=="Minimum" :
            temp = sciImg.minimum_filter(img, size=(3,3))
        elif Ftype=="Mean" :
            temp = cv2.blur(img, (3,3))
        elif Ftype=="Gaussian" :
            temp = skiFil.gaussian(img, sigma=sigma)
            
        elif Ftype=="Sobel-Max" :
            temp = skiFil.sobel(img) + sciImg.maximum_filter(img, size=(3,3))
        elif Ftype=="Sobel-Min" :
            temp = skiFil.sobel(img) + sciImg.minimum_filter(img, size=(3,3))
        elif Ftype=="Sobel-Mean" :
            temp = skiFil.sobel(img) + cv2.blur(img, (3,3))
        elif Ftype=="Sobel-Gau" :
            temp = skiFil.sobel(img) + skiFil.gaussian(img, sigma=sigma)
        elif Ftype=="Prewitt-Max" :
            temp = skiFil.prewitt(img) + sciImg.maximum_filter(img, size=(3,3))
        elif Ftype=="Prewitt-Min" :
            temp = skiFil.prewitt(img) + sciImg.minimum_filter(img, size=(3,3))
        elif Ftype=="Prewitt-Mean" :
            temp = skiFil.prewitt(img) + cv2.blur(img, (3,3))
        elif Ftype=="Prewitt-Gau" :
            temp = skiFil.prewitt(img) + skiFil.gaussian(img, sigma=sigma)
        elif Ftype=="Roberts-Max" :
            temp = skiFil.roberts(img) + sciImg.maximum_filter(img, size=(3,3))
        elif Ftype=="Roberts-Min" :
            temp = skiFil.roberts(img) + sciImg.minimum_filter(img, size=(3,3))
        elif Ftype=="Roberts-Mean" :
            temp = skiFil.roberts(img) + cv2.blur(img, (3,3))
        elif Ftype=="Roberts-Gau" :
            temp = skiFil.roberts(img) + skiFil.gaussian(img, sigma=sigma)
        else :
            raise Exception("No such filter type")
        
        diffed_images.append(temp)
        
        count += 1
        if count == 1000 :
#             print("pid: ", os.getpid(), "   size of data: ", sys.getsizeof(diffed_images)/1024/1024,"MB")
            count = 0
    # print("################ pid: ", os.getpid(), "   size of data: ", sys.getsizeof(diffed_images)/1024/1024,"MB")
    #print("################ pid: ", os.getpid(), "   cost time: ", get_cost_time( time.time()-start_time ))
    return diffed_images



def Filter_MulProcess(image_set, Ftype="AD", N=5, K=-1, diffuse_function='exp', gamma=0.15):
    """function: perform the nonlinear diffusion while in multiple process mode
        
        args:
            image_set: a set containing images that need to be diffused
            Ftype: the type of filter, 
                    "AD": anistropic diffusion
                    "Sobel": sobel transform
                    "Scharr": scharr tansform
                    "Laplace": laplace transform
                    "Prewitt": prewitt transform
                    "Roberts": roberts cross transform
                    "Median": median filter
                    "Maximum": maximum filter
                    "Minimum": minimum filter
                    "Mean": mean filter
                    "Gaussian": gaussian filter
            num: number of iteration
            tau: step size of each iteration, (2*I - 2**2*tau*Al)**(-1) -> (2*I - tau*Al)**(-1)
            h: (gi+gj)/(2*h**2), which represents the grid size
            sigma: Standard deviation for Gaussian kernel
            lamda: if gradient>lamda, it'll be regarded as edges
            pro_num: number of processes
            
        return:
            diffed_images_set: a set of diffused images
    """
    start_time = time.time()
    res_set = []
    pool = multiprocessing.Pool(processes=pro_num)
    for i in range(pro_num) :
        start = int(np.ceil( len(image_set)/pro_num)*(i) )
        stop = int(np.ceil( len(image_set)/pro_num)*(i+1) )
        res_set.append(pool.apply_async(Filter_list,
                                    args=(image_set[start:stop], Ftype,
                                          N, K, diffuse_function, gamma, )))
    pool.close()
    pool.join()
    
    # get the result from different processes
    diffed_images_set = []
    for i in range(pro_num) :
        diffed_images_set = diffed_images_set + res_set[i].get()
    diffed_images_set = np.array(diffed_images_set)
    print("complete the multiprocess of filtering,   cost time: ", get_cost_time( time.time()-start_time ))
    
    return diffed_images_set


def get_cost_time(diff) :
    second = diff % 60
    minute = diff % 3600
    minute = minute // 60
    hour = diff % 86400
    hour = hour // 3600
    day = diff // 86400
    format_string="{}d {}h:{}m:{}s"
    return format_string.format(day, hour, minute, second)