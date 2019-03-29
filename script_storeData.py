# coding: utf-8

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import argparse
import DataLoader as D

parser = argparse.ArgumentParser()
parser.add_argument("-r", "--root", type=str, help="the root of dataset")
parser.add_argument("-p", "--path", type=str, help="the path of dataset")
parser.add_argument("-name", "--name", type=str, help="the path of saved file")
parser.add_argument("-num", "--num_step", default=10, type=int, help="the number of seperation")
parser.add_argument("-c", "--color", default="rgb", type=str, help="the color of target image")
parser.add_argument("-s", "--size", default=64, type=int, help="the size of target image, a single integral")
parser.add_argument("-filter", "--filter_type", default=None, help="the type of the filter, eg, 'ND'")

args = parser.parse_args()
print("The path is '{}'".format(args.path))
print(args.name, args.num_step)


CASIA = D.DataLoaderCASIA(args.root, args.path)
video_set, group_index_label = CASIA.get_data(imgType=args.color, imgSize=(args.size,args.size),
                                              filter_type=args.filter_type, filter_paras={"N":5, "K":-1, "diffuse_function":'exp', "gamma":0.15},
                                              separator=".")

train_saver = D.Saver(args.name)
train_saver.save(video_set, group_index_label, num_step=args.num_step)

