#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019-05-30 19:08
# @Author  : moiling
# @File    : utils.py
import os
import matplotlib.pyplot as plt


def save_image(image, save_dir, file_name):
    # mkdir and touch
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(save_dir + file_name):
        os.system(r"touch {}".format(save_dir + file_name))

    plt.imsave(save_dir + file_name, image, cmap='Greys_r')


def show_image(image):
    plt.imshow(image, cmap='Greys_r')
    plt.show()
