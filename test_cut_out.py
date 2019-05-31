#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019-06-01 03:27
# @Author  : moiling
# @File    : test_cut_out.py
from utils.foreground_background import estimate_foreground_background
from utils.utils import stack_images, save_image, show_image
import matplotlib.pyplot as plt


if __name__ == '__main__':

    file_name = 'troll.png'

    image = plt.imread('./data/input_lowres/' + file_name)
    alpha = plt.imread('./out/ifm/' + file_name)

    foreground, background = estimate_foreground_background(image, alpha, print_info=True)

    # Make new image from foreground and alpha
    cutout = stack_images(foreground, alpha)

    # save
    save_image(cutout, './out/cut_out/' + 'ifm' + '/', file_name)

    # show
    show_image(cutout)

