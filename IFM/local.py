#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019-05-31 18:53
# @Author  : moiling
# @File    : local.py
import cv2
import numpy as np

from IFM.closed_form_matting import compute_weight


def local(image, trimap):
    umask = (trimap != 0) & (trimap != 1)
    w_l = compute_weight(image, mask=umask).tocsr()
    return w_l
