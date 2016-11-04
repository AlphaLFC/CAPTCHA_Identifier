#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 09:30:42 2016

@author: alpha
"""

import _init_paths
from fast_rcnn.config import cfg, cfg_from_file
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
import numpy as np
import scipy.io as sio
import os, sys, cv2
import string
import pprint
import argparse
import caffe

CLASSES = tuple(['__background__'] +
                list(string.digits + string.lowercase + string.uppercase))
ROOT_PATH = '/data2/py-faster-rcnn'
CAFFEMODEL = os.path.join(ROOT_PATH,
                          'output/my_faster_rcnn/train',
                          'skynet_faster_rcnn_iter_10000.caffemodel')
CFG_FILE = os.path.join(ROOT_PATH,
                        'experiments/cfgs/my_faster_rcnn.yml')
PROTOTXT = os.path.join(ROOT_PATH,
                        'models/captcha/SKYNET/my_faster_rcnn/test.prototxt')
NET = caffe.Net(PROTOTXT, CAFFEMODEL, caffe.TEST)


def detect_captcha(img_file):
    cfg_from_file(CFG_FILE)
    img = cv2.imread(img_file)
    scores, boxes = im_detect(NET, img)
    captcha = []
    for cls_idx, cls in enumerate(CLASSES[1:]):
        cls_idx += 1
        cls_boxes = boxes[:, 4*cls_idx:4*(cls_idx+1)]
        cls_scores = scores[:, cls_idx]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis],
                          np.full((len(cls_scores), 1), cls_idx, dtype=np.int32))).astype(np.float32)
        # if the captcha image have more than one char, use this
        # print dets.shape, dets[:, :5].shape
        keep = nms(dets[:, :5], 0.6)
        dets = dets[keep, :]
        if cls_idx == 1:
            data = dets
        else:
            data = np.vstack([data, dets])
        dets = dets[:, :5]
        #dets = dets[np.where(dets[:, 4] > 0.6)]
        #if len(dets) != 0:
        #    chars = [(bbox, cls) for bbox in dets[:, :4]]
        #    for char in chars:
        #        captcha.append(char)
        char = dets[np.argmax(dets[:, 4])]
        if char[4] > 0.6:
            captcha.append((char[:4], cls)) 
    data = data[np.where(data[:, 4] > 0.2)]
    data_keep = nms(data[:, :5], 0.35)
    data = data[data_keep, :]
    data = np.array(sorted(data, key=lambda x: x[0]))
    #print data[:, 5]
    str_idx = list(data[:, 5].astype(np.int32))
    new_str = ''.join([CLASSES[x] for x in str_idx])
    captcha = np.array(sorted(captcha, key=lambda x: x[0][0]))
    captcha_str = ''.join(captcha[:, 1])
    return captcha_str, new_str

if __name__ == '__main__':
    test_img = '../test.png'
    result = detect_captcha(test_img)
    print result
