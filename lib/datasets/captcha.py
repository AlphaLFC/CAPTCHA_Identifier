#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 09:43:16 2016

@author: alpha
"""

import os
import errno
import string
from datasets.imdb import imdb
import numpy as np
import pandas as pd
import scipy.sparse
import cPickle
import uuid
from captcha_eval import captcha_eval


class captcha(imdb):
    def __init__(self, image_set, devkit_path):
        imdb.__init__(self, image_set)
        self._image_set = image_set
        self._devkit_path = devkit_path
        self._data_path = self._devkit_path  # yeah!
        self._classes = tuple(['__background__'] +  # always index 0
                              list(string.digits + string.lowercase +
                                   string.uppercase))
        self._class_to_idx = dict(zip(self.classes,
                                      xrange(self.num_classes)))
        self._image_ext = ['.jpg', '.png']
        self._image_index = self._load_image_set_index()
        self._salt = str(uuid.uuid4())
        self._comp_id = 'comp4'

        self.config = {'cleanup': True,
                       'use_salt': True,
                       'top_k': 2000,
                       'use_diff': False,
                       'rpn_file': None}
        assert os.path.exists(self._devkit_path), \
            'Devkit path does not exist: {}'.format(self._devkit_path)
        assert os.path.exists(self._data_path), \
            'Path does not exist: {}'.format(self._data_path)

    def image_path_at(self, i):
        ''' Return absolute path to image i. '''
        return self.image_path_from_index(self._image_index[i])

    def image_path_from_index(self, index):
        ''' Construct an image path from the image's index identifier. '''
        for ext in self._image_ext:
            image_path = os.path.join(self._data_path, 'Images',
                                      index+ext)
            if os.path.exists(image_path):
                break
        assert os.path.exists(image_path), \
            'Path does not exist: {}'.format(image_path)
        return image_path

    def _load_image_set_index(self):
        image_set_file = os.path.join(self._data_path, 'ImageSets',
                                      self._image_set+'.txt')
        assert os.path.exists(image_set_file), \
            'Path does not exist: {}'.format(image_set_file)
        with open(image_set_file) as f:
            image_index = [x.strip() for x in f.readlines()]
        return image_index

    def gt_roidb(self):
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} gt roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        gt_roidb = [self._load_captcha_annotation(index)
                    for index in self.image_index]
        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote gt roidb to {}'.format(cache_file)
        return gt_roidb

    def rpn_roidb(self):
        gt_roidb = self.gt_roidb()
        rpn_roidb = self._load_rpn_roidb(gt_roidb)
        roidb = imdb.merge_roidbs(gt_roidb, rpn_roidb)
        return roidb

    def _load_rpn_roidb(self, gt_roidb):
        filename = self.config['rpn_file']
        print 'loading {}'.format(filename)
        assert os.path.exists(filename), \
            'rpn data not found at: {}'.format(filename)
        with open(filename, 'rb') as f:
            box_list = cPickle.load(f)
        return self.create_roidb_from_box_list(box_list, gt_roidb)

    def _load_captcha_annotation(self, index):
        """
        Load image and bounding boxes info from a txt file of a captcha.
        """
        filename = os.path.join(self._data_path, 'Annotations', index + '.txt')
        objs = pd.read_table(filename, header=None, delim_whitespace=True)
        objs = objs.values.tolist()
        num_objs = len(objs)
        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
        # "Seg" area here is just the box area
        seg_areas = np.zeros((num_objs), dtype=np.float32)
        # Load object bounding boxes into a data frame.
        for idx, obj in enumerate(objs):
            boxes[idx, :] = obj[:4]
            cls = self._class_to_idx[str(obj[4])]
            gt_classes[idx] = cls
            overlaps[idx, cls] = 1.0
            seg_areas[idx] = (obj[2]-obj[0]+1) * (obj[3]-obj[1]+1)
        overlaps = scipy.sparse.csr_matrix(overlaps)
        return {'boxes': boxes,
                'gt_classes': gt_classes,
                'gt_overlaps': overlaps,
                'flipped': False,
                'seg_areas': seg_areas}

    def _write_captcha_results_file(self, all_boxes):
        for cls_idx, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            print 'Writing {} results file'.format(cls)
            filename = self._get_captcha_results_file_template().format(cls)
            with open(filename, 'wt') as f:
                for im_idx, index in enumerate(self.image_index):
                    dets = all_boxes[cls_idx][im_idx]
                    if dets == []:
                        continue
                    for k in xrange(dets.shape[0]):
                        f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                format(index, dets[k, -1],
                                       dets[k, 0], dets[k, 1],
                                       dets[k, 2], dets[k, 3]))

    def evaluate_detections(self, all_boxes, output_dir):
        self._write_captcha_results_file(all_boxes)
        self._do_python_eval(output_dir)
        if self.config['cleanup']:
            for cls in self._classes:
                if cls == '__background__':
                    continue
                filename = self._get_captcha_results_file_template().format(cls)
                os.remove(filename)

    def _get_comp_id(self):
        comp_id = (self._comp_id + '_' + self._salt if self.config['use_salt']
                   else self._comp_id)
        return comp_id

    def _get_captcha_results_file_template(self):
        filename = self._get_comp_id() + '_det_' + self._image_set + '_{:s}.txt'
        try:
            os.mkdir(self._devkit_path + '/results')
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise e
        path = os.path.join(
            self._devkit_path,
            'results',
            filename)
        return path

    def _do_python_eval(self, output_dir='output'):
        annopath = os.path.join(self._data_path,
                                'Annotations',
                                '{:s}.txt')
        imagesetfile = os.path.join(self._data_path,
                                    'ImageSets',
                                    self._image_set + '.txt')
        cachedir = os.path.join(self._devkit_path, 'annotations_cache')
        aps = []
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        for i, cls in enumerate(self._classes):
            if cls == '__background__':
                continue
            filename = self._get_captcha_results_file_template().format(cls)
            rec, prec, ap = captcha_eval(filename,
                                         annopath,
                                         imagesetfile,
                                         cls,
                                         cachedir,
                                         ovthresh=0.5)
            aps.append(ap)
            print('AP for {} = {:.4f}'.format(cls, ap))
            with open(os.path.join(output_dir, cls + '_pr.pkl'), 'w') as f:
                cPickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
        print('Mean AP = {:.4f}'.format(np.mean(aps)))
        print('~~~~~~~~')
        print('Results:')
        for ap in aps:
            print('{:.3f}'.format(ap))
        print('{:.3f}'.format(np.mean(aps)))
