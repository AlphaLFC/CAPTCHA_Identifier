# CAPTCHA_by_py-faster-rcnn

Ultize the faster-rcnn framework to identify captcha

- This repo must be megered with py-faster-rcnn and caffe-fast-rcnn
- Installation guide
  1. First git clone py-faster-rcnn recursively with caffe-fast-rcnn
  2. Merge newest version of caffe with caffe-fast-rcnn. And do some minor changes (see notes in my leanote, adding soon...)
  3. Make py-faster-rcnn/lib and caffe and pycaffe (see tips in my leanote, adding soon...)
  4. Git clone this repository and merge it with py-faster-rcnn
- Usage
  1. Use CAPTCHA_Generator to make training data...
  2. Place the training data in py-faster-rcnn/data directory
  3. Use ./train_captcha.sh to start training...
  4. Use captcha_detect_flask.py in py-faster-rcnn/tools to try a captcha by yourself!
