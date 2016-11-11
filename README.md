# CAPTCHA_by_py-faster-rcnn

Utilize the faster-rcnn framework to identify captcha

- This repo must be megered with py-faster-rcnn and caffe-fast-rcnn
- Installation guide
  1. First git clone py-faster-rcnn recursively with caffe-fast-rcnn, https://github.com/rbgirshick/py-faster-rcnn
  2. Merge newest version of caffe with caffe-fast-rcnn. And do some minor changes (http://leanote.com/s/57f4aff1fa57a473f0000000)
  3. Make py-faster-rcnn/lib and caffe and pycaffe (http://leanote.com/s/578cdc759e77c8467a000000)
  4. Git clone this repository and merge it with py-faster-rcnn
- Usage
  1. Use CAPTCHA_Generator to make training data...
  2. Place the training data in py-faster-rcnn/data directory
  3. Use ./train_captcha.sh to start training...
  4. Use captcha_detect_flask.py in py-faster-rcnn/tools to try a captcha by yourself!
