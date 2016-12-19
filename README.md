# Detect CAPTCHA by py-faster-rcnn

Utilize the faster-rcnn framework to identify captcha, This repo must be megered with py-faster-rcnn and caffe-fast-rcnn

---

**Installation guide**

- First git clone py-faster-rcnn recursively with caffe-fast-rcnn, https://github.com/rbgirshick/py-faster-rcnn
- Merge newest version of caffe with caffe-fast-rcnn. And do some minor changes (http://leanote.com/s/57f4aff1fa57a473f0000000)
- Make py-faster-rcnn/lib and caffe and pycaffe (http://leanote.com/s/578cdc759e77c8467a000000)
- Git clone this repository and merge it with py-faster-rcnn

---

**Usage**

- Use CAPTCHA_Generator to make training data...
- Place the training data in py-faster-rcnn/data directory
- Use ./train_captcha.sh to start training...
- Use captcha_detect_flask.py in py-faster-rcnn/tools to try a captcha by yourself!
