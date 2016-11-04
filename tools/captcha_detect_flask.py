# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 11:59:58 2016

@author: super
"""

from flask import Flask, request, render_template
# from captcha_detector import detect_captcha
from test import detect_captcha
import os

app = Flask(__name__)

@app.route('/', methods=['POST', 'GET'])
def index():
    if request.method == 'POST':
        im = request.files['file']
        path = os.path.join(os.getcwd(), 'static', im.filename)
        im.save(path)
        det_str, new_str = detect_captcha(path)
        det_str = 'Old way prediction: ' + det_str
        new_str = 'New way prediction: ' + new_str
        return render_template('upload.html', results={'img': im.filename, 'str0': det_str,
                                                       'str1': new_str})
    else:
        return render_template('upload.html')
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3985)
