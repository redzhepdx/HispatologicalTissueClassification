from flask import Flask, make_response, request, redirect, url_for, send_from_directory, render_template, send_file
from werkzeug import secure_filename
from PIL import Image, ImageDraw
from flask_restful import reqparse, abort, Api, Resource
from io import BytesIO
from classifier import *

import requests
import os
import flask
import PIL.Image
import urllib.request
import random
import cv2

#print(test_y[201:209])
#predict(test_x[201:209], test_y[201:204])

UPLOAD_FOLDER = 'uploads/'
path = '/home/redzhep/codes/Research/HistologyDS/'

ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

app = Flask(__name__, static_url_path='/static')
api = Api(app)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

class Upload(Resource):
    def post(self):
        file = request.files['file']
        if file:
            filename = secure_filename(file.filename)
            print(filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            img = cv2.imread(app.config['UPLOAD_FOLDER'] + filename)
            label = predict([img], 2)
            #label, _ = predict(UPLOAD_FOLDER+filename, sess, bbox_util, x, y)
            
            print(label[0])
            info = {'label' : str(label[0])}
            return info
        else:
            return {'False'}

class GetFeatureMaps(Resource):
    def post(self):
        file = request.files['file']
        if file:
            filename = secure_filename(file.filename)
            print(filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            img = cv2.imread(app.config['UPLOAD_FOLDER'] + filename)
            #label = predict([img], 2)
            #label, _ = predict(UPLOAD_FOLDER+filename, sess, bbox_util, x, y)
            model, c1, c2, c3, c4, c5 = ConvNN(x)
            f1 = getActivations(c1, img, "static/c1/")
            f1.extend(getActivations(c2, img, "static/c2/"))
            f1.extend(getActivations(c3, img, "static/c3/"))
            f1.extend(getActivations(c4, img, "static/c4/"))
            f1.extend(getActivations(c5, img, "static/c5/"))
            #print(label[0])
            #info = {'label' : str(label[0])}
            #return info
            return f1
        else:
            return {'False'}

class TrainModel(Resource):
    def post(self):
        width = int(request.args.get('width'))
        height = int(request.args.get('height'))
        channel = int(request.args.get('channel'))
        hl = int(request.args.get('hl'))
        batch = int(request.args.get('batch'))
        cl = int(request.args.get('class'))
        lr = float(request.args.get('lr'))
        epoch = int(request.args.get('epoch'))
        conv1 = int(request.args.get('conv1'))
        conv2 = int(request.args.get('conv2'))
        conv3 = int(request.args.get('conv3'))
        conv4 = int(request.args.get('conv4'))
        conv5 = int(request.args.get('conv5'))
        conv1o = int(request.args.get('conv1O'))
        conv2o = int(request.args.get('conv2O'))
        conv3o = int(request.args.get('conv3O')) 
        conv4o = int(request.args.get('conv4O'))
        conv5o = int(request.args.get('conv5O'))
        '''
        TrainConvNN(x, lr=lr, epoch_count=epoch, batch_size=batch,
                Image_Height=480, Image_Width=720, num_of_channels=3,
                w1=conv1, w2=conv2, w3=conv3, w4=conv4, w5=conv5, wc1=conv1o, wc2=conv2o, wc3=conv3o, wc4=conv4o, wc5=conv5o,
                hiddenLayerSize=hl, fullyConnectedInputSize=15*10*64, class_count=cl)
        '''
        acc1 = evaluate(test_x[:200], test_y[:200])
        acc2 = evaluate(test_x[200:400], test_y[200:400])
        acc3 = evaluate(test_x[400:500], test_y[400:500])
        
        total_acc = (acc1 * 2) + (acc2 * 2) + acc3
        total_acc /= 5
        ret = {'accuracy' : total_acc}

        #print("{}, {}, {}, {}, {}, {}, {}, {}, {}, {}".format(width, height, channel, hl, batch, cl, lr, conv1, conv2, conv3))
        return ret

class GetMerged(Resource):
    def post(self):
        ret = get_merge(test_x)
        return ret

api.add_resource(Upload, '/predict')
api.add_resource(GetFeatureMaps, '/features')
api.add_resource(GetMerged, '/merged')
api.add_resource(TrainModel, '/train')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
