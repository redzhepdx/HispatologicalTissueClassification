import tensorflow as tf
import TCP as tcp
import os
import PIL
import png
import numpy as np
import time
import random
import cv2

train_x, train_y, test_x , test_y =tcp.CreateTrainTestSetsAndLabels("idxrenata2828.txt", percent=0.2)

class_count = 4
batch_size = 50
conv_window_size = 2

Image_Height = 480
Image_Width = 720
num_of_channels = 3

firstFeatureMapSize = 32
secondFeatureMapSize = 64

hiddenLayerSize = 1024

keep_rate = 0.9
#keep_prob = tf.placeholder(tf.float32)

save_dir = 'checkpoints/'
if not os.path.exists(save_dir):
   os.makedirs(save_dir)

#model_variable_scope = "end_to_end"
save_path = os.path.join(save_dir, "endtoend_model")

fulllyConnectedInputSize = 15  * 10  * 64

#saver = tf.train.Saver()
#graph = tf.Graph()
session = tf.Session()
#saver = tf.train.Saver()

def init_variables():
    session.run(tf.global_variables_initializer())

weights = {'conv_1_weights':tf.Variable(tf.random_normal([5,5,3,8]), name='w1'),
               'conv_2_weights':tf.Variable(tf.random_normal([5,5,8,16]), name='w2'),
               'conv_3_weights':tf.Variable(tf.random_normal([5,5,16,32]), name='w3'),
               'conv_4_weights':tf.Variable(tf.random_normal([5,5,32,64]), name='w4'),
               'conv_5_weights':tf.Variable(tf.random_normal([5,5,64,64]), name='w5'),
               'w_fully_connected':tf.Variable(tf.random_normal([int(fulllyConnectedInputSize), hiddenLayerSize]), name='fc'),
               'w_output':tf.Variable(tf.random_normal([hiddenLayerSize,class_count]), name='output')}

biases = {'conv_1_biases':tf.Variable(tf.random_normal([8]), name='b1'),
              'conv_2_biases':tf.Variable(tf.random_normal([16]), name='b2'),
              'conv_3_biases':tf.Variable(tf.random_normal([32]), name='b3'),
              'conv_4_biases':tf.Variable(tf.random_normal([64]), name='b4'),
              'conv_5_biases':tf.Variable(tf.random_normal([64]), name='b5'),
              'b_fully_connected':tf.Variable(tf.random_normal([hiddenLayerSize]), name='b_fc'),
              'b_output':tf.Variable(tf.random_normal([class_count]), name='b_out')}

#with graph.as_default():
for d in ['/gpu:0','/gpu:1','/gpu:2','/gpu:3']:
   with tf.device(d):
       x = tf.placeholder('float', [None, Image_Height, Image_Width, num_of_channels], name='x')
       y = tf.placeholder('float', name='y')
        #with tf.variable_scope(model_variable_scope):
            #predictions = model_pass(x)

#saver = tf.train.Saver()

def Convolution2D(_input, Weights, _name='conv'):
    return tf.nn.conv2d(_input, Weights, strides=[1,1,1,1], padding = 'SAME', name=_name)

def MaxPool2D(_input, _name='maxpool'):
    return tf.nn.max_pool(_input, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME', name=_name)

def MaxPool2DStrides(_input,s_size, _name='maxpool'):
    return tf.nn.max_pool(_input, ksize=[1, s_size, s_size,1], strides = [1, s_size, s_size, 1], padding = 'SAME', name=_name)
	
def ReLU(_input, bias, _name='relu'):
    return tf.nn.relu(_input + bias, name=_name)

def ConvNN(_input, Image_Height=480, Image_Width=720, num_of_channels=3,
                   w1=5, w2=5, w3=5, w4=5, w5=5, 
                   wc1=8, wc2=16, wc3=32, wc4=64, wc5=64,
                   hiddenLayerSize=1024, fullyConnectedInputSize=15*10*64,
                   class_count=4):

    _input = tf.reshape(_input, shape = [-1, Image_Height, Image_Width , num_of_channels])

    #CONVOLUTION LAYER
    conv1 = Convolution2D(_input, weights['conv_1_weights'], _name='conv1')
    conv1 = ReLU(conv1, biases['conv_1_biases'], _name='relu1')
    conv1 = MaxPool2DStrides(conv1,2, _name='mp_1') #360x240x8

    conv2 = Convolution2D(conv1, weights['conv_2_weights'], _name='conv2')
    conv2 = ReLU(conv2, biases['conv_2_biases'], _name='relu2')
    conv2 = MaxPool2D(conv2, _name='mp_2') #180x120x16

    conv3 = Convolution2D(conv2, weights['conv_3_weights'], _name='conv3')
    conv3 = ReLU(conv3, biases['conv_3_biases'], _name='relu3')
    conv3 = MaxPool2D(conv3, _name='mp_3') #90x60x32

    conv4 = Convolution2D(conv3, weights['conv_4_weights'], _name='conv4')
    conv4 = ReLU(conv4, biases['conv_4_biases'], _name='relu4')
    conv4 = MaxPool2D(conv4, _name='mp_4') #45x30x64

    conv5 = Convolution2D(conv4, weights['conv_5_weights'], _name='conv5')
    conv5 = ReLU(conv5, biases['conv_5_biases'], _name='relu5')
    conv5 = MaxPool2DStrides(conv5,3, _name='mp_5') #15x10x64
    

    #FULLY CONNECTED LAYER
    fullyConnected = tf.reshape(conv5, [-1, int(fulllyConnectedInputSize)])
    fullyConnected = ReLU(tf.matmul(fullyConnected, weights['w_fully_connected']), biases['b_fully_connected'], _name='relu_flatten')
#    fullyConnected = tf.nn.dropout(fullyConnected, keep_rate)

    #print(np.array(fullyConnected).shape)
    #OUTPUT

    output = tf.add(tf.matmul(fullyConnected, weights['w_output'], name='output_mult') , biases['b_output'], name='output_add')

    return output, conv1, conv2, conv3, conv4, conv5

saver = tf.train.Saver()

def TrainConvNN(x, lr=0.001, epoch_count=20, batch_size=50,
                Image_Height=480, Image_Width=720, num_of_channels=3,
                w1=5, w2=5, w3=5, w4=5, w5=5, wc1=8, wc2=16, wc3=32, wc4=64, wc5=64,
                hiddenLayerSize=1024, fullyConnectedInputSize=15*10*64, class_count=4):
    #graph = tf.Graph()
    #with graph.as_default():
    for d in ['/gpu:0','/gpu:1','/gpu:2','/gpu:3']:
        with tf.device(d):
            prediction, c1, c2, c3, c4, c5 = ConvNN(x, Image_Height=Image_Height, Image_Width=Image_Width, num_of_channels=num_of_channels,
                                                    w1=w1, w2=w2, w3=w3, w4=w4, w5=w5, wc1=wc1, wc2=wc2, wc3=wc3, wc4=wc4, wc5=wc5,
                                                    hiddenLayerSize=hiddenLayerSize, fullyConnectedInputSize=fullyConnectedInputSize, 
                                                    class_count=class_count)

            cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction,y), name='prediction')
            optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(cost)

    EPOCH_COUNT = epoch_count
    #with session as sess:
    session.run(tf.global_variables_initializer())

    for epoch in range(EPOCH_COUNT):
        loss = 0
        print("In Epoch...")
        chunk_size = 0
        while chunk_size < len(train_x):
            start = chunk_size
            end = chunk_size + batch_size

            batch_x = np.array(train_x[start:end])
            batch_y = np.array(train_y[start:end])

            _, c = session.run([optimizer,cost], feed_dict = {x: batch_x, y: batch_y})
            loss += c

            chunk_size += batch_size

        print('Epoch',epoch + 1, 'Completed with ', EPOCH_COUNT, 'loss : ',loss)
    saver.save(sess=session, save_path=save_path)

config={"EPS": 1e-7}

def image_normalization(image, s = 0.1, ubound = 255.0):
	img_min = np.min(image)
	img_max = np.max(image)
	return (((image - img_min) * ubound) / (img_max - img_min + config["EPS"])).astype('uint8')

def get_prediction(X, session):
    p = []
    batch_iterator = BatchIterator(batch_size=128)
    for x_batch, _ in batch_iterator(X):
        [p_batch] = session.run([predictions], feed_dict={ tf_x_batch : x_batch})
        p.extend(p_batch)
    return p

def getActivations(layer, image, path):
    saver.restore(sess=session, save_path=save_path)
    units = session.run(layer,feed_dict={x:[image]})
    print(units.shape)
    image_names = save_featuremaps(units, path)
    return image_names

def save_featuremaps(units, path):
    filter_count = units.shape[3]
    images = [None] * filter_count
    image_names = []
    for i in range(filter_count):    
        img = np.array(units[0,:,:,i])
        img = image_normalization(img)
        png.from_array(img, 'L').save(path + "featmap-" + str(i) + ".png")
        #img = PIL.Image.fromarray(img)
        #img.convert('L')
        #img.save("featmap" + str(i) + ".png")
        print(units[0,:,:,i].shape)
        image_names.append("http://208.101.10.219:5000/" + path + "featmap-" + str(i) + ".png")
    return image_names

def evaluate(xa, ya):
    saver.restore(sess=session, save_path=save_path)
    
    model, c1, c2, c3, c4, c5 = ConvNN(x)    
    #sess.run(tf.global_variables_initializer())
    correctClassified1 = tf.equal(tf.argmax(model,1),tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correctClassified1,'float'))
    val = accuracy.eval(session=session, feed_dict={x:xa, y:ya})
    print('Accuracy : ', val)
    return val

def predict(xa, ya):
    saver.restore(sess=session, save_path=save_path)
    model, c1, c2, c3, c4, c5 = ConvNN(x)
    pred = tf.argmax(model, 1)
    classification = session.run(pred, feed_dict={x: xa, y:ya})
    print("class {}".format(classification))
    return classification

def predictMerged(img):
    labels = []
    for x in range(0, 730, 240):
        for y in range(0, 490, 240):
            #cv2.imwrite("images/" + str(i) + "-part.jpg", merged[y:y+480, x:x+720])
            pred = predict([img[y:y+480, x:x+720]], 1)
            labels.append(pred)
            print(pred)

def merge_images(img1, img2, img3, img4):
    ret = []
    
    merged_img_h = np.concatenate((img1, img2), axis=1)
    merged_img_h2 = np.concatenate((img3, img4), axis=1)
    merged = np.concatenate((merged_img_h, merged_img_h2), axis=0)
    
    cv2.imwrite("static/merged.png", merged)
    ret.append("http://208.101.10.219:5000/static/merged.png")
    i = 0
    for x in range(0, 730, 240):
        for y in range(0, 490, 240):
            cv2.imwrite("static/parts/" + str(i) + "-part.png", merged[y:y+480, x:x+720])    
            ret.append("http://208.101.10.219:5000/static/parts/" + str(i) + "-part.png")
            i += 1
    return ret

def get_merge(images):
    indexes = random.sample(range(0,len(images)), 4)
    ret = merge_images(images[indexes[0]], images[indexes[1]], images[indexes[2]], images[indexes[3]])
    print(ret)
    return ret

#get_merge(test_x)

#def predictMerged(img):
 

#train_x, train_y, test_x , test_y =tcp.CreateTrainTestSetsAndLabels("idxrenata2828.txt", percent=0.2)
#model, c1, c2, c3, c4, c5 = ConvNN(x)
#start_time = time.time()
#getActivations(c1, test_x[0], "featuremaps/c1/") 
#getActivations(c2, test_x[0], "featuremaps/c2/")
#getActivations(c3, test_x[0], "featuremaps/c3/")
#getActivations(c4, test_x[0], "featuremaps/c4/")
#getActivations(c5, test_x[0], "featuremaps/c5/")
#png.from_array(test_x[0], 'rgb').save("image.png")
#elapsed_time = time.time() - start_time
#print("Time : {}".format(elapsed_time))
'''
TrainConvNN(x, lr=0.001, epoch_count=40, batch_size=50,
                Image_Height=480, Image_Width=720, num_of_channels=3,
                w1=5, w2=5, w3=5, w4=5, w5=5, wc1=8, wc2=16, wc3=32, wc4=64, wc5=64,
                hiddenLayerSize=1024, fullyConnectedInputSize=15*10*64, class_count=4)
'''
#graph = tf.Graph()
#model = ConvNN(x)
#sess.run(tf.global_variables_initializer())
#train_x, train_y, test_x , test_y =tcp.CreateTrainTestSetsAndLabels("idxrenata2828.txt", percent=0.2)
#merge_images(test_x[200], test_x[201], test_x[202], test_x[203], test_y[200], test_y[201], test_y[202], test_y[203])
#evaluate(test_x[:200], test_y[:200])
#evaluate(test_x[200:400], test_y[200:400])
#evaluate(test_x[400:500], test_y[400:500])
#print(test_y[201:209])
#predict(test_x[201:209], test_y[201:204])
