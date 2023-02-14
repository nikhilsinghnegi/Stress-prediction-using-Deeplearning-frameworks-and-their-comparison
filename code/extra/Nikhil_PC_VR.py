#%% imports
from warnings import filters
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from time import time 

print("done imports")
#%%for running it on specified GPU
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID";
##print(os.environ["CUDA_DEVICE_ORDER"]) #printing created environmental variable's(key) value
os.environ["CUDA_VISIBLE_DEVICES"] = "0";

##import pprint 
##env_var = os.environ  #TO PRINT ENVIRONMET VARIABLES
##pprint.pprint(dict(env_var),width = 1)   

Config=tf.compat.v1.ConfigProto(allow_soft_placement=True)
Config.gpu_options.allow_growth=True # dynamically grow the memory used on the GPU
 

print("No of gpu's available:", len(tf.config.experimental.list_physical_devices('GPU')))
tf.test.is_built_with_cuda()

# Just disables the warning, doesn't take advantage of AVX/FMA to run faster
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print("done with GPU selecting")
#%% Loading the Data and Preprocessing
#grid size
rectangle_x = 0.01
rectangle_y = 0.01

#dataset size
data_set = 300

#Dimensions for input data
n_x = 79 #the carttesian map divisions
n_y = 79

grid_dist_x = rectangle_x/n_x
grid_dist_y = rectangle_y/n_y

m_x = int(n_x + 1)
m_y = int(n_y + 1)

#initialize data matrices
data_cart_new = np.zeros((data_set,m_x,m_y)) #geometry matrix
data_stress_new = np.zeros((data_set,m_x,m_y)) #stress matrix
one_cart_new = np.zeros((data_set,m_x,m_y)) #Uni-geometry matrix for clean module
one_stress_new = np.zeros((data_set,m_x,m_y)) #Uni-stress matrix for clean module
data_property = np.zeros((data_set,m_x,m_y)) #property matrix

#record time
start_1 = time()

#reading data from the ABAQUS analysis results files
for i in range(data_set):
    idx = str(i+1)
    txt_cart = 'D:\\8th sem\\BTP_II\\Compressed\\DiNN_Model_Data\\fix_hole_hollow_small\\Composite_uniform_SDF_Cart_'+ idx + '.dat'
    txt_stress = 'D:\\8th sem\\BTP_II\\Compressed\\DiNN_Model_Data\\fix_hole_hollow_small\\Composite_uniform_Stress_Cart_'+ idx + '.dat' 
    one_cart = 'D:\\8th sem\\BTP_II\\Compressed\\DiNN_Model_Data\\fix_hole_hollow_small\\One_SDF_cart_'+ idx + '.dat'
    one_stress = 'D:\\8th sem\\BTP_II\\Compressed\\DiNN_Model_Data\\fix_hole_hollow_small\\One_Stress_cart_'+ idx + '.dat'


    data_cart = np.loadtxt(txt_cart)
    data_stress = np.loadtxt(txt_stress)
    one_cart = np.loadtxt(one_cart)
    one_stress = np.loadtxt(one_stress)
    
    (m,n) = data_cart.shape
    
    for j in range(m):
        x = round(data_cart[j][0]/grid_dist_x)
        y = round(data_cart[j][1]/grid_dist_y)
        data_cart_new[i][y][x] = data_cart[j][2]  
        data_stress_new[i][y][x] = data_stress[j][2]
        one_cart_new[i][y][x] = one_cart[j][2]
        one_stress_new[i][y][x] = one_stress[j][2]

print("done with data loading")
    
#%% train test split random
X_train, X_test, Y_train, Y_test, one_cart_train, one_cart_test, one_stress_train, one_stress_test = train_test_split(data_cart_new, data_stress_new, one_cart_new, one_stress_new, test_size=0.2, random_state = 1)
X_test, X_cv, Y_test, Y_cv, one_cart_test, one_cart_cv, one_stress_test, one_stress_cv = train_test_split(X_test, Y_test, one_cart_test, one_stress_test, test_size = 0.5, random_state = 1) 

#print(X_train.shape)
#print(Y_train.shape)

#reshaping data into matrices for neural network training (#sample, X, Y, feature)
one_cart_train = tf.reshape(one_cart_train, [-1, m_x, m_y, 1])
#print(one_cart_train.shape)
one_stress_train = tf.reshape(one_stress_train, [-1, m_x, m_y, 1])
one_cart_test    = tf.reshape(one_cart_test,   [-1, m_x, m_y, 1])
one_stress_test  = tf.reshape(one_stress_test, [-1, m_x, m_y, 1])
one_cart_cv      = tf.reshape(one_cart_cv,     [-1, m_x, m_y, 1])
one_stress_cv    = tf.reshape(one_stress_cv,   [-1, m_x, m_y, 1])

input_train      = tf.reshape(X_train, [-1, m_x, m_y, 1])
output_train     = tf.reshape(Y_train, [-1, m_x, m_y, 1])
input_test  = tf.reshape(X_test, [-1, m_x, m_y, 1])
output_test = tf.reshape(Y_test, [-1, m_x, m_y, 1])
input_cv  = tf.reshape(X_cv, [-1, m_x, m_y, 1])
output_cv = tf.reshape(Y_cv, [-1, m_x, m_y, 1])

# Taking mean as reference geometry and reference stress contours (taking mean along axis 0, meaning for all dataset values)
sdf_ave = tf.reduce_mean(input_train, 0)
#print(sdf_ave.shape)
sdf_ave = tf.reshape(sdf_ave, [-1, m_x, m_y, 1])
#print(sdf_ave.shape)
stress_ave = tf.reduce_mean(output_train, 0)
stress_ave = tf.reshape(stress_ave, [-1, m_x, m_y, 1])

# calculating the geometry and stress difference contours for training, CV, test set
input_train_new = input_train - sdf_ave
output_train_new = output_train - stress_ave
# making ave value tensors to match with the size of input (axis 0 (of avg. contour) value '1' will get multiplied with a value '80')
[a1,b1,c1,d1] = input_train.shape 
stress_ave_train = tf.keras.backend.repeat_elements(stress_ave, rep = a1, axis = 0)

input_cv_new = input_cv - sdf_ave
[a2,b2,c2,d2] = input_cv.shape 
stress_ave_cv = tf.keras.backend.repeat_elements(stress_ave, rep = a2, axis = 0)

input_test_new = input_test - sdf_ave
[a3,b3,c3,d3] = input_test.shape 
stress_ave_test = tf.keras.backend.repeat_elements(stress_ave, rep = a3, axis = 0)

#%% Normalization module(chages from the original)
max_sdf     = np.max(input_train_new)
max_stress  = np.max(output_train_new)
min_sdf     = np.min(input_train_new)
min_stress  = np.min(output_train_new)

print(max_sdf)
print(min_sdf)
print(max_stress)
print(min_stress)

#normalizing
input_train_new = (input_train_new-min_sdf)/(max_sdf - min_sdf)
input_train_new = tf.math.multiply(input_train_new, one_cart_train) #for the clean module

input_cv_new = (input_cv_new-min_sdf)/(max_sdf - min_sdf)
input_cv_new = tf.math.multiply(input_cv_new, one_cart_cv) #for the clean module

input_test_new = (input_test_new-min_sdf)/(max_sdf - min_sdf)
input_test_new = tf.math.multiply(input_test_new, one_cart_test) #for the clean module
# taking the preprocessing time
t1 = time() - start_1
# start recording the training time
start_2 = time()

#%% according to me normalization
max_sdf1 = np.max(data_cart_new-tf.reduce_mean(data_cart_new, 0))
min_sdf1 = np.min(data_cart_new-tf.reduce_mean(data_cart_new, 0))

max_stress1 = np.max(data_stress_new-tf.reduce_mean(data_stress_new, 0))
min_stress1 = np.min(data_stress_new-tf.reduce_mean(data_stress_new, 0))

print("new ones:",max_sdf1)
print(min_sdf1)
print(max_stress1)
print(min_stress1)
#normalizing
input_train_new = (input_train_new-min_sdf1)/(max_sdf1 - min_sdf1)
input_train_new = tf.math.multiply(input_train_new, one_cart_train)

input_cv_new = (input_cv_new-min_sdf1)/(max_sdf1 - min_sdf1)
input_cv_new = tf.math.multiply(input_cv_new, one_cart_cv)

input_test_new = (input_test_new-min_sdf1)/(max_sdf1 - min_sdf1)
input_test_new = tf.math.multiply(input_test_new, one_cart_test)
# taking the preprocessing time
t1 = time() - start_1
# start recording the training time
start_2 = time()

print("done with data preprocessing")
#%% defining neural network different blocks

def conv_relu_block(x, filt, names):
    y = tf.keras.layers.Conv2D(filters=filt, kernel_size=[2,2], strides = 2, padding='same', activation='linear', use_bias=True, name = names)(x)
    y = tf.keras.layers.ReLU()(y)
    y = tf.keras.layers.BatchNormalization()(y)
    return y

def se_block(x,filt,ratio=16):
    
    init = x
    se_shape = (1, 1, filt)
    
    se = tf.keras.layers.GlobalAveragePooling2D()(init)
    se = tf.keras.layers.Reshape(se_shape)(se)
    se = tf.keras.layers.Dense(filt // ratio, activation='relu', 
                               kernel_initializer='he_normal', 
                               use_bias=False)(se)
    se = tf.keras.layers.Dense(filt, activation='sigmoid', 
                               kernel_initializer='he_normal', 
                               use_bias=False)(se)
    se = tf.keras.layers.multiply([init, se])
    
    return se

def resnet_block(x,filt):

    y = tf.keras.layers.Conv2D(filters=filt, kernel_size=[3,3], 
                               padding='same', activation='linear', 
                               use_bias=True)(x)
    y = tf.keras.layers.ReLU()(y)
    y = tf.keras.layers.BatchNormalization()(y)
    
    y = tf.keras.layers.Conv2D(filters=filt, kernel_size=[3,3], 
                               padding='same', activation='linear',
                               use_bias=True)(y)
    y = tf.keras.layers.ReLU()(y)
    y = tf.keras.layers.BatchNormalization()(y)

    y = se_block(y,filt)
     
    y = tf.keras.layers.Add()([y,x])
    
    return y

def deconv_norm_linear(x,filt,kernel,stride,names):
    
    y = tf.keras.layers.Conv2DTranspose(filters=filt,kernel_size=kernel,
        strides=stride,padding='same',activation='linear', use_bias=True,
        name=names)(x)
    
    y = tf.keras.layers.Activation(activation='linear')(y)
    
    y = tf.keras.layers.BatchNormalization()(y)

    return y

def dense_block(x,filt,names):
    
    y = tf.keras.layers.Dense(filt, activation='linear',
                              kernel_initializer='he_normal', use_bias=False,
                              name=names)(x)
    
    return y

print("defining DiNN blocks")

#%% Encoder-Decoder Neural Network Structure
input_layer_1 = tf.keras.Input(shape= (m_x, m_y, 1), dtype=tf.float32)
input_layer_2 = tf.keras.Input(shape= (m_x, m_y, 1), dtype=tf.float32)
input_layer_3 = tf.keras.Input(shape= (m_x, m_y, 1), dtype=tf.float32)

conv_1 = conv_relu_block(input_layer_1, 32, 'conv1')
se_1 = se_block(conv_1, 32)

conv_2 = conv_relu_block(se_1, 64, 'conv2')
se_2 = se_block(conv_2, 64)

conv_3 = conv_relu_block(se_2, 128, 'conv3')
se_3 = se_block(conv_3, 128)

resnet_1 = resnet_block(se_3, 128)
resnet_2 = resnet_block(resnet_1, 128)
resnet_3 = resnet_block(resnet_2, 128)
resnet_4 = resnet_block(resnet_3, 128)
resnet_5 = resnet_block(resnet_4, 128)

deconv_0 = deconv_norm_linear(resnet_5, 128, [2,2], (2,2), 'deconv0') 
deconv_1 = deconv_norm_linear(deconv_0, 64, [2,2], (2,2), 'deconv1')
deconv_2 = deconv_norm_linear(deconv_1, 32, [2,2], (2,2), 'deconv2')
deconv_3 = deconv_norm_linear(deconv_2, 1, [2,2], (1,1), 'deconv3')

#denormalizing
deconv_3 = deconv_3*(max_stress1-min_stress1)+ min_stress1
#deconv_3 = deconv_3*(max_stress-min_stress)+ min_stress

#we consider that large stress concentration exists if the stress
#ratio defined as Rσ = σmax/σmean is larger than 2

add = deconv_3 + input_layer_2
#followed by 2D De-Convolution block to smooth the prediction

deconv_4 = deconv_norm_linear(add, 1, [2,2], (1,1), 'deconv4')
deconv_4 = tf.keras.layers.ReLU()(deconv_4)
deconv_4 = tf.math.multiply(deconv_4, input_layer_3)

output_layer = deconv_4

model = tf.keras.models.Model(inputs = [input_layer_1, input_layer_2, input_layer_3], outputs = output_layer)

model.summary()

print("setting up NN")
#%% training the DiNN 
# optimiser
sgd = tf.keras.optimizers.SGD(learning_rate=1e-3, decay=1e-6, momentum=0.6, nesterov=True)

# Compile the model
model.compile(optimizer=sgd, loss=tf.keras.losses.mean_squared_error, metrics=['accuracy'])

epoch = 80
# Fit (Train) the model
history = model.fit([input_train_new, stress_ave_train, one_stress_train], output_train, batch_size=256, epochs=epoch, 
                    steps_per_epoch=40, validation_data=([input_cv_new, stress_ave_cv, one_stress_cv], output_cv))

# Evaluate the model on test set
predict = model.predict([input_test_new, stress_ave_test, one_stress_test])

# Evaludate the model on test set
score = model.evaluate([input_test_new, stress_ave_test, one_stress_test], output_test, verbose=1)
print('\n', 'Test accuracy', score)

# Record Neural Network Training and Prediction Time
t2 = time() - start_2 

print("done with training and prediction")
#%% Generating history plots of training

# Summarize history for accuracy
fig_acc = plt.figure(1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy in training')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
fig_acc.savefig('training_accuracy.png')

# Summarize history for loss
fig_loss = plt.figure(2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
fig_loss.savefig('training_loss.png')


print("finally done")