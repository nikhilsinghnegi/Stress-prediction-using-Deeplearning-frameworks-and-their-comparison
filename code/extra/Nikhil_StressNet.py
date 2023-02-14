from google.colab import drive
drive.mount('/content/gdrive')
!unzip gdrive/My\ Drive/my\ academics/BTP_2/PC-VR.zip

#%% Loading the Data and Preprocessing
#grid size
rectangle_x = 0.01
rectangle_y = 0.01

#dataset size
data_set = 1000

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

data_property = np.zeros((data_set,m_x,m_y)) #property matrix

#record time
start_1 = time()

#reading data from the ABAQUS analysis results files
for i in range(data_set):
    idx = str(i+1)
    txt_cart = '//content//fix_hole_hollow_small//Composite_uniform_SDF_Cart_'+ idx + '.dat'
    txt_stress = '//content//fix_hole_hollow_small//Composite_uniform_Stress_Cart_'+ idx + '.dat' 
    
    data_cart = np.loadtxt(txt_cart)
    data_stress = np.loadtxt(txt_stress)
   
    (m,n) = data_cart.shape
    
    for j in range(m):
        x = round(data_cart[j][0]/grid_dist_x)
        y = round(data_cart[j][1]/grid_dist_y)
        data_cart_new[i][y][x] = data_cart[j][2]  
        data_stress_new[i][y][x] = data_stress[j][2]
        

print("done with data loading")

#%% train test split random
X_train, X_test, Y_train, Y_test = train_test_split(data_cart_new, data_stress_new, test_size=0.2, random_state = 1)
X_test, X_cv, Y_test, Y_cv = train_test_split(X_test, Y_test, test_size = 0.5, random_state = 1) 

#reshaping data into matrices for neural network training (#sample, X, Y, feature) "-1(unknown dimension) inferred to be 800 here" "row wise flattening"
input_train      = tf.reshape(X_train, [-1, m_x, m_y, 1])
output_train     = tf.reshape(Y_train, [-1, m_x, m_y, 1])
input_test  = tf.reshape(X_test, [-1, m_x, m_y, 1])
output_test = tf.reshape(Y_test, [-1, m_x, m_y, 1])
input_cv  = tf.reshape(X_cv, [-1, m_x, m_y, 1])
output_cv = tf.reshape(Y_cv, [-1, m_x, m_y, 1])

#%% is there they are doing normalization

#%% StressNet architecture

#%% defining neural network different blocks

def conv_relu_block(x, filt, names):
    y = tf.keras.layers.Conv2D(filters=filt, kernel_size=[2,2], strides = 2, padding='same', activation='linear', use_bias=True, name = names)(x)
    y = tf.keras.layers.BatchNormalization()(y)
    y = tf.keras.layers.ReLU()(y)
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

def deconv_ReLU(x,filt,kernel,stride,names):
    
    y = tf.keras.layers.Conv2DTranspose(filters=filt,kernel_size=kernel,
        strides=stride,padding='same',activation='linear', use_bias=True,
        name=names)(x)
    
    y = tf.keras.layers.Activation(activation='ReLU')(y)
    
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


conv_1 = conv_relu_block(input_layer_1, 32, 'conv1')

conv_2 = conv_relu_block(conv_1, 64, 'conv2')

conv_3 = conv_relu_block(conv_2, 128, 'conv3')

resnet_1 = resnet_block(conv_3, 128)
resnet_2 = resnet_block(resnet_1, 128)
resnet_3 = resnet_block(resnet_2, 128)
resnet_4 = resnet_block(resnet_3, 128)
resnet_5 = resnet_block(resnet_4, 128)

deconv_1 = deconv_ReLU(resnet_5, 64, [2,2], (2,2), 'deconv1')
deconv_2 = deconv_ReLU(deconv_1, 32, [2,2], (2,2), 'deconv2')
deconv_3 = deconv_ReLU(deconv_2, 1, [2,2], (1,1), 'deconv3')



output_layer = deconv_3

model = tf.keras.models.Model(inputs = [input_layer_1], outputs = output_layer)

model.summary()

print("setting up NN")




#%% training the DiNN 
# optimiser
sgd = tf.keras.optimizers.SGD(learning_rate=1e-3, decay=1e-6, momentum=0.6, nesterov=True)

# Compile the model
model.compile(optimizer=sgd, loss=tf.keras.losses.mean_squared_error, metrics=['accuracy' ])

epoch = 150
# Fit (Train) the model
history = model.fit([input_train], output_train, batch_size=256, epochs=epoch,\
                    validation_data=([input_cv], output_cv))

# Evaluate the model on test set
predict = model.predict([input_test])

# Evaludate the model on test set
score = model.evaluate([input_test], output_test, verbose=1)
print('\n', 'Test accuracy', score)

# Record Neural Network Training and Prediction Time
t2 = time() - start_2 

print("done with training and prediction")
#%% training the DiNN 


#%% Generating history plots of training

# Summarize history for accuracy
fig_acc = plt.figure()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy in training')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
fig_acc.savefig('training_accuracy.png')

# Summarize history for loss
fig_loss = plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
fig_loss.savefig('training_loss.png')


print("finally done")