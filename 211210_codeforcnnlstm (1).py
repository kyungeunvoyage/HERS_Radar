#!/usr/bin/env python
# coding: utf-8

# In[4]:


import matplotlib.pyplot as plt
import numpy as np 
import tensorflow as tf
from sklearn import metrics
from tqdm import tqdm
import glob
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import *
from keras.models import Model
from keras.layers import Dense
from keras.layers import Input
from keras.layers import Flatten
from keras.layers import Conv2D, Conv3D, MaxPool2D
from tensorflow.keras.layers import LayerNormalization
from keras.layers.normalization import BatchNormalization
from keras.layers import Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import plot_model
import datetime
import pickle
import torch
import imageio
# set the directory
import os
path = os.getcwd()
os.chdir(path)


# In[5]:


#load the feature and labels, 24066, 8033, and 7984 frames for train, validate, and test
featuremap_train = np.load('feature/featuremap_train.npy')
featuremap_validate = np.load('feature/featuremap_validate.npy')
featuremap_test = np.load('feature/featuremap_test.npy')
featuremap_train = featuremap_train[:, :5,:5, :3]
featuremap_validate = featuremap_validate[:, :5,:5, :3]
featuremap_test = featuremap_test[:, :5,:5, :3]

labels_train = np.load('feature/Train_label_25_arr.npy')
labels_validate = np.load('feature/Val_label_25_arr.npy')
labels_test = np.load('feature/Test_label_25_arr.npy')

# featuremap_train_1 = featuremap_train[:, :5,:5, :3]
# featuremap_validate_1 = featuremap_validate[:, :5,:5, :3]
# featuremap_test_1 = featuremap_test[:, :5,:5, :3]
# featuremap_train_2 = featuremap_train[:, :5,:5, 4:5]
# featuremap_validate_2 = featuremap_validate[:, :5,:5, 4:5]
# featuremap_test_2 = featuremap_test[:, :5,:5, 4:5]
# featuremap_train3  = np.concatenate((featuremap_train_1,featuremap_train_2), axis=3)
# featuremap_validate3  = np.concatenate((featuremap_validate_1,featuremap_validate_2), axis=3)
# featuremap_test3  = np.concatenate((featuremap_test_1,featuremap_test_2), axis=3)
# featuremap_train = featuremap_train3
# featuremap_validate = featuremap_validate3
# featuremap_test = featuremap_test3

print(featuremap_train.shape)
print(featuremap_validate.shape)
print(featuremap_test.shape)
print(labels_train.shape)
print(labels_validate.shape)
print(labels_test.shape)


# ### 1. Load Model

# In[6]:


# Initialize the result array
paper_result_list = []

# define batch size and epochs
batch_size = 64
epochs = 100


# In[7]:



# #define the model - 1. original
# def define_CNN(in_shape, n_keypoints):


#     in_one = Input(shape=in_shape)
#     conv_one_1 = Conv2D(16, kernel_size=(3, 3), activation='relu', strides=(1, 1), padding = 'same')(in_one)
#     conv_one_1 = Dropout(0.3)(conv_one_1)
#     conv_one_2 = Conv2D(32, kernel_size=(3, 3), activation='relu', strides=(1, 1), padding = 'same')(conv_one_1)
#     conv_one_2 = Dropout(0.3)(conv_one_2)

    
#     conv_one_2 = BatchNormalization(momentum=0.95)(conv_one_2)


#     fe = Flatten()(conv_one_2)
#     # dense1
#     dense_layer1 = Dense(512, activation='relu')(fe)
#     dense_layer1 = BatchNormalization(momentum=0.95)(dense_layer1)
#     # # dropout

#     # dropout
#     dense_layer1 = Dropout(0.4)(dense_layer1)
    
#     out_layer = Dense(n_keypoints, activation = 'linear')(dense_layer1)
    

#     # model
#     model = Model(in_one, out_layer)
#     opt = Adam(lr=0.001, beta_1=0.5)

#     # compile the model
#     model.compile(loss='mse', optimizer=opt, metrics=['mae', 'mse', 'mape', tf.keras.metrics.RootMeanSquaredError()])
#     return model


# In[8]:



# #define the model - 2.layer 추가

# def define_CNN(in_shape, n_keypoints):
#     in_one = Input(shape=in_shape)
#     conv_one_1 = Conv2D(16, kernel_size=(3, 3), activation='relu', strides=(1, 1), padding = 'same')(in_one)
#     conv_one_1 = Dropout(0.3)(conv_one_1)
#     conv_one_2 = Conv2D(32, kernel_size=(3, 3), activation='relu', strides=(1, 1), padding = 'same')(conv_one_1)
#     conv_one_2 = Dropout(0.3)(conv_one_2)
#     conv_one_3 = Conv2D(64, kernel_size=(3, 3), activation='relu', strides=(1, 1), padding = 'same')(conv_one_2)
#     conv_one_3 = Dropout(0.3)(conv_one_3)
    
#     conv_one_4 = BatchNormalization(momentum=0.98)(conv_one_3)


#     fe = Flatten()(conv_one_4)
#     # dense1
#     dense_layer1 = Dense(1024, activation='relu')(fe)
#     dense_layer1 = BatchNormalization(momentum=0.98)(dense_layer1)
    
#     # dropout
#     dense_layer2 = Dropout(0.4)(dense_layer1)
#     dense_layer3 = Dropout(0.4)(dense_layer2)
#     out_layer = Dense(n_keypoints, activation = 'linear')(dense_layer3)
    

#     # model
#     model = Model(in_one, out_layer)
#     opt = Adam(lr=0, beta_1=0.5)

#     # compile the model
#     model.compile(loss='mse', optimizer=opt, metrics=['mae', 'mse', 'mape', tf.keras.metrics.RootMeanSquaredError()])
#     return model


# In[9]:


def create_convlstm_model(in_shape, n_keypoints):
    '''
    This function will construct the required convlstm model.
    Returns:
        model: It is the required constructed convlstm model.
    '''
    in_one = Input(shape=in_shape)
    print(in_one.shape)
    print(type(in_one))
    print('fff')
    # We will use a Sequential model for model construction
    model = Sequential()
 
    # Define the Model Architecture.
    ########################################################################################################################
    
    model.add(ConvLSTM2D(filters = 4, kernel_size = (3, 3), activation = 'tanh',data_format = "channels_last",
                         recurrent_dropout=0.2, return_sequences=True, input_shape = (in_one[1],in_one[2],in_one[3],3)))
              
    model.add(MaxPooling3D(pool_size(1,2,2), padding = 'same', data_format='channels_last'))
    model.add(TimeDistributed(Dropout(0.2)))
    
    model.add(ConvLSTM2D(filters = 8, kernel_size = (3, 3), activation = 'tanh', data_format = "channels_last",
                         recurrent_dropout=0.2, return_sequences=True))
    
    model.add(MaxPooling3D(pool_size=(1, 2, 2), padding='same', data_format='channels_last'))
    model.add(TimeDistributed(Dropout(0.2)))
    
    model.add(ConvLSTM2D(filters = 14, kernel_size = (3, 3), activation = 'tanh', data_format = "channels_last",
                         recurrent_dropout=0.2, return_sequences=True))
    
    model.add(MaxPooling3D(pool_size=(1, 2, 2), padding='same', data_format='channels_last'))
    model.add(TimeDistributed(Dropout(0.2)))
    
    model.add(ConvLSTM2D(filters = 16, kernel_size = (3, 3), activation = 'tanh', data_format = "channels_last",
                         recurrent_dropout=0.2, return_sequences=True))
    
    model.add(MaxPooling3D(pool_size=(1, 2, 2), padding='same', data_format='channels_last'))
    #model.add(TimeDistributed(Dropout(0.2)))
    
    model.add(Flatten()) 
    
    model.add(Dense(n_keypoints, activation = "softmax"))
    
    ########################################################################################################################
     
    # Display the models summary.
    model.summary()
    
    # Return the constructed convlstm model.
    return model


# In[10]:


keypoint_model = create_convlstm_model(featuremap_train[0].shape, 75)


# In[ ]:


keypoint_model.summary()


# In[ ]:


# instantiate the model
# keypoint_model = define_CNN(featuremap_train[0].shape, 57)
# keypoint_model = define_CNN(featuremap_train[0].shape, 75)
keypoint_model = create_convlstm_model(featuremap_train[0].shape, 75)

# initial maximum error 
score_min = 10
history = keypoint_model.fit(featuremap_train, labels_train,
                             batch_size=batch_size, epochs=epochs, verbose=1, 
                             validation_data=(featuremap_validate, labels_validate))
# save and print the metrics
score_train = keypoint_model.evaluate(featuremap_train, labels_train,verbose = 1)
print('train MAPE = ', score_train[3])
score_test = keypoint_model.evaluate(featuremap_test, labels_test,verbose = 1)
print('test MAPE = ', score_test[3])
result_test = keypoint_model.predict(featuremap_test)

# Plot accuracy
plt.plot(history.history['mae'])
plt.plot(history.history['val_mae'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Xval'], loc='upper left')
plt.show()

# Plot loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Xval'], loc='upper left')
plt.xlim([0,100])
plt.ylim([0,0.1])
plt.show()


# ### 2. Get_pcl_data

# In[ ]:


# # => for instensity
# def get_pcl_data_(file_path):
#     file = open(file_path, "r")
#     radar_pcls = file.readlines()
#     cnt = 0
#     frame = 0
#     pcl = 0
#     pcl_data = []
#     x_data = []
#     y_data = []
#     z_data = []
#     inten_data = []

#     total_x = []
#     total_y = []
#     total_z = []
#     total_inten = []

#     for i in range(7, len(radar_pcls)):


#         if 'x:' in radar_pcls[i]:
#             x_int = radar_pcls[i][3:-1]
#             x_int_data = float(x_int)
#             x_data.append(x_int_data)

#         elif 'y:' in radar_pcls[i] and 'velocity:' not in radar_pcls[i] and 'intensity:' not in radar_pcls[i]:
#             y_int = radar_pcls[i][3:-1]
#             y_int_data = float(y_int)
#             y_data.append(y_int_data)
#         elif 'z:' in radar_pcls[i]:
#             z_int = radar_pcls[i][3:-1]
#             z_int_data = float(z_int)
#             z_data.append(z_int_data)
#         elif 'intensity' in radar_pcls[i]:
#             inten_int = radar_pcls[i][11:-1]
#             inten_int_data = float(inten_int)
#             inten_data.append(inten_int_data)


#         elif(radar_pcls[i] == 'point_id: 0\n'):
#             total_x.append(x_data)  
#             total_y.append(y_data)
#             total_z.append(z_data) 
#             total_inten.append(inten_data)
#             x_data = []
#             y_data = []
#             z_data = []
#             inten_data = []

#     for idx in range(len(total_x)):
#         if (len(total_x[idx])) < 25:
#             zero_list = [0] * (25 - len(total_x[idx]))
#             total_x[idx] = total_x[idx] +zero_list
#             total_y[idx] = total_y[idx] +zero_list
#             total_z[idx] = total_z[idx] +zero_list
#             total_inten[idx] = total_inten[idx] +zero_list
#         else:
#             total_x[idx] = total_x[idx][:25]
#             total_y[idx] = total_y[idx][:25]
#             total_z[idx] = total_z[idx][:25]       
#             total_inten[idx] = total_inten[idx][:25]

#     if(len(total_x)>=600):
#         total_x = total_x[0:600]
#         total_y = total_y[0:600]
#         total_z = total_z[0:600]
#         total_inten = total_inten[0:600]

#     else:    
#         while (len(total_x) < 600):
#             total_x.append(total_x[-1])
#             total_y.append(total_y[-1])
#             total_z.append(total_z[-1])
#             total_inten.append(total_inten[-1])

#     total_x = np.array(total_x)
#     total_y = np.array(total_y)
#     total_z = np.array(total_z) 
#     total_inten = np.array(total_inten)

#     total_x = total_x.reshape(600,-1,1)
#     total_y = total_y.reshape(600,-1,1)
#     total_z = total_z.reshape(600,-1,1)     
#     total_inten = total_inten.reshape(600,-1,1)     

#     final = []
#     for i in range(len(total_x)):
#         a = np.concatenate([total_x[i], total_y[i],total_z[i], total_inten[i]], axis=1)
# #         a = np.concatenate([total_x[i], total_y[i],total_z[i]], axis=1)
#         final.append(a)
#     final = np.array(final)
#     final = final.reshape(600,5,5,4)   
# #     final = final.reshape(600,5,5,3)  
        
#     return final  



# => intensity
def get_pcl_data_(file_path):
    file = open(file_path, "r")
    radar_pcls = file.readlines()
    cnt = 0
    frame = 0
    pcl = 0
    pcl_data = []
    x_data = []
    y_data = []
    z_data = []
#     inten_data = []

    total_x = []
    total_y = []
    total_z = []
#     total_inten = []

    for i in range(7, len(radar_pcls)):


        if 'x:' in radar_pcls[i]:
            x_int = radar_pcls[i][3:-1]
            x_int_data = float(x_int)
            x_data.append(x_int_data)

        elif 'y:' in radar_pcls[i] and 'velocity:' not in radar_pcls[i] and 'intensity:' not in radar_pcls[i]:
            y_int = radar_pcls[i][3:-1]
            y_int_data = float(y_int)
            y_data.append(y_int_data)
        elif 'z:' in radar_pcls[i]:
            z_int = radar_pcls[i][3:-1]
            z_int_data = float(z_int)
            z_data.append(z_int_data)
#         elif 'intensity' in radar_pcls[i]:
#             inten_int = radar_pcls[i][11:-1]
#             inten_int_data = float(inten_int)
#             inten_data.append(inten_int_data)


        elif(radar_pcls[i] == 'point_id: 0\n'):
            total_x.append(x_data)  
            total_y.append(y_data)
            total_z.append(z_data) 
#             total_inten.append(inten_data)
            x_data = []
            y_data = []
            z_data = []
#             inten_data = []

    for idx in range(len(total_x)):
        if (len(total_x[idx])) < 25:
            zero_list = [0] * (25 - len(total_x[idx]))
            total_x[idx] = total_x[idx] +zero_list
            total_y[idx] = total_y[idx] +zero_list
            total_z[idx] = total_z[idx] +zero_list
#             total_inten[idx] = total_inten[idx] +zero_list
        else:
            total_x[idx] = total_x[idx][:25]
            total_y[idx] = total_y[idx][:25]
            total_z[idx] = total_z[idx][:25]       
#             total_inten[idx] = total_inten[idx][:25]

    if(len(total_x)>=600):
        total_x = total_x[0:600]
        total_y = total_y[0:600]
        total_z = total_z[0:600]
#         total_inten = total_inten[0:600]

    else:    
        while (len(total_x) < 600):
            total_x.append(total_x[-1])
            total_y.append(total_y[-1])
            total_z.append(total_z[-1])
#             total_inten.append(total_inten[-1])

    total_x = np.array(total_x)
    total_y = np.array(total_y)
    total_z = np.array(total_z) 
#     total_inten = np.array(total_inten)

    total_x = total_x.reshape(600,-1,1)
    total_y = total_y.reshape(600,-1,1)
    total_z = total_z.reshape(600,-1,1)     
#     total_inten = total_inten.reshape(600,-1,1)     

    final = []
    for i in range(len(total_x)):
#         a = np.concatenate([total_x[i], total_y[i],total_z[i], total_inten[i]], axis=1)
        a = np.concatenate([total_x[i], total_y[i],total_z[i]], axis=1)
        final.append(a)
    final = np.array(final)

#     final = final.reshape(600,5,5,4)   
    final = final.reshape(600,5,5,3)  
        
    return final  


# In[ ]:


def pre_process_pcl(result):
    total_data =[]
    x_preprocessing = []
    y_preprocessing = []
    z_preprocessing = []
    
    for idx_r in range(len(result)):
        x_ = []
        y_ = []
        z_ = []
        for idx in range(len(result[idx_r][0:25])):
            x_.append([result[idx_r][0:25][idx], 0.0])
            y_.append([result[idx_r][25:50][idx], 0.0])
            z_.append([result[idx_r][50:][idx], 0.0])
        x_preprocessing.append(x_)    
        y_preprocessing.append(y_)   
        z_preprocessing.append(z_)  
    total_data.append([x_preprocessing ,y_preprocessing,  z_preprocessing])    
    total_data = np.array(total_data)
    return total_data


# In[ ]:


now = datetime.datetime.now()
nowDatetime = now.strftime('%Y%m%d_%H_%M_%S')
nowDatetime  = nowDatetime[2:]
print(nowDatetime)  


# ### boxing

# In[ ]:


#. First
data_ = get_pcl_data_('C:\\Users\\user\\Desktop\\KY_folder\\Data\\Train\\boxing\\boxing_20_7.txt')
result = keypoint_model.predict(data_)
total_data = pre_process_pcl(result)
total_data.shape

filenames = []
for i in range(len(result)):
    fig = plt.figure()

    #플로팅 하려은 좌표를 3D로 지정
    ax = fig.gca(projection='3d')
    # colors = ['salmon', 'orange', 'steelblue']    

    ax.set_xlim3d(-1.0, 1.0)
    ax.set_ylim3d(0.0, 3.0)
    ax.set_zlim3d(-1.0, 1.0)

    # ax.scatter(df['x'], df['y'], color = 'hotpink')
    # ax.scatter(df['y'], df['z'], color = 'pink')
    # ax.scatter(df['x'], df['z'], color = 'purple')
    ax.scatter(result[i][0:25], result[i][25:50], result[i][50:], color = 'green')

    ax.set_xlabel('X', fontsize=10, fontstyle='oblique')
    ax.set_ylabel('Y', fontsize=10, fontstyle='oblique')
    ax.set_zlabel('Z', fontsize=10, fontstyle='oblique')
    # plt.savefig('boxing.png')
#     plt.show()
    filename = f'{i}_.png'
    filenames.append(filename)
    plt.savefig(filename)

boxing = nowDatetime + '_25_Radar_boxing_5x5_' + str(batch_size)+'_'+ str(epochs) + '.gif'

with imageio.get_writer(boxing, mode='I') as writer:
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)    
        
# Remove files
for filename in set(filenames):
    os.remove(filename)        


# ## jack

# In[ ]:


#. First
data_ = get_pcl_data_('C:\\Users\\user\\Desktop\\KY_folder\\Data\\Train\\jack\\jacks_20_6.txt')
result_jack = keypoint_model.predict(data_)
total_data = pre_process_pcl(result_jack)
total_data.shape



filenames = []
for i in range(len(result_jack)):
    fig = plt.figure()

    #플로팅 하려은 좌표를 3D로 지정
    ax = fig.gca(projection='3d')
    # colors = ['salmon', 'orange', 'steelblue']    

    ax.set_xlim3d(-1.0, 1.0)
    ax.set_ylim3d(0.0, 3.0)
    ax.set_zlim3d(-1.0, 1.0)

    # ax.scatter(df['x'], df['y'], color = 'hotpink')
    # ax.scatter(df['y'], df['z'], color = 'pink')
    # ax.scatter(df['x'], df['z'], color = 'purple')
    ax.scatter(result_jack[i][0:25], result_jack[i][25:50], result_jack[i][50:], color = 'green')

    ax.set_xlabel('X', fontsize=10, fontstyle='oblique')
    ax.set_ylabel('Y', fontsize=10, fontstyle='oblique')
    ax.set_zlabel('Z', fontsize=10, fontstyle='oblique')
    # plt.savefig('boxing.png')
#     plt.show()
    filename = f'{i}_.png'
    filenames.append(filename)
    plt.savefig(filename)

jack = nowDatetime + '_25_Radar_jack_5x5_' + str(batch_size)+'_'+ str(epochs) + '.gif'

with imageio.get_writer(jack, mode='I') as writer:
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)    
        
# Remove files
for filename in set(filenames):
    os.remove(filename)        


# ## Jump

# In[ ]:


#. First
data_ = get_pcl_data_('C:\\Users\\user\\Desktop\\KY_folder\\Data\\Train\\jump\\___jump_20_5.txt')
result = keypoint_model.predict(data_)
total_data = pre_process_pcl(result)
total_data.shape

filenames = []
for i in range(len(result)):
    fig = plt.figure()

    #플로팅 하려은 좌표를 3D로 지정
    ax = fig.gca(projection='3d')
    # colors = ['salmon', 'orange', 'steelblue']    

    ax.set_xlim3d(-1.0, 1.0)
    ax.set_ylim3d(0.0, 3.0)
    ax.set_zlim3d(-1.0, 1.0)

    # ax.scatter(df['x'], df['y'], color = 'hotpink')
    # ax.scatter(df['y'], df['z'], color = 'pink')
    # ax.scatter(df['x'], df['z'], color = 'purple')
    ax.scatter(result[i][0:25], result[i][25:50], result[i][50:], color = 'green')

    ax.set_xlabel('X', fontsize=10, fontstyle='oblique')
    ax.set_ylabel('Y', fontsize=10, fontstyle='oblique')
    ax.set_zlabel('Z', fontsize=10, fontstyle='oblique')
    # plt.savefig('boxing.png')
#     plt.show()
    filename = f'{i}_.png'
    filenames.append(filename)
    plt.savefig(filename)


jump = nowDatetime + '_25_Radar_jump_5x5_' + str(batch_size)+'_'+ str(epochs) + '.gif'    

with imageio.get_writer(jump, mode='I') as writer:
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)    
        
# Remove files
for filename in set(filenames):
    os.remove(filename)            


# ## Squats

# In[ ]:


#. First
data_ = get_pcl_data_('C:\\Users\\user\\Desktop\\KY_folder\\Data\\Train\\squats\\___squats_20_5.txt')
result = keypoint_model.predict(data_)
total_data = pre_process_pcl(result)
total_data.shape

filenames = []
for i in range(len(result)):
    fig = plt.figure()

    #플로팅 하려은 좌표를 3D로 지정
    ax = fig.gca(projection='3d')
    # colors = ['salmon', 'orange', 'steelblue']    

    ax.set_xlim3d(-1.0, 1.0)
    ax.set_ylim3d(0.0, 3.0)
    ax.set_zlim3d(-1.0, 1.0)

    # ax.scatter(df['x'], df['y'], color = 'hotpink')
    # ax.scatter(df['y'], df['z'], color = 'pink')
    # ax.scatter(df['x'], df['z'], color = 'purple')
    ax.scatter(result[i][0:25], result[i][25:50], result[i][50:], color = 'green')

    ax.set_xlabel('X', fontsize=10, fontstyle='oblique')
    ax.set_ylabel('Y', fontsize=10, fontstyle='oblique')
    ax.set_zlabel('Z', fontsize=10, fontstyle='oblique')
    # plt.savefig('boxing.png')
#     plt.show()
    filename = f'{i}_.png'
    filenames.append(filename)
    plt.savefig(filename)
    

squats = nowDatetime + '_25_Radar_squats_5x5_' + str(batch_size)+'_'+ str(epochs) + '.gif'    

with imageio.get_writer(squats, mode='I') as writer:
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)    
        
# Remove files
for filename in set(filenames):
    os.remove(filename)            


# ## Walk

# In[ ]:


#. First
data_ = get_pcl_data_('C:\\Users\\user\\Desktop\\KY_folder\\Data\\Train\\walk\\__20_walk_5.txt')
result = keypoint_model.predict(data_)
total_data = pre_process_pcl(result)
total_data.shape

filenames = []
for i in range(len(result)):
    fig = plt.figure()

    #플로팅 하려은 좌표를 3D로 지정
    ax = fig.gca(projection='3d')
    # colors = ['salmon', 'orange', 'steelblue']    

    ax.set_xlim3d(-1.0, 1.0)
    ax.set_ylim3d(0.0, 3.0)
    ax.set_zlim3d(-1.0, 1.0)

    # ax.scatter(df['x'], df['y'], color = 'hotpink')
    # ax.scatter(df['y'], df['z'], color = 'pink')
    # ax.scatter(df['x'], df['z'], color = 'purple')
    ax.scatter(result[i][0:25], result[i][25:50], result[i][50:], color = 'green')

    ax.set_xlabel('X', fontsize=10, fontstyle='oblique')
    ax.set_ylabel('Y', fontsize=10, fontstyle='oblique')
    ax.set_zlabel('Z', fontsize=10, fontstyle='oblique')
    # plt.savefig('boxing.png')
#     plt.show()
    filename = f'{i}_.png'
    filenames.append(filename)
    plt.savefig(filename)


walk = nowDatetime + '_25_Radar_walk_5x5_' + str(batch_size)+'_'+ str(epochs) + '.gif'    
with imageio.get_writer(walk, mode='I') as writer:
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)    
        
# Remove files
for filename in set(filenames):
    os.remove(filename)            


# ## 3. make npy

# In[ ]:


def pre_process_pcl(result):
    total_data =[]
    x_preprocessing = []
    y_preprocessing = []
    z_preprocessing = []
    
    for idx_r in range(len(result)):
        x_ = []
        y_ = []
        z_ = []
        for idx in range(len(result[idx_r][0:25])):
            x_.append([result[idx_r][0:25][idx], 0.0])
            y_.append([result[idx_r][25:50][idx], 0.0])
            z_.append([result[idx_r][50:][idx], 0.0])
        x_preprocessing.append(x_)    
        y_preprocessing.append(y_)   
        z_preprocessing.append(z_)  
    total_data.append([x_preprocessing ,y_preprocessing,  z_preprocessing])    
    total_data = np.array(total_data)
    return total_data


# ## 4. Loop (Train)

# In[ ]:


parent_dir  = 'C:\\Users\\user\\Desktop\\KY_folder\\direct_Train'
sub_dirs=['boxing','jack','jump','squats','walk']
# sub_dirs=['boxing']
file_ext ='*.txt'


# In[ ]:


#. First
data_ = get_pcl_data_('C:\\Users\\user\\Desktop\\KY_folder\\Data\\Train\\boxing\\20_boxing_1.txt')
result = keypoint_model.predict(data_)
total_data = pre_process_pcl(result)
total_data.shape


# In[ ]:


for sub_dir in tqdm(range(len(sub_dirs))):
    files=sorted(glob.glob(os.path.join(parent_dir,sub_dirs[sub_dir], file_ext)))
    for fn in range(len(files)):
        print(files[fn])
        data_ = get_pcl_data_(files[fn])
        result = keypoint_model.predict(data_)
        total_data_2 = pre_process_pcl(result)
        total_data = np.vstack([total_data, total_data_2])


# In[ ]:


total_data.shape


# In[ ]:


np.save('211117_pcl_MARS_train_OneConvPlus.npy', total_data)


# In[ ]:


pcl_MARS = np.load('211117_pcl_MARS_train_OneConvPlus.npy') 
pcl_MARS.shape


# ## 5. Loop (Test)

# In[ ]:


parent_dir  = 'C:\\Users\\user\\Desktop\\KY_folder\\direct_Test'
sub_dirs=['boxing','jack','jump','squats','walk']
# sub_dirs=['boxing']
file_ext ='*.txt'


# In[ ]:


#. First
data_ = get_pcl_data_('C:\\Users\\user\\Desktop\\KY_folder\\Data\\Test\\boxing\\___20_boxing_5.txt')
result = keypoint_model.predict(data_)
total_data = pre_process_pcl(result)
total_data.shape


# In[ ]:


for sub_dir in tqdm(range(len(sub_dirs))):
    files=sorted(glob.glob(os.path.join(parent_dir,sub_dirs[sub_dir], file_ext)))
    for fn in range(len(files)):
        print(files[fn])
        data_ = get_pcl_data_(files[fn])
        result = keypoint_model.predict(data_)
        total_data_2 = pre_process_pcl(result)
        total_data = np.vstack([total_data, total_data_2])


# In[ ]:


total_data.shape


# In[ ]:


np.save('211123_pcl_MARS_test_OneConvPlus.npy', total_data)


# In[ ]:


pcl_MARS_test = np.load('211123_pcl_MARS_test_OneConvPlus.npy') 
pcl_MARS_test.shape


# In[ ]:


import torch.nn.functional as F

dataset = torch.randn(4)
print(dataset)
dataset_2 = (F.softmax(dataset, dim=0))
max, inx = torch.max(dataset_2, dim=0)


# In[ ]:


inx


# In[ ]:




