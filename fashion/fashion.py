# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 13:39:05 2019

@author: artemis
"""

import argparse
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Conv2D,MaxPooling2D,Dropout
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint
from keras.optimizers import adam
from time import time

#loading dataset
df = keras.datasets.fashion_mnist

#splitting dataset
(x_train, y_train), (x_test, y_test) = df.load_data()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

#Normalization of data
x_train=x_train/255
x_test=x_test/255

# Reshape input data from (28, 28) to (28, 28, 1)
w, h = 28, 28
x_train = x_train.reshape(x_train.shape[0], w, h, 1)
x_test = x_test.reshape(x_test.shape[0], w, h, 1)

#function for 1st network
def network_one(combination,learning_rate, epochs, batches,seed):


    print("Combination One with learning rate: {} epochs: {} and batch size: {}".format(learning_rate, epochs, batches))


    model=Sequential()
    #Convolution layer with 100 filter, with 5x5 kernal size
    model.add(Conv2D(100, (5,5), input_shape=(28,28,1)))
    model.add(Activation("relu"))
    #Pooling layer
    model.add(MaxPooling2D(pool_size=(2)))
    #Flattenign the input
    model.add(Flatten())
    #hidden layer
    model.add(Dense(256, activation='relu'))
    #output layer
    model.add(Dense(10, activation='softmax'))
        
    model.compile(optimizer = adam(lr=learning_rate),  loss = 'sparse_categorical_crossentropy', metrics = ['acc'])
    #creating logs 
    logd = "./logs/run-{}".format(time())
    tensorboard = TensorBoard(log_dir=logd)
    #creating checkpoints
    filepath='fashion_weights_best-{}-{}-{}-{}-{}.hdf5'.format(combination, learning_rate, epochs, batches,seed)
    checkpoint= ModelCheckpoint(filepath,monitor='val_acc',verbose=1, save_best_only=True, mode='max')
    callbacks_list=[checkpoint,tensorboard]
    #fitting the model    
    model.fit(x_train, y_train, batch_size = batches, epochs = epochs, validation_split=0.25,callbacks=callbacks_list,verbose=1)
    score=model.evaluate(x_test,y_test,verbose=1)
    #saving the model
    model.save('./fashion-{}-{}-{}-{}-{}.ckpt'.format(combination, learning_rate, epochs, batches,seed))
    #importing best model to validate on test data
    new_mod=keras.models.load_model('./fashion-{}-{}-{}-{}-{}.ckpt'.format(combination, learning_rate, epochs, batches,seed))
    new_mod.load_weights('fashion_weights_best-{}-{}-{}-{}-{}.hdf5'.format(combination, learning_rate, epochs, batches,seed))
    new_mod.summary()
    score=new_mod.evaluate(x_test,y_test,verbose= 1)
    print('Model Accuraccy on Test Data', score[1])
    
#function for 2nd network
def network_two(combination, learning_rate, epochs, batches,seed):



    print("Combination Two with learning rate: {} epochs: {} and batch size: {}".format(learning_rate, epochs, batches))
    model=Sequential()
    #Convolution layer with 100 filter, with 5x5 kernal size
    model.add(Conv2D(100, (5,5), input_shape=(28,28,1)))
    model.add(Activation("relu"))
    #Pooling layer
    model.add(MaxPooling2D(pool_size=(2)))
    #Flattenign the input
    model.add(Flatten())
    #hidden layer with 600 neurons and with a 20% drop out rate
    model.add(Dense(600, activation='relu'))
    model.add(Dropout(0.2))
    #output layer
    model.add(Dense(10, activation='softmax'))
    model.compile(optimizer = adam(lr=learning_rate),  loss = 'sparse_categorical_crossentropy', metrics = ['acc'])
    #creating logs 
    logd = "./logs/run-{}".format(time())
    tensorboard = TensorBoard(log_dir=logd)
    #creating checkpoints
    filepath='fashion_weights_best-{}-{}-{}-{}-{}.hdf5'.format(combination, learning_rate, epochs, batches,seed)
    checkpoint= ModelCheckpoint(filepath,monitor='val_acc',verbose=1, save_best_only=True, mode='max')
    callbacks_list=[checkpoint,tensorboard]
    #fitting the model
    model.fit(x_train, y_train, batch_size = batches, epochs = epochs, validation_split=0.25,callbacks=callbacks_list,verbose=1)
    score=model.evaluate(x_test,y_test,verbose=1)
    #saving the model
    model.save('./fashion-{}-{}-{}-{}-{}.ckpt'.format(combination, learning_rate, epochs, batches,seed))
    #importing best model to validate on test data
    new_mod=keras.models.load_model('./fashion-{}-{}-{}-{}-{}.ckpt'.format(combination, learning_rate, epochs, batches,seed))
    new_mod.load_weights('fashion_weights_best-{}-{}-{}-{}-{}.hdf5'.format(combination, learning_rate, epochs, batches,seed))
    new_mod.summary()
    score=new_mod.evaluate(x_test,y_test,verbose=1)
    print('Model Accuraccy on Test Data', score[1])
    



def main(combination, learning_rate, epochs, batches, seed):



    # Set Seed

    print("Seed: {}".format(seed))



    if int(combination)==1:

        network_one(combination,learning_rate, epochs, batches,seed)

    if int(combination)==2:

        network_two(combination,learning_rate, epochs, batches,seed)



    print("Done!")



def check_param_is_numeric(param, value):



    try:

        if(param == 'batches' or param == 'epochs'):
            value = int(value)
        else:
            value = float(value)

    except:

        print("{} must be numeric".format(param))

        quit(1)

    return value





if __name__ == "__main__":



    arg_parser = argparse.ArgumentParser(description="Assignment Program")

    arg_parser.add_argument("combination", help="Flag to indicate which network to run")

    arg_parser.add_argument("learning_rate", help="Learning Rate parameter")

    arg_parser.add_argument("iterations", help="Number of iterations to perform")

    arg_parser.add_argument("batches", help="Number of batches to use")

    arg_parser.add_argument("seed", help="Seed to initialize the network")



    args = arg_parser.parse_args()



    combination = check_param_is_numeric("combination", args.combination)

    learning_rate = check_param_is_numeric("learning_rate", args.learning_rate)

    epochs = check_param_is_numeric("epochs", args.iterations)

    batches = check_param_is_numeric("batches", args.batches)

    seed = check_param_is_numeric("seed", args.seed)



    main(combination, learning_rate, epochs, batches, seed)