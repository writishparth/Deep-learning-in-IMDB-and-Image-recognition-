# using CNN
# Import packages 
import tensorflow as tf
import argparse
from tensorflow import keras
from keras.layers import Dense, Embedding, Conv1D, Dropout, GlobalMaxPooling1D
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.optimizers import Adam
from keras.datasets import imdb
from keras.callbacks import TensorBoard, ModelCheckpoint
from time import time


#initiliazation of network  parameters
max_features = 5000
maxlen = 250
embedding_dims = 16
filters = 100
kernel_size = 10
hidden_size = 100

#loading and splitting dataset
(x_train, y_train), (x_test,y_test) = imdb.load_data(num_words = max_features,)

#applying padding to data
x_train = sequence.pad_sequences(x_train, maxlen = maxlen)
x_test = sequence.pad_sequences(x_test, maxlen = maxlen)

#function for 1st network
def network_one(combination, learning_rate, epochs, batches, seed):
    
    print("Combination One with learning rate: {} epochs: {} and batch size: {}".format(learning_rate, epochs, batches))
    model = Sequential()
    #Embedding layer
    model.add(Embedding(max_features, embedding_dims, input_length = maxlen))
    #Convolution Layer
    model.add(Conv1D(filters, kernel_size, padding='valid', activation = 'relu', strides = 1))
    #GlobalMax Pooling
    model.add(GlobalMaxPooling1D())
    #hidden layer
    model.add(Dense(300, activation = 'relu'))
    model.add(Dropout(0.3))
    #hidden layer
    model.add(Dense(300, activation = 'relu'))
    #Single unit output layer
    model.add(Dense(1, activation = 'sigmoid'))
    model.compile(optimizer = Adam(lr=learning_rate),  loss = 'binary_crossentropy', metrics = ['acc'])
    #log creating
    tensorboard = TensorBoard(log_dir= './logs/run-{}'.format(time()))
    filepath='weights.imdb.hdf5'
    #creating checkpoint
    checkpoint= ModelCheckpoint(filepath,monitor='val_acc',verbose=1, save_best_only=True, mode='max')
    callbacks_list=[checkpoint,tensorboard]
    #fitting model
    model.fit(x_train, y_train, batch_size = batches, epochs = epochs,callbacks=callbacks_list,verbose=1, validation_split=0.25)
    #saving model    
    model.save('./imdb-{}-{}-{}-{}-{}.ckpt'.format(combination, learning_rate, epochs, batches, seed))
    #evaluation of model 
    score=model.evaluate(x_test,y_test,verbose=1)
    print('Model Accuraccy on Test Data', score[1])
    #loading and evaluating best model from checkpoint file 
    new_mod=keras.models.load_model('./imdb-{}-{}-{}-{}-{}.ckpt'.format(combination, learning_rate, epochs, batches, seed))
    new_mod.load_weights('weights.imdb.hdf5')
    new_mod.summary()
    score=new_mod.evaluate(x_test,y_test,verbose=1)
    print('Model Accuraccy on Test Data', score[1])
    
#function for model 2   
def network_two(combination, learning_rate, epochs, batches, seed):

    print("Combination Two with learning rate: {} epochs: {} and batch size: {}".format(learning_rate, epochs, batches))
    model = Sequential()
    #Embedding layer
    model.add(Embedding(max_features, embedding_dims, input_length = maxlen))
    #Convolution Layer
    model.add(Conv1D(filters, kernel_size, padding='valid', activation = 'relu', strides = 1))
    model.add(Conv1D(filters, kernel_size, padding='valid', activation = 'relu', strides = 1))
    #GlobalMax Pooling
    model.add(GlobalMaxPooling1D())
    #hidden layer
    model.add(Dense(500, activation = 'relu'))
    model.add(Dropout(0.30))
    #Single unit output layer
    model.add(Dense(1, activation = 'sigmoid'))
    
    model.compile(optimizer = Adam(lr=learning_rate),  loss = 'binary_crossentropy', metrics = ['acc'])
    
    #log creating
    tensorboard = TensorBoard(log_dir= './logs/run-{}'.format(time()))
    filepath='weights.best.hdf5_combination2'
    
    #creating checkpoint
    checkpoint= ModelCheckpoint(filepath,monitor='val_acc',verbose=1, save_best_only=True, mode='max')
    callbacks_list=[checkpoint,tensorboard]
    #fitting model
    model.fit(x_train, y_train, batch_size = batches, epochs = epochs,callbacks=callbacks_list,verbose=1, validation_split=0.25)
    model.save('./imdb-{}-{}-{}-{}-{}.ckpt'.format(combination, learning_rate, epochs, batches, seed))
    #evaluation of mode
    score=model.evaluate(x_test,y_test,verbose=1)
    print('Model Accuraccy on Test Data', score[1])
    #loading and evaluating best model from checkpoint file 
    new_mod=keras.models.load_model('./imdb-{}-{}-{}-{}-{}.ckpt'.format(combination, learning_rate, epochs, batches, seed))
    new_mod.load_weights('weights.best.hdf5_combination2')
    new_mod.summary()
    score=new_mod.evaluate(x_test,y_test,verbose=1)
    print('Model Accuraccy on Test Data', score[1])
    
    
def main(combination, learning_rate, epochs, batches, seed):

    # Set Seed

    print("Seed: {}".format(seed))



    if int(combination)==1:

        network_one(combination, learning_rate, epochs, batches, seed)

    if int(combination)==2:

        network_two(combination, learning_rate, epochs, batches, seed)

    print("Done!")

def check_param_is_numeric(param, value):

    try:
        if(param == 'batches' or param == 'epochs' or param == 'combination'):
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