
import numpy as np
from keras.datasets import cifar10
from keras import Sequential
from keras.utils import to_categorical
from keras.layers import Conv2D,Activation,MaxPooling2D,Flatten,Dense,Dropout,BatchNormalization
import matplotlib.pyplot as plt


# Initialize the training and testing data
(train_images, train_labels),(test_images,test_labels) = cifar10.load_data()


# Convert the training and testing labels (vector/integer) to a binary (0 or 1) matrix. Ex : [6] ---> [0,0,0,0,0,1,0,0,0,0]
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)


def create_model():

    # Initialize stack of layers for the neural network 
    model = Sequential()


    # Add a convolutional layer with 32 3x3 filters with a zero padding to guarantee that the output size is the same as the input size
    model.add(Conv2D(32,(3,3),padding = 'same',input_shape = (32,32,3)))
    model.add(BatchNormalization())
    model.add(Activation('elu'))
    model.add(Conv2D(32,(3,3),padding = 'same'))
    model.add(BatchNormalization())
    model.add(Activation('elu'))
    # Max pooling : Dividing the input map into sets of rectangles, in this case, a 2x2 square and outputting the max input to the output.
    #               Reduces the resolution while keeping the key features required for classification.
    model.add(MaxPooling2D((2,2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(64,(3,3),padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(64,(3,3),padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2,2)))
    model.add(Dropout(0.5))

    
    model.add(Conv2D(128,(3,3),padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(128,(3,3),padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2,2)))
    model.add(Dropout(0.5))

    # Flatten the input. Ex : [0,1,2]
    #                         [3,4,5]  ----- > [0,1,2,3,4,5,6,7,8]
    #                         [6,7,8]
    model.add(Flatten())
    model.add(BatchNormalization())
    # Add a flatten layer of size 300 
    model.add(Dense(300))
    model.add(Activation('elu'))
    model.add(Dense(10))
    model.add(Activation('softmax'))

    # Compile model with a cross-entropy loss function to evaluate how close the predicted output is to the original output, a adam optimizer to
    # iteratively updates the learning rate for different inputs and a evaluation metric of accuracy.
    model.compile(loss = 'categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

    return model




model = create_model()

# Train with a batch size of 256 (256 images) and epoch (repeat after 50000 images) of 30
model.fit(train_images,train_labels,256,30,validation_data=(test_images,test_labels))

model.evaluate(test_images,test_labels,256)

