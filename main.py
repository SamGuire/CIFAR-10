import numpy as np
from keras.datasets import cifar10
from keras import Sequential
from keras.utils import to_categorical
from keras.layers import Conv2D,Activation,MaxPooling2D,Flatten,Dense,Dropout,BatchNormalization
import matplotlib.pyplot as plt


model = Sequential()
(train_images, train_labels),(test_images,test_labels) = cifar10.load_data()


train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)




'''plt.imshow(train_images[0])
print(train_labels[0])
plt.show() '''





def create_model():
    model = Sequential()


    model.add(Conv2D(32,(3,3),padding = 'same',input_shape = (32,32,3)))
    model.add(BatchNormalization())
    model.add(Activation('elu'))
    model.add(Conv2D(32,(3,3),padding = 'same'))
    model.add(BatchNormalization())
    model.add(Activation('elu'))
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

    model.add(Flatten())
    model.add(BatchNormalization())
    model.add(Dense(300))
    model.add(Activation('elu'))
    model.add(Dense(10))
    model.add(Activation('softmax'))

    model.compile(loss = 'categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

    return model




model = create_model()
model.fit(train_images,train_labels,256,30,1,validation_data=(test_images,test_labels))