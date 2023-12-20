import keras
import matplotlib.pyplot as plt
from keras import layers
from keras import models
from keras.preprocessing.image import ImageDataGenerator


trainDataPath = 'drive/MyDrive/datasetCarsimulation 2.0/train'
valDataPath = 'drive/MyDrive/datasetCarsimulation 2.0/validation'


trainDataGenerator = ImageDataGenerator(rescale = 1./255,
                                       rotation_range=40,
                                       width_shift_range=0.2,
                                       height_shift_range=0.2,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True)


valDataGenerator = ImageDataGenerator(rescale = 1./255)


trainGenerator = trainDataGenerator.flow_from_directory(trainDataPath,
                                                       target_size=(150, 150),
                                                       batch_size=1,
                                                       class_mode='categorical')


valGenerator = valDataGenerator.flow_from_directory(valDataPath,
                                                   target_size=(150, 150),
                                                   batch_size=1,
                                                   class_mode='categorical')


model = models.Sequential()


model.add(layers.Conv2D(32,(3,3), activation='relu', input_shape=(150,150,3)))
model.add(layers.MaxPooling2D((2,2)))


model.add(layers.Conv2D(64,(3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))


model.add(layers.Conv2D(128,(3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))


model.add(layers.Conv2D(128,(3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))


model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))


#model outputs percentage over 4 classes
model.add(layers.Dense(4, activation='softmax'))


model.summary()


model.compile(optimizer='rmsprop',
             loss='categorical_crossentropy',
             metrics=['accuracy'])


history = model.fit(trainGenerator,
                   steps_per_epoch=200,
                   epochs=100,
                   validation_data=valGenerator,
                   validation_steps=100)


model.save('FinalVersionCarsimNetwork.h5')


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']


epochs = range(len(acc))


plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()


plt.figure()


plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()


plt.show()