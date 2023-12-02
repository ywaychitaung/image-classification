import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from keras import models
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Conv2D, Dense, MaxPooling2D, Flatten, Dropout
from PIL import Image

image_height = 256
image_width = 256

train_directory = 'Datasets/train_add'
test_directory = 'Datasets/test'
validation_directory = 'Datasets/validation'

'''
Create image classification model
'''
def create_model():
    model = Sequential()

    # create a CNN model
    model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(image_height, image_width, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(units=128, activation='relu'))
    model.add(Dropout(0.5))

    # output layer: performs classification; the image is either ['apple', 'banana', 'mixed' and 'orange']
    # 4 possible classes
    model.add(Dense(units=4, activation='softmax'))

    # build the model
    model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

    return model


'''
Main program.
'''
def main():

    # create our CNN model
    model = create_model()

    # model architecture
    print(model.summary())

    ''' 
        Here, we use a data generator to feed in our data to our model.
        This is useful when we have a large dataset and we do not want
        to load all of it into memory at once. Instead, we can load a
        batch of data at a time.
    '''
    # RGB values are from 0 to 255; rescale each value by 1/255
    ImageFlow = ImageDataGenerator(
        rescale=1/255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    # specify which directory our train data should be read from
    train_generator = ImageFlow.flow_from_directory(
        directory=train_directory,
        target_size=(image_height, image_width),
        batch_size=128,
        class_mode='categorical',
        color_mode='rgb'
    )   

    # specify which directory our validation data should be read from
    validation_generator = ImageFlow.flow_from_directory(
        directory=validation_directory,
        target_size=(image_height, image_width),
        batch_size=64,
        class_mode='categorical',
        color_mode='rgb'
    )

    # compute the number of batches per epoch
    # the symbol // means to divide and floor the result
    steps_per_epoch = train_generator.n//train_generator.batch_size

    # train our model by feeding in the data generator 
    model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        batch_size=32,
        epochs=1,
        validation_data=validation_generator
    )
      
    # test how well our model performs against data
    # that it has not seen before
    # model.evaluate(x=x_test/255, y=y_test)
    model.evaluate(validation_generator)

    def load_and_preprocess_image(img_path, target_size=(image_height, image_width)):
        img = image.load_img(img_path, target_size=target_size)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0
        
        return img_array

    def test_model(model, test_directory, class_labels):
        test_images = [f for f in os.listdir(test_directory) if f.endswith('.jpg')]

        image_names = []
        actual_class = []
        predicted_class = []

        for image_name in test_images:
            img_path = os.path.join(test_directory, image_name)
            img_array = load_and_preprocess_image(img_path)

            predictions = model.predict(img_array)
            predicted_label = np.argmax(predictions)
            actual_label = image_name.split('_')[0]

            image_names.append(image_name)
            actual_class.append(actual_label)
            predicted_class.append(predicted_label)

        results_df = pd.DataFrame({'Image': image_names, 'Actual Class': actual_class, 'Predicted Class': predicted_class})
        results_df['Predicted Class'] = results_df['Predicted Class'].map(class_labels)

        return results_df

    def save_results_and_display_accuracy(results_df, save_path):
        results_df.to_csv(save_path, index=False)

        correct_predictions = np.sum(np.array(results_df['Actual Class']) == np.array(results_df['Predicted Class']))
        total_predictions = len(results_df)
        accuracy = correct_predictions / total_predictions

        print(f'Test Accuracy: {accuracy * 100}%')

    # Load the trained model
    model_path = 'Models/image_classification_model.h5'
    model.save(model_path)
    trained_model = models.load_model(model_path)

    class_labels = {0: 'apple', 1: 'banana', 2: 'mixed', 3: 'orange'}

    # Test the model
    results_df = test_model(trained_model, test_directory, class_labels)

    # Save results and display accuracy
    save_path = 'Results/result.csv'
    save_results_and_display_accuracy(results_df, save_path)

# running via "python mnist_sample.py"
if __name__ == '__main__':
  main()