import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

"""
Install Cuda
nvcc --version
sudo apt install nvidia-cuda-toolkit -y
pip install tensorrt
#sudo apt-get -y install cuda
conda install tensorflow-gpu
conda install tensorflow-estimator==2.1.0 or any latest version
https://stackoverflow.com/questions/75614728/cuda-12-tf-nightly-2-12-could-not-find-cuda-drivers-on-your-machine-gpu-will
"""


# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
zip_dir = '.'


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


class CatsAndDogsDataSet:
    def __init__(self):
        self.BATCH_SIZE = 100
        self.IMG_SHAPE = 150

        self.base_dir = None
        self.train_dir = None
        self.train_dogs_dir = None
        self.train_cats_dir = None
        self.train_data_gen = None
        self.total_train = None

        self.validation_dir = None
        self.validation_dogs_dir = None
        self.validation_cats_dir = None
        self.validation_data_gen = None
        self.total_val = None

        self.assign_variables()
        self.train_data_generator()
        self.validation_data_generator()
        # self.understanding_data()

    def assign_variables(self):
        self.base_dir = os.path.join(os.path.dirname(zip_dir), 'cats_and_dogs_filtered')
        self.train_dir = os.path.join(self.base_dir, 'train')
        self.validation_dir = os.path.join(self.base_dir, 'validation')
        self.train_cats_dir = os.path.join(self.train_dir, 'cats')  # directory with our training cat pictures
        self.train_dogs_dir = os.path.join(self.train_dir, 'dogs')  # directory with our training dog pictures
        self.validation_cats_dir = os.path.join(self.validation_dir,
                                                'cats')  # directory with our validation cat pictures
        self.validation_dogs_dir = os.path.join(self.validation_dir,
                                                'dogs')  # directory with our validation dog pictures

    def understanding_data(self):
        num_cats_tr = len(os.listdir(self.train_cats_dir))
        num_dogs_tr = len(os.listdir(self.train_dogs_dir))

        num_cats_val = len(os.listdir(self.validation_cats_dir))
        num_dogs_val = len(os.listdir(self.validation_dogs_dir))

        self.total_train = num_cats_tr + num_dogs_tr
        self.total_val = num_cats_val + num_dogs_val

        print("=" * 40)
        print('total training cat images:', num_cats_tr)
        print('total training dog images:', num_dogs_tr)

        print('total validation cat images:', num_cats_val)
        print('total validation dog images:', num_dogs_val)
        print("-" * 40)
        print("Total training images:", self.total_train)
        print("Total validation images:", self.total_val)
        print("=" * 40)

        self.data_augmentation()
        augmented_images = [self.train_data_gen[0][0][0] for i in range(5)]
        self.plot_images(augmented_images)

        self.image_rotating()
        augmented_images = [self.train_data_gen[0][0][0] for i in range(5)]
        self.plot_images(augmented_images)

        self.image_zoom()
        augmented_images = [self.train_data_gen[0][0][0] for i in range(5)]
        self.plot_images(augmented_images)

        self.images_all_together()
        augmented_images = [self.train_data_gen[0][0][0] for i in range(5)]
        self.plot_images(augmented_images)

    # This function will plot images in the form of a grid with 1 row and 5 columns
    # where images are placed in each column.
    def plot_images(self, images_arr):
        fig, axes = plt.subplots(1, 5, figsize=(20, 20))
        axes = axes.flatten()
        for img, ax in zip(images_arr, axes):
            ax.imshow(img)
        plt.tight_layout()
        plt.show()

    def data_augmentation(self):
        image_gen = ImageDataGenerator(rescale=1. / 255, horizontal_flip=True)
        self.train_data_gen = image_gen.flow_from_directory(batch_size=self.BATCH_SIZE,
                                                            directory=self.train_dir,
                                                            shuffle=True,
                                                            target_size=(self.IMG_SHAPE, self.IMG_SHAPE),
                                                            class_mode='binary')
        self.total_train = self.train_data_gen.samples
        # augmented_images = [self.train_data_gen[0][0][0] for i in range(5)]
        # self.plot_images(augmented_images)

    def image_rotating(self):
        image_gen = ImageDataGenerator(rescale=1. / 255, rotation_range=45)

        self.train_data_gen = image_gen.flow_from_directory(batch_size=self.BATCH_SIZE,
                                                            directory=self.train_dir,
                                                            shuffle=True,
                                                            target_size=(self.IMG_SHAPE, self.IMG_SHAPE),
                                                            class_mode='binary')
        self.total_train = self.train_data_gen.samples
        # augmented_images = [self.train_data_gen[0][0][0] for i in range(5)]
        # self.plot_images(augmented_images)

    def image_zoom(self):
        image_gen = ImageDataGenerator(rescale=1. / 255, zoom_range=0.5)

        self.train_data_gen = image_gen.flow_from_directory(batch_size=self.BATCH_SIZE,
                                                            directory=self.train_dir,
                                                            shuffle=True,
                                                            target_size=(self.IMG_SHAPE, self.IMG_SHAPE),
                                                            class_mode='binary')

        self.total_train = self.train_data_gen.samples
        # augmented_images = [self.train_data_gen[0][0][0] for i in range(5)]
        # self.plot_images(augmented_images)

    def images_all_together(self):
        image_gen_train = ImageDataGenerator(
            rescale=1. / 255,
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest')

        self.train_data_gen = image_gen_train.flow_from_directory(batch_size=self.BATCH_SIZE,
                                                                  directory=self.train_dir,
                                                                  shuffle=True,
                                                                  target_size=(self.IMG_SHAPE, self.IMG_SHAPE),
                                                                  class_mode='binary')
        self.total_train = self.train_data_gen.samples
        # augmented_images = [self.train_data_gen[0][0][0] for i in range(5)]
        # self.plot_images(augmented_images)

    def train_data_generator(self):
        image_gen = ImageDataGenerator(rescale=1. / 255)
        self.train_data_gen = image_gen.flow_from_directory(batch_size=self.BATCH_SIZE,
                                                            directory=self.train_dir,
                                                            target_size=(self.IMG_SHAPE, self.IMG_SHAPE),
                                                            class_mode='binary')
        self.total_train = self.train_data_gen.samples

    def validation_data_generator(self):
        image_gen_val = ImageDataGenerator(rescale=1. / 255)

        self.validation_data_gen = image_gen_val.flow_from_directory(batch_size=self.BATCH_SIZE,
                                                                     directory=self.validation_dir,
                                                                     target_size=(self.IMG_SHAPE, self.IMG_SHAPE),
                                                                     class_mode='binary')
        self.total_val = self.validation_data_gen.samples


class Model:
    def __init__(self, augmentation=None):
        self.model = None
        self.epochs = 60
        self.history = None

        data_set = CatsAndDogsDataSet()
        self.BATCH_SIZE = data_set.BATCH_SIZE

        if augmentation is None:
            self.train_data_gen = data_set.train_data_gen
            self.total_train = data_set.train_data_gen.samples
            print("Used default dataset.")
        else:
            data_set.data_augmentation()
            self.train_data_gen = data_set.train_data_gen
            self.total_train = data_set.train_data_gen.samples
            print("Used augmented dataset.")

        self.validation_data_gen = data_set.validation_data_gen
        self.total_val = data_set.total_val

        self.define_the_model()
        self.compiling_the_model()
        self.train_the_model()
        self.visualizing_results_of_the_training()

    def define_the_model(self):
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
            tf.keras.layers.MaxPooling2D(2, 2),

            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2, 2),

            tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2, 2),

            tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2, 2),

            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(2)
        ])

    def compiling_the_model(self):
        self.model.compile(optimizer='adam',
                           loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                           metrics=['accuracy'])

    def model_summary(self):
        self.model.summary()

    def train_the_model(self):
        self.history = self.model.fit_generator(
            self.train_data_gen,
            steps_per_epoch=int(np.ceil(self.total_train / float(self.BATCH_SIZE))),
            epochs=self.epochs,
            validation_data=self.validation_data_gen,
            validation_steps=int(np.ceil(self.total_val / float(self.BATCH_SIZE)))
        )

    def visualizing_results_of_the_training(self):
        acc = self.history.history['accuracy']
        val_acc = self.history.history['val_accuracy']

        loss = self.history.history['loss']
        val_loss = self.history.history['val_loss']

        epochs_range = range(self.epochs)

        plt.figure(figsize=(8, 8))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, acc, label='Training Accuracy')
        plt.plot(epochs_range, val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')

        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, loss, label='Training Loss')
        plt.plot(epochs_range, val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')
        plt.show()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
    ds = CatsAndDogsDataSet()
    # ds.understanding_data()

    model_without_data_augmentation = Model()
    model_with_data_augmentation = Model('Flip')

    model_without_data_augmentation.validation_data_gen()
    model_with_data_augmentation.validation_data_gen()
