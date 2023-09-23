# DL_Weekly_Assignment_06

This Week's Lab Assignment
- Repeat the experiment but this time without Data Augmentation
- Observe how the accuracy changes
- Add Data Augmentation but this time add only Image Flipping (without
rotation or zooming)
- Observe how the accuracy changes
- Write all your observations in a report. You may only submit this report to
Moodle.


_URL = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'

zip_dir = tf.keras.utils.get_file('cats_and_dogs_filterted.zip', origin=_URL, extract=True)
