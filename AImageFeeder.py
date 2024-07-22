import pyautogui
import cv2
import os
import time
import random
import time
import numpy as np
import tensorflow as tf

IMAGES_PATH = 'path\\to\\any\\folder\\to\\store\\images'
MODEL_PATH = 'path\\to\\model.h5'
model = tf.keras.models.load_model(MODEL_PATH)
print("please make sure screen resolution is 1920x1200 and selected video is on monitor 1")
x, y, width, height=420,180,1100,400
mask_rectangle = (0, 20, 176, 21)

class_names = {
    0: 'ad deep',
    1: 'ad mid',
    2: 'body deep',
    3: 'body mid',
    4: 'deuce deep',
    5: 'deuce mid',
}

def preprocess_image(file_name, mask_rectangle, top_crop_fraction=0.13, bottom_crop_fraction=0.38, left_crop_fraction=0.1, right_crop_fraction=0.1):
    img = cv2.imread(file_name)
    img = cv2.resize(img, (220, 80))
    x_start, y_start, width, height = mask_rectangle
    height, width, _ = img.shape
    top_crop = int(height * top_crop_fraction)
    bottom_crop = int(height * (1 - bottom_crop_fraction))
    left_crop = int(width * left_crop_fraction)
    right_crop = int(width * (1 - right_crop_fraction))
    img = img[top_crop:bottom_crop, left_crop:right_crop, :]
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img_mask = img
    img_mask[y_start:y_start+height, x_start:x_start+width] = 0 
    img = img/255.0
    return img, img_mask

while True:
    input('hit enter: ')
    randomint = random.randint(0,10000)
    screenshot = pyautogui.screenshot(region=(x, y, width, height))
    screenshot = np.array(screenshot)
    screenshot = screenshot[:, :, ::-1].copy()
    imgname = os.path.join(IMAGES_PATH, f'{randomint}.jpg')
    cv2.imwrite(imgname, screenshot)
    img = preprocess_image(imgname, mask_rectangle)[0]
    img = img.reshape(1, 39, 176, 1)
    predictions = model.predict(img)
    predicted_class_index = tf.argmax(predictions[0]).numpy()
    probabilities = np.argmax(predictions)
    predicted_probability = predictions[0, probabilities]
    predicted_percentage = predicted_probability * 100
    predicted_class_name = class_names[predicted_class_index]
    print(f'Predicted ball land: {predicted_class_name}, {predicted_percentage}% sure. image: {randomint}.jpg')
    time.sleep(0.1)