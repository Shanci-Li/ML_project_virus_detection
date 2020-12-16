
import tensorflow as tf
import segmentation_models as sm
import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import imageio
import sys

#PATH to your datafolder
PATH ='../data/image_and_masks/'
#LABELS of your segmentation task
LABELS=['background','virus']

def predict_unlabeled(model,PATH,show_predictions=True): 
  test_datagen= ImageDataGenerator(rescale=1./255)
  unlabeled_image_generator = test_datagen.flow_from_directory(PATH+"test_imgs",class_mode=None,batch_size = 1, shuffle =False)

  if show_predictions:
    for i in range(len(unlabeled_image_generator)):
      sample_image=next(unlabeled_image_generator)
      sample_mask = model.predict(sample_image)
      sample_mask=np.expand_dims(sample_mask[0].argmax(axis=-1),axis=-1)
      display([sample_image[0],sample_mask],title=[unlabeled_image_generator.filenames[i][5:],"masks"])
      imageio.imwrite(PATH+"pred/"+ unlabeled_image_generator.filenames[i][5:],sample_mask)

def display(display_list,title):
  plt.figure(figsize=(15, 15))
  for i in range(len(display_list)):
    plt.subplot(1, len(display_list), i+1)
    plt.title(title[i])
    plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]),cmap='magma')
    plt.axis('off')
  plt.show()

model = sm.Unet('mobilenet', encoder_weights='imagenet',classes=2, activation='softmax', encoder_freeze=False)
model.load_weights("./final_model/cp-0030.ckpt")

predict_unlabeled(model,PATH,show_predictions=True)
