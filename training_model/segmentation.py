import segmentation_models as sm #1.0.1
import albumentations as A  #0.1.12
import tensorflow as tf #2.3.1
import matplotlib.pyplot as plt
import numpy as np
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

sm.set_framework('tf.keras')

class segmentation:
  def __init__(self,PATH,LABELS,target_size=(512,512), params=None):
    """
    parameters
    ----------
    PATH : string path to directory containing folders: 'images' and 'masks' optionally it will include 'val-images' and 'val-masks' for validation
    LABELS : list Names of labels
    target_size : tuple of ints (WxH) Images will be resized to this. Each dimension should be divisible by 32 for best results.
    """
    self.PATH=PATH
    self.classes=LABELS
    self.num_classes = len(LABELS)
    self.activation = 'sigmoid' if self.num_classes == 1 else 'softmax'
    self.metrics=['accuracy',sm.metrics.IOUScore(threshold=0.5)]
    self.target_size = target_size
    
    # Check if validation directory exists
    self.have_validation = False
    if os.path.isdir(os.path.join(PATH, 'val_imgs')) and os.path.isdir(os.path.join(PATH, 'val_masks')):
      self.have_validation = True


    # Parameters for Unet neural network
    self.backbone=params['backbone']
    self.loss =params['loss']
    self.weights =params['weights']
    self.augmentation =params['augmentation']
    self.weights_pretrained =params['weights_pretrained']
    self.batch_size =params['batch_size']
    self.steps_per_epoch =params['steps_per_epoch']
    self.n_epochs =params['n_epochs']
    self.encoder_freeze =params['encoder_freeze']

    # Data
    self.create_datagenerator(PATH)
    # Model
    self.model = sm.Unet(self.backbone, encoder_weights=self.weights_pretrained,classes=self.num_classes, activation=self.activation, encoder_freeze=self.encoder_freeze)   

    try:
      #Now Train
      self.train()
    except KeyboardInterrupt:
      # allow user to press crtl-C to end execution without losing model
      pass

  def create_augmentation(self):
    augmentation = A.Compose(
    [
        A.HorizontalFlip(p = 0.5), # apply horizontal flip to 50% of images
        A.OneOf([A.RandomContrast(), A.RandomGamma(), A.RandomBrightness()], p = 0.5 ), # apply one of transforms to 50% of images
        A.OneOf( [A.ElasticTransform( alpha = 120, sigma = 120 * 0.05,alpha_affine = 120 * 0.03),A.GridDistortion() ],p = 0.5)  # apply one of transforms to 50% images
    ],
    p = 0.9 # 10% of cases keep same image 
    )   
    return augmentation


  def create_datagenerator(self,PATH,):
      options={'horizontal_flip': True,'vertical_flip': True}
      image_datagen = ImageDataGenerator(rescale=1./255,**options)
      mask_datagen = ImageDataGenerator(**options)
      val_datagen = ImageDataGenerator(rescale=1./255)
      val_datagen_mask = ImageDataGenerator(rescale=1)
      #Create custom zip and custom batch_size

      def combine_image_mask(gen1, gen2,batch_size=6,training=True):
          while True:
              image_batch, label_batch=next(gen1)[0], np.expand_dims(next(gen2)[0][:,:,0],axis=-1)
              image_batch, label_batch=np.expand_dims(image_batch,axis=0),np.expand_dims(label_batch,axis=0)

              for i in range(batch_size-1):
                image_i,label_i = next(gen1)[0], np.expand_dims(next(gen2)[0][:,:,0],axis=-1)
                
                if (self.augmentation == True) and training==True :
                  aug=self.create_augmentation()
                  augmented = aug(image = image_i, mask = label_i)
                  image_i,label_i=augmented['image'],augmented['mask']

                image_i, label_i=np.expand_dims(image_i,axis=0),np.expand_dims(label_i,axis=0)
                image_batch=np.concatenate([image_batch,image_i],axis=0)
                label_batch=np.concatenate([label_batch,label_i],axis=0)
                
              yield((image_batch,label_batch))

      seed = np.random.randint(0,1e5)

      train_image_generator = image_datagen.flow_from_directory(PATH+'train_imgs',seed=seed, target_size=self.target_size,class_mode=None,batch_size = self.batch_size)
      train_mask_generator = mask_datagen.flow_from_directory(PATH+'train_masks',seed=seed, target_size=self.target_size,class_mode=None,batch_size = self.batch_size)
      self.train_generator = combine_image_mask(train_image_generator, train_mask_generator,training=True)
      
      if self.have_validation:
        val_image_generator = val_datagen.flow_from_directory(PATH+'val_imgs',seed=seed, target_size=self.target_size,class_mode=None,batch_size = self.batch_size)
        val_mask_generator = val_datagen_mask.flow_from_directory(PATH+'val_masks',seed=seed, target_size=self.target_size,class_mode=None,batch_size = self.batch_size)
        self.val_generator = combine_image_mask(val_image_generator, val_mask_generator,training=False)


    
  
  def train(self):
    
    print("\n Starting Training \n")
  
    self.model.compile('adam', self.loss, self.metrics, loss_weights=self.weights)
    if self.have_validation:
      checkpoint_path = "training_1/cp.ckpt"
      cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, monitor='val_iou_score', verbose=1, save_weights_only=True, save_best_only=True, mode='max')
      self.model_history=self.model.fit(self.train_generator, epochs=self.n_epochs, 
                          steps_per_epoch = self.steps_per_epoch,
                          validation_data=self.val_generator, 
                          validation_steps=1,
                          callbacks=[cp_callback])
      
    else:
      self.model_history=self.model.fit(self.train_generator, epochs=self.n_epochs, 
                          steps_per_epoch = self.steps_per_epoch,
                          )

    print("\n Finished Training \n")
  
  def save_model(location):
    self.model.save_weights(location)

  def plot_history(self):
      
    fig,ax=plt.subplots(1,2,figsize=(30,10))
    epochs = range(self.n_epochs)

    loss = self.model_history.history['loss']
    val_loss = self.model_history.history['val_loss']
    
    ax[0].plot(epochs, loss, 'r', label='Training loss')
    ax[0].plot(epochs, val_loss, 'bo', label='Validation loss')
    ax[0].set_title('Training and Validation Loss')
    ax[0].set_ylabel('Loss Value')

    # accuracy = self.model_history.history['accuracy']
    # val_accuracy = self.model_history.history['val_accuracy']
    # ax[1].plot(epochs, accuracy, 'r', label='Training Iou Score')
    # ax[1].plot(epochs, val_accuracy, 'bo', label='Validation Iou Score')
    # ax[1].set_title('Training and Validation Accuracy')
    # ax[1].set_ylabel('Accuracy Score')
    # ax[1].set_ylim(0,1)
    
    iou_score = self.model_history.history['iou_score']
    val_iou_score = self.model_history.history['val_iou_score']
    ax[1].plot(epochs, iou_score, 'r', label='Training Iou Score')
    ax[1].plot(epochs, val_iou_score, 'bo', label='Validation Iou Score')
    ax[1].set_title('Training and Validation IOU Score')
    ax[1].set_ylabel('IOU Score')    
    ax[1].set_ylim(0,1)
    
    for i in range(2):
        ax[i].set_xlabel('Epoch')
        ax[i].legend();
    plt.show()

  def show_predictions(self,generator=None, num=10):
    if generator ==None:
      generator = self.train_generator
    for i in range(num):
      image, mask=next(generator)
      sample_image, sample_mask= image[1], mask[1]
      image = np.expand_dims(sample_image, axis=0)

      pr_mask = self.model.predict(image)
      pr_mask=np.expand_dims(pr_mask[0].argmax(axis=-1),axis=-1)

      display([sample_image, sample_mask,pr_mask])


  def predict_unlabeled(self,PATH,show_predictions=True):
    test_datagen= ImageDataGenerator(rescale=1./255)
    unlabeled_image_generator = test_datagen.flow_from_directory(PATH+"test_imgs",class_mode=None,batch_size = 1)

    if show_predictions:
      for _ in range(len(unlabeled_image_generator)):
        test_image=next(unlabeled_image_generator)
        pred_mask = self.model.predict(test_image)
        pred_mask=np.expand_dims(pred_mask[0].argmax(axis=-1),axis=-1)
        display([test_image[0],pred_mask],title=['Input Image','Predicted Mask'])

    return self.model.predict(unlabeled_image_generator,steps=len(os.listdir(PATH+"test_imgs"+'/test')))

# display the image and mask
def display(display_image,title=['Input Image', 'True Mask', 'Predicted Mask']):
  plt.figure(figsize=(15, 15))
  for i in range(len(display_image)):
    plt.subplot(1, len(display_image), i+1)
    plt.title(title[i])
    plt.imshow(tf.keras.preprocessing.image.array_to_img(display_image[i]),cmap='magma')
    plt.axis('off')
  plt.show()
  
