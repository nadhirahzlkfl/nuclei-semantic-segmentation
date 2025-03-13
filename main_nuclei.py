#%% import packages
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os, cv2, keras,datetime
from tensorflow_examples.models.pix2pix import pix2pix
from sklearn.model_selection import train_test_split
from IPython.display import clear_output
from keras import layers,losses,callbacks,applications

filepath=os.path.join(os.getcwd(),'dataset')
images=[]
masks=[]

# load images
image_path=os.path.join(filepath,'inputs')
for img in os.listdir(image_path):
    #Get the full path of the image file
    full_path=os.path.join(image_path,img)
    #Read the image file based on the full path
    img_np=cv2.imread(full_path)
    #Convert the image from bgr to rgb
    img_np=cv2.cvtColor(img_np,cv2.COLOR_BGR2RGB)
    #Resize the image into 128x128
    img_np=cv2.resize(img_np,(128,128))
    #Place the image into the empty list
    images.append(img_np)

# load masks
mask_path=os.path.join(filepath,'masks')
for mask in os.listdir(mask_path):
    #Get the full path of the mask file
    full_path=os.path.join(mask_path,mask)
    #Read the mask file as a grayscale image
    mask_np=cv2.imread(full_path,cv2.IMREAD_GRAYSCALE)
    #Resize the image into 128x128
    mask_np=cv2.resize(mask_np,(128,128))
    #Place the mask into the empty list
    masks.append(mask_np)

# convert the list of np array into a full np array
images_np=np.array(images)
masks_np=np.array(masks)

# data preprocessing
# expand the mask dimension to include the channel axis
masks_np_exp=np.expand_dims(masks_np,axis=-1)
# convert the mask value into just 0 and 1
converted_masks_np=np.round(masks_np_exp/255)
# normalize the images pixel value
normalized_images_np=images_np/255.0

# perform train test split
from sklearn.model_selection import train_test_split
SEED=947
X_train,X_test,y_train,y_test = train_test_split(normalized_images_np,converted_masks_np,shuffle=True,random_state=SEED)

# convert the numpy array into tensorflow tensors
X_train_tensor=tf.data.Dataset.from_tensor_slices(X_train)
X_test_tensor=tf.data.Dataset.from_tensor_slices(X_test)
y_train_tensor=tf.data.Dataset.from_tensor_slices(y_train)
y_test_tensor=tf.data.Dataset.from_tensor_slices(y_test)

# combine features and labels together to form a zip dataset
train=tf.data.Dataset.zip((X_train_tensor,y_train_tensor))
test=tf.data.Dataset.zip((X_test_tensor,y_test_tensor))

# custom class for augmentation
class Augment(tf.keras.layers.Layer):
  def __init__(self, seed=42):
    super().__init__()
    self.augment_inputs=tf.keras.layers.RandomFlip(mode="horizontal", seed=seed)
    self.augment_labels=tf.keras.layers.RandomFlip(mode="horizontal", seed=seed)

  def call(self,inputs,labels):
    inputs=self.augment_inputs(inputs)
    labels=self.augment_labels(labels)
    return inputs,labels

#%% prepare dataset
TRAIN_LENGTH=len(train)
BATCH_SIZE=64
BUFFER_SIZE=1000
STEPS_PER_EPOCH=TRAIN_LENGTH//BATCH_SIZE
train_batches=(
    train
    .cache()
    .shuffle(BUFFER_SIZE)
    .batch(BATCH_SIZE)
    .repeat()
    .map(Augment())
    .prefetch(buffer_size=tf.data.AUTOTUNE))
test_batches=test.batch(BATCH_SIZE)

#%% function to vissualize image, label and prediction
def display(display_list):
  plt.figure(figsize=(15,15))

  title=['Input Image','True Mask','Predicted Mask']

  for i in range(len(display_list)):
    plt.subplot(1,len(display_list),i+1)
    plt.title(title[i])
    plt.imshow(tf.keras.utils.array_to_img(display_list[i]))
    plt.axis('off')
  plt.show()

# test out the function to display an example
for images,masks in train_batches.take(2):
  sample_image,sample_mask=images[0],masks[0]
  display([sample_image,sample_mask])
  print(images.shape)
  print(masks.shape)

#%% model development
# get the feature extractor using keras.applications
base_model=applications.MobileNetV2(input_shape=sample_image.shape,include_top=False,alpha=0.35)
base_model.summary()

# properly create a downwardpath
base_model=tf.keras.applications.MobileNetV2(input_shape=[128,128,3],include_top=False)

# use the activations of these layers
layer_names=[
    'block_1_expand_relu',   # 64x64
    'block_3_expand_relu',   # 32x32
    'block_6_expand_relu',   # 16x16
    'block_13_expand_relu',  # 8x8
    'block_16_project',      # 4x4
]
base_model_outputs=[base_model.get_layer(name).output for name in layer_names]

# create the feature extraction model
down_stack=tf.keras.Model(inputs=base_model.input,outputs=base_model_outputs)

down_stack.trainable=False
down_stack.summary()

# uppath
up_stack=[
    pix2pix.upsample(512,3),  # 4x4 -> 8x8
    pix2pix.upsample(256,3),  # 8x8 -> 16x16
    pix2pix.upsample(128,3),  # 16x16 -> 32x32
    pix2pix.upsample(64,3),   # 32x32 -> 64x64
]
# up_stack.summary

# create unet model
def unet_model(output_channels:int):
  inputs=tf.keras.layers.Input(shape=[128, 128, 3])

  # downsampling through the model
  skips=down_stack(inputs)
  x=skips[-1]
  skips=reversed(skips[:-1])

  # upsampling and establishing the skip connections
  for up, skip in zip(up_stack,skips):
    x=up(x)
    # concatenate the output
    concat=layers.Concatenate()
    x=concat([x,skip])

  # this is the last layer of the model
  last=layers.Conv2DTranspose(
      filters=output_channels,kernel_size=3,strides=2,
      padding='same')  #64x64 -> 128x128

  x=last(x)

  model=keras.Model(inputs=inputs,outputs=x)
  return model

# use the function to create the Unet
model=unet_model(output_channels=3)
model.summary()
keras.utils.plot_model(model,to_file='model.png')

# compile the model
loss=losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer='adam',loss=loss,metrics=['accuracy'])

# create a custom callback to display result in the middle of the training
def create_mask(pred_mask):
  pred_mask=tf.math.argmax(pred_mask,axis=-1)
  pred_mask=pred_mask[..., tf.newaxis]
  return pred_mask[0]

def show_predictions(dataset=None,num=1):
  if dataset:
    for image,mask in dataset.take(num):
      pred_mask=model.predict(image)
      display([image[0],mask[0],create_mask(pred_mask)])
  else:
    display([sample_image, sample_mask,
             create_mask(model.predict(sample_image[tf.newaxis, ...]))])
    
class DisplayCallback(callbacks.Callback):
  def on_epoch_end(self, epoch, logs=None):
    clear_output(wait=True)
    show_predictions()

# prepare for the training
logpath=os.path.join('seg_log',datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
es=callbacks.EarlyStopping(patience=10,verbose=1)
tb=callbacks.TensorBoard(log_dir=logpath)

# model training
EPOCHS=50
VAL_SUBSPLITS=5
VALIDATION_STEPS=len(test)//BATCH_SIZE//VAL_SUBSPLITS
model_history=model.fit(
    x=train_batches,
    epochs=EPOCHS,
    steps_per_epoch=STEPS_PER_EPOCH,
    validation_data=(test_batches),
    validation_steps=VALIDATION_STEPS,
    callbacks=[tb,es,DisplayCallback()]
)

# evaluate the model
print(model.evaluate(test_batches))

#%% use the model to make prediction
for test_img,test_label in test_batches.take(2):
    predictions=model.predict(test_img)
    predictions=np.argmax(predictions,axis=-1)
    predictions=np.expand_dims(predictions,axis=-1)

# take a data out to plot
display([test_img[10],test_label[10],predictions[10]])

#%% save model
saved_model=os.path.join(os.getcwd(),'saved_model')
model.save(os.path.join(saved_model,'model.keras'))