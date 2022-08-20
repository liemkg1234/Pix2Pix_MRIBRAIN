import tensorflow as tf
import tensorflow.keras.backend as K


def downsample(filters, size, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                               kernel_initializer=initializer, use_bias=False))

    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())

    result.add(tf.keras.layers.LeakyReLU())

    return result

def upsample(filters, size, apply_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                        padding='same',
                                        kernel_initializer=initializer,
                                        use_bias=False))

    result.add(tf.keras.layers.BatchNormalization())

    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))

    result.add(tf.keras.layers.ReLU())

    return result

# define the standalone generator model
def Generator():
    inputs = tf.keras.layers.Input(shape=[256, 256, 1])

    down_stack = [
        downsample(64, 4, apply_batchnorm=False),  # (batch_size, 128, 128, 64)
        downsample(128, 4),  # (batch_size, 64, 64, 128)
        downsample(256, 4),  # (batch_size, 32, 32, 256)
        downsample(512, 4),  # (batch_size, 16, 16, 512)
        downsample(512, 4),  # (batch_size, 8, 8, 512)
        downsample(512, 4),  # (batch_size, 4, 4, 512)
        downsample(512, 4),  # (batch_size, 2, 2, 512)
        downsample(512, 4),  # (batch_size, 1, 1, 512)
    ]

    up_stack = [
        upsample(512, 4, apply_dropout=True),  # (batch_size, 2, 2, 1024)
        upsample(512, 4, apply_dropout=True),  # (batch_size, 4, 4, 1024)
        upsample(512, 4, apply_dropout=True),  # (batch_size, 8, 8, 1024)
        upsample(512, 4),  # (batch_size, 16, 16, 1024)
        upsample(256, 4),  # (batch_size, 32, 32, 512)
        upsample(128, 4),  # (batch_size, 64, 64, 256)
        upsample(64, 4),  # (batch_size, 128, 128, 128)
    ]

    initializer = tf.random_normal_initializer(0., 0.02)
    last = tf.keras.layers.Conv2DTranspose(1, 4,
                                           strides=2,
                                           padding='same',
                                           kernel_initializer=initializer,
                                           activation='tanh')  # (batch_size, 256, 256, 1)

    x = inputs

    # Downsampling through the model
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)

# define the discriminator model
def Discriminator():
  initializer = tf.random_normal_initializer(0., 0.02)

  inp = tf.keras.layers.Input(shape=[256, 256, 1], name='input_image')
  tar = tf.keras.layers.Input(shape=[256, 256, 1], name='target_image')

  x = tf.keras.layers.concatenate([inp, tar])  # (batch_size, 256, 256, channels*2)

  down1 = downsample(64, 4, False)(x)  # (batch_size, 128, 128, 64)
  down2 = downsample(128, 4)(down1)  # (batch_size, 64, 64, 128)
  down3 = downsample(256, 4)(down2)  # (batch_size, 32, 32, 256)

  zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)  # (batch_size, 34, 34, 256)
  conv = tf.keras.layers.Conv2D(512, 4, strides=1,
                                kernel_initializer=initializer,
                                use_bias=False)(zero_pad1)  # (batch_size, 31, 31, 512)

  batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

  leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

  zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)  # (batch_size, 33, 33, 512)

  last = tf.keras.layers.Conv2D(1, 4, strides=1,
                                kernel_initializer=initializer)(zero_pad2)  # (batch_size, 30, 30, 1)

  return tf.keras.Model(inputs=[inp, tar], outputs=last)


# Precision = TruePositive / (TruePositive + FalsePositive)
def Precision(y_true, y_pred): #K phat. khi doan sai 0->1
  y_true = (y_true+1)/2   #[-1,1] to [0,1]
  y_pred = (y_pred+1)/2   #[-1,1] to [0,1]
  y_true = tf.cast((y_true >= 0.5), tf.float32) #> 0.5 = 1, <0.5 = 0
  y_pred = tf.cast((y_pred >= 0.5), tf.float32) #> 0.5 = 1, <0.5 = 0
  # print(y_true)
  # print(y_pred)
  #TruePositive
  TP = y_true*y_pred
  pr = K.sum(TP) / K.sum(y_true)
  if tf.math.is_nan(pr):
     pr = tf.constant(0.0)
  return pr

# iou = true_positives / (true_positives + false_positives + false_negatives)
def IoU(y_true, y_pred):
  y_true = (y_true+1)/2   #[-1,1] to [0,1]
  y_pred = (y_pred+1)/2   #[-1,1] to [0,1]
  y_true = tf.cast((y_true >= 0.5), tf.float32) #> 0.5 = 1, <0.5 = 0
  y_pred = tf.cast((y_pred >= 0.5), tf.float32) #> 0.5 = 1, <0.5 = 0
  TP = y_true*y_pred
  FN = tf.nn.relu(y_pred - y_true) # <0 => = 0
  iou = K.sum(TP) / (K.sum(y_true) + K.sum(FN))
  if tf.math.is_nan(iou):
    iou = tf.constant(0.0)
  return iou

def load_imagemask(path_image, path_mask):
  image = tf.io.read_file(path_image)
  image = tf.io.decode_jpeg(image)
  image = tf.image.resize(image, (256, 256)) #Resize (512->256)
  image = image[:,:,0]
  image = image[..., tf.newaxis]


  mask = tf.io.read_file(path_mask)
  mask = tf.io.decode_jpeg(mask)
  mask = tf.image.resize(mask, (256, 256))
  mask = mask[:,:,0]
  mask = mask[..., tf.newaxis]


  image = tf.cast(image, tf.float32)
  mask = tf.cast(mask, tf.float32)


  return image, mask

#Standardize/Normalize  the images to [-1, 1]
def standardize(image, mask):
  image = (image - 127.5) / 127.5
  mask = (mask - 127.5) / 127.5
  return image, mask

def load_image_train(path_image, path_mask):
  image, mask = load_imagemask(path_image, path_mask)
  image, mask = standardize(image, mask)
  return image, mask


def load_classifi(path_image, path_mask):
    image = tf.io.read_file(path_image)
    image = tf.io.decode_jpeg(image)
    image = tf.image.resize(image, (224, 224))  # Resize (512->256)
    # image = image[:,:,0]
    # image = image[..., tf.newaxis]

    # mask = tf.io.read_file(path_mask)
    # mask = tf.io.decode_jpeg(mask)
    # mask = tf.image.resize(mask, (256, 256))
    # mask = mask[:,:,0]
    # mask = mask[..., tf.newaxis]

    image = tf.cast(image, tf.float32)
    # mask = tf.cast(mask, tf.float32)

    image = image / 255
    # mask = mask/255

    return image, path_mask