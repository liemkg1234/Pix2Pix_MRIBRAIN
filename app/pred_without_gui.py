from PIL import Image, ImageOps
import numpy as np
import os
from IPython.display import display
import argparse
import tensorflow as tf
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt

# from app.model import Generator, Discriminator, Precision, IoU, load_image_train
from model import Generator, Discriminator, Precision, IoU, load_image_train, load_classifi
import tensorflow as tf
import tensorflow.keras.backend as K

# Classification
def f1_metric(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2 * (precision * recall) / (precision + recall + K.epsilon())
    return f1_val
def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision
def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall
def specificity(y_true, y_pred):
    neg_y_true = 1 - y_true
    neg_y_pred = 1 - y_pred
    fp = K.sum(neg_y_true * y_pred)
    tn = K.sum(neg_y_true * neg_y_pred)
    specificity = tn / (tn + fp + K.epsilon())
    return specificity

# Load the classification model
model_Classifi = tf.keras.models.load_model ('/content/drive/MyDrive/Luan_Van/best/Effi7.h5', custom_objects={"precision": precision, "recall": recall, 'specificity': specificity, "f1_metric": f1_metric})

# Learning rate and other parameters
d_lr = 2e-4
g_lr = 2e-4
beta_1 = 0.5
LAMBDA = 200

# Load the Generator and Discriminator models
G = Generator()
generator_optimizer = tf.keras.optimizers.Adam(g_lr, beta_1=beta_1)
D = Discriminator()
discriminator_optimizer = tf.keras.optimizers.Adam(d_lr, beta_1=beta_1)

# Load weights
checkpoint_dir = '/content/drive/MyDrive/Luan_Van/best/'
checkpoint = tf.train.Checkpoint(
    generator_optimizer=generator_optimizer,
    discriminator_optimizer=discriminator_optimizer,
    generator=G,
    discriminator=D
)
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

def show_image(path_image):
    image = Image.open(path_image).convert('L')
    image = image.resize((256, 256))
    display(image)

def show_mask(path_mask):
    image = Image.open(path_mask)
    image = image.resize((256, 256))
    rgb_img = image.convert('RGB')
    
    # Convert the image to a red-yellow mask
    r, g, b = rgb_img.split()
    b = b.point(lambda i: i * 0.0)
    result = Image.merge('RGB', (r, g, b))
    display(result)


import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def predict_without_gui(path_image, path_mask, output_dir='output'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    image, mask = load_image_train(path_image, path_mask)
    
    # Classification
    image_classifi, mask_classifi = load_classifi(path_image, path_mask)
    image_classifi = image_classifi[tf.newaxis, ...]
    
    classifi = model_Classifi.predict(image_classifi)
    classifi = 1 if classifi >= 0.5 else 0
    
    # Predict using the Generator model
    G_out = G(image[tf.newaxis, ...], training=True)
    G_image = G_out[0, :, :, 0]

    # Calculate Precision and IoU
    pre = Precision(G_image, mask[:, :, 0])
    iou = IoU(G_image, mask[:, :, 0])
    print(f"Precision: {pre:.6f}, IoU: {iou:.6f}, Classification: {classifi}")

    # Convert generated image and mask for display
    G_image = (G_image + 1) * 127.5
    mask = (mask + 1) * 127.5

    G_image = Image.fromarray(G_image.numpy().astype(np.uint8))
    rgb_G_out = G_image.convert('RGB')
    r, g, b = rgb_G_out.split()
    r = r.point(lambda i: i * 0.0)
    g = g.point(lambda i: i * 0.0)
    result = Image.merge('RGB', (r, g, b))

    mask = Image.fromarray(mask[:, :, 0].numpy().astype(np.uint8))
    mask_rgb = mask.convert('RGB')
    r, g, b = mask_rgb.split()
    b = b.point(lambda i: i * 0.0)
    result_mask = Image.merge('RGB', (r, g, b))

    # Merge the generated image and mask
    result_arr = np.array(result)
    mask_arr = np.array(result_mask)
    merge_arr = np.add(result_arr, mask_arr)
    merge_img = Image.fromarray(merge_arr)
    display(result)
    display(merge_img)
    # Save the images
    grayscale_result = ImageOps.grayscale(result)
    binary_result = grayscale_result.point(lambda p: 255 if p > 0 else 0)
    binary_result.save(os.path.join(output_dir, 'generated_image.png'))

def main(image_path, mask_path):
    if not os.path.exists(image_path):
        print(f"Image file not found: {image_path}")
        return
    if not os.path.exists(mask_path):
        print(f"Mask file not found: {mask_path}")
        return

    print(f"Predicting for Image: {image_path} and Mask: {mask_path}")
    show_image(image_path)
    show_mask(mask_path)
    predict_without_gui(image_path, mask_path)
    print("Prediction completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict using image and mask")
    parser.add_argument('--image', type=str, required=True, help='Path to the input image')
    parser.add_argument('--mask', type=str, required=True, help='Path to the input mask')

    args = parser.parse_args()

    main(args.image, args.mask)
