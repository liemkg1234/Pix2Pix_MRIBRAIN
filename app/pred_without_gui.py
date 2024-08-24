from PIL import Image
import numpy as np
import os
import tensorflow as tf
import tensorflow.keras.backend as K

# Custom metrics
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
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
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
model_Classifi = tf.keras.models.load_model(
    'app/best/Effi7.h5',
    custom_objects={"precision": precision, "recall": recall, 'specificity': specificity, "f1_metric": f1_metric}
)

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
checkpoint_dir = 'app/best/'
checkpoint = tf.train.Checkpoint(
    generator_optimizer=generator_optimizer,
    discriminator_optimizer=discriminator_optimizer,
    generator=G,
    discriminator=D
)
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

# Define the predict function without GUI
def predict_without_gui(image_path, mask_path):
    # Load image and mask
    image, mask = load_image_train(image_path, mask_path)
    
    # Classification
    image_classifi, mask_classifi = load_classifi(image_path, mask_path)
    image_classifi = image_classifi[tf.newaxis, ...]
    classifi = model_Classifi.predict(image_classifi)
    classifi = 1 if classifi >= 0.5 else 0

    # Generate image using the Generator model
    G_out = G(image[tf.newaxis, ...], training=True)
    G_image = G_out[0, :, :, 0]

    # Calculate Precision and IoU
    pre = Precision(G_image, mask[:, :, 0])
    iou = IoU(G_image, mask[:, :, 0])
    print(f"Precision: {pre:.6f}, IoU: {iou:.6f}, Class: {classifi}")

    # Process images for output
    G_image = (G_image + 1) * 127.5
    mask = (mask + 1) * 127.5

    generated_image = Image.fromarray(G_image.numpy())

    # Convert mask to yellow
    mask_image = Image.fromarray(mask[:, :, 0].numpy())
    mask_rgb = mask_image.convert('RGB')
    r, g, b = mask_rgb.split()
    b = b.point(lambda i: i * 0.0)
    yellow_mask = Image.merge('RGB', (r, g, b))

    # Convert generated image to blue
    generated_rgb = generated_image.convert('RGB')
    r, g, b = generated_rgb.split()
    r = r.point(lambda i: i * 0.0)
    g = g.point(lambda i: i * 0.0)
    blue_generated = Image.merge('RGB', (r, g, b))

    # Merge the mask and generated images
    result_arr = np.array(blue_generated)
    mask_arr = np.array(yellow_mask)
    merge_arr = np.add(result_arr, mask_arr)
    merge_img = Image.fromarray(merge_arr)

    # Save or display results
    generated_image.show(title="Generated Image")
    yellow_mask.show(title="Mask")
    merge_img.show(title="Merged Image")

    # You can save the images if needed
    # generated_image.save("generated_image.png")
    # yellow_mask.save("yellow_mask.png")
    # merge_img.save("merged_image.png")

# Example usage
image_path = 'path/to/your/image.jpg'
mask_path = 'path/to/your/mask.jpg'
predict_without_gui(image_path, mask_path)
