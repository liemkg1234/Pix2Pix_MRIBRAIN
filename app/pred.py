import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import os

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

model_Classifi = tf.keras.models.load_model ('best/efib6split_1.h5', custom_objects={"precision": precision, "recall": recall, 'specificity': specificity, "f1_metric": f1_metric})


#Learning rate
d_lr = 2e-4
g_lr = 2e-4
beta_1=0.5
LAMBDA = 200

#Load model
G = Generator()
generator_optimizer = tf.keras.optimizers.Adam(g_lr, beta_1=beta_1)
D = Discriminator()
discriminator_optimizer = tf.keras.optimizers.Adam(d_lr, beta_1=beta_1)

#load weight
checkpoint_dir = 'best/'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=G,
                                 discriminator=D)
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))



root = tk.Tk()
root.title("Pix2Pix")

def clear_frame():
    for widget in frame1.winfo_children():
        widget.destroy()
    for widget in frame2.winfo_children():
        widget.destroy()
    for widget in frame3.winfo_children():
        widget.destroy()
    for widget in frame4.winfo_children():
        widget.destroy()

    OpenFile2 = tk.Button(frame1, text='Open Image/Mask and Predict', padx=10, pady=3, fg='white', bg='#263D42',
                          command=predict)
    OpenFile2.pack()

    label1 = tk.Label(frame1, text='Input Image')
    label1.config(font=('helvetica', 12))
    label1.pack()
    label2 = tk.Label(frame2, text='Input Mask')
    label2.config(font=('helvetica', 12))
    label2.pack()
    label3 = tk.Label(frame3, text='Generated Image')
    label3.config(font=('helvetica', 12))
    label3.pack()
    label4 = tk.Label(frame4, text='Merge')
    label4.config(font=('helvetica', 12))
    label4.pack()


def load_image():
    path_image = filedialog.askopenfilename(initialdir="C:", title="Select Image", filetypes=(('.jpg','*.jpg'),('all file','*.*')))
    image = Image.open(path_image).convert('L')
    image = image.resize((256, 256))
    #show_image
    image_TK = ImageTk.PhotoImage(image)
    panel = tk.Label(frame1, image=image_TK)
    panel.image = image_TK
    panel.pack()
    return path_image

def load_mask():
    path_image = filedialog.askopenfilename(initialdir="C:", title="Select Image", filetypes=(('.jpg','*.jpg'),('all file','*.*')))
    image = Image.open(path_image)
    image = image.resize((256, 256))
    rgb_img = image.convert('RGB')
    # change color (white -> yellow)
    r, g, b = rgb_img.split()
    b = b.point(lambda i: i * 0.0)
    result = Image.merge('RGB', (r, g, b))

    # show_image
    image_TK = ImageTk.PhotoImage(result)
    panel = tk.Label(frame2, image=image_TK)
    panel.image = image_TK
    panel.pack()

    return path_image

def predict():
    path_image = load_image()
    path_mask = load_mask()
    image, mask = load_image_train(path_image, path_mask)
    #Classifi
    image_classifi, mask_classifi = load_classifi(path_image, path_mask)
    image_classifi = image_classifi[tf.newaxis, ...]
    # print(image_classifi.shape)
    #Classifi
    classifi = model_Classifi.predict(image_classifi)
    if classifi >= 0.5:
        classifi = 1
    else:
        classifi = 0

    #Predict
    G_out = G(image[tf.newaxis, ...], training=True)
    G_image = G_out[0, :, :, 0]
    #Pre,IoU
    # print(G_image.shape)
    # print(mask.shape)
    pre = Precision(G_image, mask[:,:,0])
    iou = IoU(G_image, mask[:,:,0])
    print(pre)
    print(iou)
    G_image = (G_image + 1) * 127.5
    mask = (mask + 1) * 127.5

    image_out = Image.fromarray(G_image.numpy())
    # white to blue
    rgb_G_out = image_out.convert('RGB')
    r, g, b = rgb_G_out.split()
    r = r.point(lambda i: i * 0.0)
    g = g.point(lambda i: i * 0.0)
    result = Image.merge('RGB', (r, g, b))


    #merge
    mask = Image.fromarray(mask[:,:,0].numpy())
    mask_rgb = mask.convert('RGB')
    # change color (white -> yellow)
    r, g, b = mask_rgb.split()
    b = b.point(lambda i: i * 0.0)
    result_mask = Image.merge('RGB', (r, g, b))

    #
    result_arr = np.array(result)
    mask_arr = np.array(result_mask)
    merge_arr = np.add(result_arr, mask_arr)
    merge_img  = Image.fromarray(merge_arr)


    #show genetared
    result = result.resize((256, 256))
    image_TK = ImageTk.PhotoImage(result)
    panel = tk.Label(frame3, image=image_TK)
    panel.image = image_TK
    panel.pack()
    #show show merge
    merge_img = merge_img.resize((256, 256))
    image_TK = ImageTk.PhotoImage(merge_img)
    panel = tk.Label(frame4, image=image_TK)
    panel.image = image_TK
    panel.pack()

    label_Accu = tk.Label(frame4, text='Precision: {:.6f}, IOU: {:.6f}, Class: {}'.format(pre, iou, classifi))
    label_Accu.config(font=('helvetica', 10))
    label_Accu.pack()


#Root
canvas = tk.Canvas(root, height=700, width=1000, bg="#263D42")
canvas.pack()
# Frame
frame1 = tk.Frame(root, bg="white")
frame1.place(relwidth=0.3, relheight=0.45, relx=0.05, rely=0.05)

frame2 = tk.Frame(root, bg="white")
frame2.place(relwidth=0.3, relheight=0.4, relx=0.05, rely=0.52)

frame3 = tk.Frame(root, bg="white")
frame3.place(relwidth=0.3, relheight=0.4, relx=0.65, rely=0.05)

frame4 = tk.Frame(root, bg="white")
frame4.place(relwidth=0.3, relheight=0.42, relx=0.65, rely=0.52)


OpenFile2 = tk.Button(frame1, text='Open Image/Mask and Predict', padx=10, pady=3, fg='white', bg='#263D42',command= predict)
OpenFile2.pack()
clear_fr = tk.Button(root, text='Clear', padx=10, pady=3, fg='white', bg='#263D42', command=clear_frame)

clear_fr.pack()

#textbox
label1 = tk.Label(frame1, text='Input Image')
label1.config(font=('helvetica', 12))
label1.pack()
label2 = tk.Label(frame2, text='Input Mask')
label2.config(font=('helvetica', 12))
label2.pack()
label3 = tk.Label(frame3, text='Generated Image')
label3.config(font=('helvetica', 12))
label3.pack()
label4 = tk.Label(frame4, text='Merge')
label4.config(font=('helvetica', 12))
label4.pack()


root.mainloop()


