import segmentation_models as sm
import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
import pandas as pd
from sklearn.metrics import confusion_matrix

## ================================= IMPORTANT VARIABLE =================================
SIZE_X = 128 
SIZE_Y = 128

list_color = [[0, 0, 0],
            [0, 128, 128],
            [128, 0, 0],
            [0, 128, 0],
            [0, 0, 128]]

colormap = np.array(list_color)
colormap = colormap.astype(np.uint8)

## ================================= EVAL INFERENCE =================================

def preprocess_input(backbone) :
    var = sm.get_preprocessing(backbone)
    return var

def infer(model, image_tensor):
    predictions = model.predict(np.expand_dims((image_tensor), axis=0))
    predictions = np.squeeze(predictions)
    predictions = np.argmax(predictions, axis=2)
    return predictions

def probs(model, image_tensor):
    predictions = model.predict(np.expand_dims((image_tensor), axis=0))
    predictions = np.squeeze(predictions)
    return predictions

def decode_segmentation_masks(mask, colormap, n_classes):
    r = np.zeros_like(mask).astype(np.uint8)
    g = np.zeros_like(mask).astype(np.uint8)
    b = np.zeros_like(mask).astype(np.uint8)
    for l in range(0, n_classes):
        idx = mask == l
        r[idx] = colormap[l, 0]
        g[idx] = colormap[l, 1]
        b[idx] = colormap[l, 2]
    rgb = np.stack([r, g, b], axis=2)
    return rgb


def get_overlay(image, colored_mask):
    image = tf.keras.preprocessing.image.array_to_img(image)
    image = np.array(image).astype(np.uint8)
    overlay = cv2.addWeighted(image, 0.35, colored_mask, 0.65, 0)
    return overlay


def plot_samples_matplotlib(display_list, plot_dir, figsize=(5, 3)):
    title = ['Raw Image', 'Prediction', 'Prediction GT', 'True GT']
    for idx, ttl in enumerate(title) :
        name = f'{ttl}{idx}.jpg'
        dirs = f'{plot_dir}/{name}'
        plt.figure(figsize=(5, 5))
        plt.title(name)
        plt.imshow(display_list[idx])
        plt.savefig(dirs, dpi=250)
        plt.clf

    #_, axes = plt.subplots(nrows=1, ncols=len(display_list), figsize=figsize)
    #for i in range(len(display_list)):
        #axes[i].imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        #axes[i].set_title(title[i], fontweight ="bold")
        # if display_list[i].shape[-1] == 3:
        #     axes[i].imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        # else:
        #     axes[i].imshow(display_list[i])
    #plt.savefig(plot_dir, dpi=250)
    #plt.show()


def plot_predictions(images_list, raw, gt, colormap, model, plot_dir):
    for idx, image_file in enumerate(images_list):
        image_tensor = image_file
        image_raw = raw[idx]
        
        prediction_mask = infer(image_tensor=image_tensor, model=model)
        prediction_colormap = decode_segmentation_masks(prediction_mask, colormap, len(colormap))

        true_mask = gt[idx].reshape(gt[idx].shape[0], gt[idx].shape[1])
        true_colormap = decode_segmentation_masks(true_mask, colormap, len(colormap))

        overlay = get_overlay(image_tensor, prediction_colormap)
        dirs = plot_dir
        plot_samples_matplotlib(display_list=[image_raw, overlay, prediction_colormap, true_colormap], plot_dir=dirs, figsize=(18, 14))
    
def plot_train(display_list, plot_dir, epoch=np.arange(0, 150), figsize=(5, 3)):
    lbl = [['Train_Accuracy', 'Val_Accuracy',], ['Train_Loss', 'Val_Loss']]
    title = ['Accuracy', 'Loss']
    #_, axes = plt.subplots(nrows=1, ncols=len(display_list), figsize=figsize)
    for idx, i in enumerate(display_list):
        kind = i
        for idx2, j in enumerate(kind) :
            plt.plot(epoch, j, label=lbl[idx][idx2])
        plt.title(title[idx])
        plt.legend()
        plt.savefig(f'{plot_dir}/{title[idx]}.jpg', dpi=250)
        plt.clf()
        # if display_list[i].shape[-1] == 3:
        #     axes[i].imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        # else:
        #     axes[i].imshow(display_list[i])
    #plt.show()

## ================================= EVAL FUNCTION =================================
def columnar (img) : 
    shape = img.shape
    temp = np.zeros(shape=shape)

    for x in range(shape[0]) :
        for y in range(shape[1]) :
            a1 = img[x, y, :]
            #comp = a1 == [0, 0, 128]
            if ((a1[0] == 0) & (a1[1] == 128) & (a1[2] == 0)) :
                temp[x, y, :] = [0, 128, 0]
            else : 
                temp[x, y, :] = [0, 0, 0]
    return temp

def lesi (img) :
    shape = img.shape
    temp = np.zeros(shape=shape)

    for x in range(shape[0]) :
        for y in range(shape[1]) :
            a1 = img[x, y, :]
            #comp = a1 == [0, 0, 128]
            if ((a1[0] == 0) & (a1[1] == 0) & (a1[2] == 128)) :
                temp[x, y, :] = [0, 0, 128]
            else : 
                temp[x, y, :] = [0, 0, 0]
    return temp

def TZ (img) :
    shape = img.shape
    temp = np.zeros(shape=shape)

    for x in range(shape[0]) :
        for y in range(shape[1]) :
            a1 = img[x, y, :]
            #comp = a1 == [0, 0, 128]
            if ((a1[0] == 0) & (a1[1] == 128) & (a1[2] == 128)) :
                temp[x, y, :] = [128, 0, 0]
            elif ((a1[0] == 0) & (a1[1] == 128) & (a1[2] == 0)) :
                temp[x, y, :] = [128, 0, 0]
            elif ((a1[0] == 0) & (a1[1] == 0) & (a1[2] == 128)) :
                temp[x, y, :] = [128, 0, 0]
            else : 
                temp[x, y, :] = [0, 0, 0]
    return temp

def CA (img) :
    shape = img.shape
    temp = np.zeros(shape=shape)

    for x in range(shape[0]) :
        for y in range(shape[1]) :
            a1 = img[x, y, :]
            #comp = a1 == [0, 0, 128]
            if ((a1[0] == 0) & (a1[1] == 0) & (a1[2] == 128)) :
                temp[x, y, :] = [0, 128, 128]
            elif ((a1[0] == 0) & (a1[1] == 128) & (a1[2] == 128)) :
                temp[x, y, :] = [0, 128, 128]
            elif ((a1[0] == 0) & (a1[1] == 128) & (a1[2] == 0)) :
                temp[x, y, :] = [0, 128, 128]
            elif ((a1[0] == 0) & (a1[1] == 0) & (a1[2] == 128)) :
                temp[x, y, :] = [0, 128, 128]
            else : 
                temp[x, y, :] = [0, 0, 0]
    return temp

def mean_eval(test_list, mask_list, savedModel) :
    iou_list = []
    dice_list = []
    for idx, value in enumerate(test_list) :
        img_tensor = value
        true = np.array(mask_list[idx])
        true = true.reshape(SIZE_X, SIZE_Y)
        pred = infer(savedModel, img_tensor)

        true_final = decode_segmentation_masks(true, colormap, len(colormap))
        pred_final = decode_segmentation_masks(pred, colormap, len(colormap))

        y_true = true_final.flatten()
        y_pred = pred_final.flatten()
        
        current = confusion_matrix(y_true, y_pred, labels=[0, 128])
        intersection = np.diag(current)
        ground_truth_set = current.sum(axis=1)
        predicted_set = current.sum(axis=0)
        union = ground_truth_set + predicted_set - intersection
        IoU = intersection / union.astype(np.float32)
        dice = (2 * intersection) / (ground_truth_set + predicted_set)
        
        iou_list.append(np.mean(IoU))
        dice_list.append(np.mean(dice))
    
    return [np.mean(iou_list), np.mean(dice_list)]

def class_eval(test_list, mask_list, classes, savedModel) :
    iou_list = []
    dice_list = []
    for idx, value in enumerate(test_list) :
        img_tensor = value
        true = np.array(mask_list[idx])
        true = true.reshape(SIZE_X, SIZE_Y)
        pred = infer(savedModel, img_tensor)

        true_final = decode_segmentation_masks(true, colormap, len(colormap))
        pred_final = decode_segmentation_masks(pred, colormap, len(colormap))
        
        if (classes == 'columnar') :
            true_cls = columnar(true_final)
            pred_cls = columnar(pred_final) 
        elif (classes == 'lesi') :
            true_cls = lesi(true_final)
            pred_cls = lesi(pred_final)
        elif (classes == 'TZ') :
            true_cls = TZ(true_final)
            pred_cls = TZ(pred_final)
        else :
            true_cls = CA(true_final)
            pred_cls = CA(pred_final)

        y_true = true_cls.flatten()
        y_pred = pred_cls.flatten()
        
        current = confusion_matrix(y_true, y_pred, labels=[0, 128])
        intersection = np.diag(current)
        ground_truth_set = current.sum(axis=1)
        predicted_set = current.sum(axis=0)
        union = ground_truth_set + predicted_set - intersection
        IoU = intersection / union.astype(np.float32)
        dice = (2 * intersection) / (ground_truth_set + predicted_set)
        
        iou_list.append(np.mean(IoU))
        dice_list.append(np.mean(dice))

    iou_list = pd.Series(iou_list).fillna(0)
    dice_list = pd.Series(dice_list).fillna(0)

    return [np.mean(np.array(iou_list)), np.mean(np.array(dice_list))]

def pr_curve(test_list, mask_list, savedModel) :
    recall_list = []
    prec_list = []
    for idx, value in enumerate(test_list) :
        img_tensor = value
        true = np.array(mask_list[idx])
        true = true.reshape(SIZE_X, SIZE_Y)
        pred = infer(savedModel, img_tensor)

        true_final = decode_segmentation_masks(true, colormap, len(colormap))
        pred_final = decode_segmentation_masks(pred, colormap, len(colormap))

        y_true = true_final.flatten()
        y_pred = pred_final.flatten()
        
        current = confusion_matrix(y_true, y_pred, labels=[0, 128])
        tp = current[1][1]
        tn = current[0][0]
        fp = current[1][0]
        fn = current[0][1]

        recall = tp / (tp + fn)
        precision = tp / (tp + fp)

        if (recall < 0.9) :
            recall_list.append(0.0)
        else :
            recall_list.append(recall)

        if (precision < 0.9) :
            prec_list.append(0)
        else :
            prec_list.append(precision)
        # recall_list.append(recall)
        # prec_list.append(precision)

    return [recall_list, prec_list]
