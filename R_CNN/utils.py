from matplotlib.transforms import Bbox
import selectivesearch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle as Rect
import pandas as pd
import os
from PIL import Image
import sys

def extract_regions(src):
    src = np.asarray(src)
    img_lbl, regions = selectivesearch.selective_search(src, \
            scale=200, min_size=100)

    # drop regions which has area lower 5% or larger than src image
    selected_regions = list()
    image_area = np.prod(src.shape[:2])

    for region in regions:
        if region['size'] < 0.05*image_area \
            or region['size'] > image_area:
                continue
        selected_regions += list([region['rect']])
    
    return selected_regions

def show_image(image, bboxes, channel_first=False):
    """
    Params:
        image:
        bboxes: format(x1,y1,x2,y2)
    """
    fig, axis = plt.subplots(figsize=(8,6))

    if channel_first:
        image = np.asarray(image)
        image = np.transpose(image, (1, 2, 0))
    axis.imshow(image)

    if bboxes:
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox
            rect = Rect((x1, y1), x2-x1, y2-y1, lw=3, fill=None, edgecolor='r')
            axis.add_patch(rect) 

    axis.axis('off')
    axis.set_title('Bounding boxes over original image', fontsize=15)
    plt.show()

def get_iou(bbox_1, bbox_2, epsilon=1e-5):
    """
    Parameter:
        bbox_1, bbox_2: bounding boxes following the format (x, y, x,y)
    Return: intersection over union index (iou)
    """

    # intersection
    # top left corner
    tl_x = max(bbox_1[0], bbox_2[0])
    tl_y = max(bbox_1[1], bbox_2[1])
    # bot right corner
    br_x = min(bbox_1[2], bbox_2[2])
    br_y = min(bbox_1[3], bbox_2[3])
    
    # check if none intersection
    if  tl_x > br_x or tl_y > br_y:
        return 0
    intersection_area =  (br_x - tl_x) * (br_y - tl_y)
    area_1 = (bbox_1[2] - bbox_1[0]) * (bbox_2[3] - bbox_2[1])
    area_2 = (bbox_2[2] - bbox_2[0]) * (bbox_2[3] - bbox_2[1])
    union_area = area_1 + area_2 - intersection_area

    return intersection_area / (union_area+epsilon)

def data_processing(csv_path, img_dir):
    """
    Return:
        indexes (list): index of images in the dataset
        images (list): image content
        regions of interest (list): proposal regions
        labels (list): label assigned to correspond to each region
        coord_offsets (list): offset locations betweet label's ground truth bounding box vs region
        bboxes (list): ground truth bounding boxes
    """
    df = pd.read_csv(csv_path)
    df.drop_duplicates(subset=['ImageID'], inplace=True)
    unique_ids = df.ImageID.unique()
    unique_ids = unique_ids[0:100]
    IMG_PATHS, BBOXES, LABELS = list(), list(), list()

    for image_id in unique_ids:
        data = df[df.ImageID == image_id]
        bboxes = data[['XMin', 'YMin', 'XMax', 'YMax']]
        bboxes = np.array(bboxes)
        labels = data.LabelName
        image_path = os.path.join(img_dir, image_id+'.jpg')

        IMG_PATHS.append(image_path)
        BBOXES.append(bboxes)
        LABELS.append(labels)

    IMAGE_IDS, ROIS, TARGETS, COORD_OFFSETS = \
        list(), list(), list(), list()
    for image_id, image_path, bboxes, labels \
            in zip(unique_ids, IMG_PATHS, BBOXES, LABELS):
        image = Image.open(image_path)
        image = np.asarray(image)
        h, w, _ = image.shape
        regions = extract_regions(image)
        # region forwat (x, y, w, h) -> (x, y, X, Y)
        regions = np.array([(x, y, x+w, y+h) \
            for x, y, w, h in regions])
        _bboxes = [np.array(bbox) * np.array([w,h,w,h]) for bbox in bboxes]
        labels = list(labels)
        for region in regions:
            ious = [get_iou(region, bbox) for bbox in _bboxes]
            best_iou_at = np.argmax(ious)
            best_iou = ious[best_iou_at]
            best_bbox = _bboxes[best_iou_at]
            if best_iou > 0.3:
                label = labels[0]
            else:
                label = 'Background'
            r_x, r_y, r_X, r_Y = region
            b_x, b_y, b_X, b_Y = best_bbox
            coord_offset = (b_x-r_x, b_y-r_y, b_X-r_X, b_Y-r_Y)
            coord_offset = np.array(coord_offset) / np.array([w,h,w,h])
            # add to list
            IMAGE_IDS.append(image_id)
            ROIS.append(region)
            TARGETS.append(label)
            COORD_OFFSETS.append(coord_offset)

    return IMAGE_IDS, ROIS, TARGETS, COORD_OFFSETS