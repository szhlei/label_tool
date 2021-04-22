import cv2
import numpy as np

from PIL import Image, ImageTk

def generate_grids(bbox, width, height):
    grids = []
    shift = [-0.1, -0.05, 0, 0.05, 0.1]
    scale = [0.9, 0.95, 1, 1.05, 1.1]

    for x in shift:
        for y in shift:
            for s in scale:
                grids.append((x, y, s, s))

    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]

    grids = np.asarray(grids) * np.asarray((w, h, w, h))

    bboxes = grids + (bbox[0], bbox[1], bbox[0], bbox[1])

    bboxes = np.asarray(bboxes, np.int)

    bboxes[np.where(bboxes < 0)] = 0
    bboxes[np.where(bboxes[:, 0] > width), 0] = width
    bboxes[np.where(bboxes[:, 1] > height), 1] = height
    bboxes[np.where(bboxes[:, 2] > width), 2] = width
    bboxes[np.where(bboxes[:, 3] > height), 3] = height

    return bboxes

def generate_grids1(bbox, width, height):
    grids = []

    shift = [-12, -9, -6, -3, 0, 3, 6, 9, 12]
    scale = [-12, -9, -6, -3, 0, 3, 6, 9, 12]

    for x in shift:
        for y in shift:
            for s in scale:
                grids.append((x, y, s, s))

    x_center, y_center, w, h = (bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2, bbox[2] - bbox[0], bbox[3] - bbox[1]

    grids = np.asarray(grids)
    grids = grids + [x_center, y_center, w, h]
    bboxes = np.zeros(grids.shape)
    bboxes[:, 0] = grids[:, 0] - np.multiply(grids[:, 2], 0.5)
    bboxes[:, 1] = grids[:, 1] - np.multiply(grids[:, 3], 0.5)
    bboxes[:, 2] = grids[:, 0] + np.multiply(grids[:, 2], 0.5)
    bboxes[:, 3] = grids[:, 1] + np.multiply(grids[:, 3], 0.5)

    bboxes = np.asarray(bboxes, np.int)

    bboxes[np.where(bboxes < 1)] = 1
    bboxes[np.where(bboxes[:, 0] > width), 0] = width
    bboxes[np.where(bboxes[:, 1] > height), 1] = height
    bboxes[np.where(bboxes[:, 2] > width), 2] = width
    bboxes[np.where(bboxes[:, 3] > height), 3] = height

    return bboxes


def generate_grids2(bbox, width, height):
    shift_y, shift_x, scale = np.mgrid[-12:13:3, -12:13:3, -12:13:3]
    shift_x = np.reshape(shift_x, -1)
    shift_y = np.reshape(shift_y, -1)
    scale = np.reshape(scale, -1)

    x_center, y_center, w, h = (bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2, bbox[2] - bbox[0], bbox[3] - bbox[1]

    grids = [shift_x + x_center, shift_y + y_center, scale + w, scale + h]

    grids = np.asarray(grids).T

    bboxes = np.zeros(grids.shape)
    bboxes[:, 0] = np.clip(grids[:, 0] - grids[:, 2] * 0.5, 0, width)
    bboxes[:, 1] = np.clip(grids[:, 1] - grids[:, 3] * 0.5, 0, height)
    bboxes[:, 2] = np.clip(grids[:, 0] + grids[:, 2] * 0.5, 0, width)
    bboxes[:, 3] = np.clip(grids[:, 1] + grids[:, 3] * 0.5, 0, height)

    bboxes = np.asarray(bboxes, np.int)

    return bboxes

def template_match(img, template, bbox):
    img_gray = img.convert('L')
    tem_gray = template.convert('L')

    tem_gray = tem_gray.resize((64, 64))
    tem = np.asarray(tem_gray, 'float')
    mean_tem = np.mean(tem)
    sub_tem = tem - mean_tem
    square_tem = np.sum(sub_tem * sub_tem)**0.5
    height, width = np.array(img).shape[0:2]
    grids = generate_grids2(bbox, width, height)

    ncc_list = []

    for grid in grids:
        crop = img_gray.crop(grid)
        crop = crop.resize((64, 64))
        crop = np.asarray(crop, 'float')
        mean_crop = np.mean(crop)
        sub_crop = crop - mean_crop
        square_crop = np.sum(sub_crop * sub_crop)**0.5

        coor = np.sum(sub_crop * sub_tem)
        ncc = coor/(square_crop * square_tem)
        ncc_list.append(ncc)
    max_ncc = np.max(ncc_list)
    if max_ncc >= 0.6:
        return grids[np.argmax(ncc_list)], max_ncc
    return [], 0

if __name__ == '__main__':
    generate_grids2([20, 30, 80, 120], 640, 480)

