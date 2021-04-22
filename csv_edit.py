import pandas as pd
import os
import numpy as np

# def del_previous(file, img_name):
#     anno_file = file + '/' + os.path.splitext(img_name)[0] + '.csv'
#     if not os.path.exists(anno_file):
#         return
#
#     df = pd.read_csv(anno_file, header = 0, encoding='utf-8')
#     df = df.astype(str)
#     lines = [i for i in df.index if df.values[i, 0].find(img_name) >= 0]
#     if len(lines) > 0:
#         df = df.drop(lines)
#         df.to_csv(file, index=0)

def get_previous(file, img_name):
    bboxes = []
    label_list = []
    anno_file = file + '/' + os.path.splitext(img_name)[0] + '.csv'
    if not os.path.exists(anno_file):
        return bboxes, label_list

    data = None
    with open(anno_file, encoding='utf-8') as f:
        data = f.read()

    data = data.split('\n')
    data = [i.split(',') for i in data]

    # df = pd.read_csv(file, header=0, encoding='utf-8')
    # df = df.astype(str)
    # lines = [i for i in df.index if df.values[i, 0].find(img_name) >= 0]
    # bboxes = np.asarray(np.asarray(data[:, 1:5], np.float), np.int)
    # label_list = data[:, -1]
    for line in data:
        if len(line) != 6:
            continue
        coor = np.asarray(line[1:5]).astype(np.float).astype(np.int)

        bboxes.append([coor[0], coor[1], coor[2], coor[3]])
        label_list.append(line[5])


    return bboxes, label_list



if __name__ == '__main__':
    print('main')