import ast
import os
import sys
import bz2
import json
from math import ceil

import numpy as np
from PIL import Image as im


def tile4(t3, margin_color=63, margin_width=1):
    n_images = t3.shape[0]
    w = n_images // int(np.sqrt(n_images))
    h = ceil(float(n_images) / w)

    pad_axes = (0, h*w-n_images), (0, 1), (0, 1)
    pad_width = (margin_width * np.array(pad_axes)).tolist()
    t3 = np.pad(t3, pad_width, 'constant', constant_values=margin_color)
    t2 = np.vstack([
            np.hstack([t3[i * w + j] 
                for j in range(w)])
                    for i in range(h)])
    t2 = t2[:-margin_width, :-margin_width]
    return t2


def read_json_bz2(path2data):
    print("Loading", path2data)
    bz2_fp = bz2.BZ2File(path2data, 'r')
    data = np.array(json.loads(bz2_fp.read().decode('utf-8')))
    bz2_fp.close()
    return data


def read_json(path2data):
    print("Loading", path2data)
    with open(path2data, 'r') as fp:
        data = np.array(json.loads(fp.read()))
    return data


# ######################################################################

def main():
    if len(sys.argv) < 4:
        print("Usage:python3 {0} images.bz2 labels.bz2 labellings.lbl"
              "\n\te.g:- python3 {0} num.images.bz2 num.labels.bz2 "
              "../labellings/numbers09.lbl"
              "\n"
              "Prints tiled images of all images of one class".format(
            sys.argv[0]))
        sys.exit(-1)

    data_file_name = sys.argv[1]
    labels_file_name = sys.argv[2]
    labelings_file_name = sys.argv[3]

    with open(labelings_file_name, 'r') as labels_fp:
        labellings = ast.literal_eval(labels_fp.read())
    reverse_labels = dict((v, k) for k, v in labellings.items())

    labels = read_json_bz2(labels_file_name)
    if data_file_name.endswith('.bz2'):
        imgs = read_json_bz2(data_file_name)
    elif data_file_name.endswith('.json'):
        imgs = read_json(data_file_name)
    imgs *= -255
    imgs += 255
    imgs = imgs.astype('uint8')
    print(f"Images\n\tMax: {imgs.max()} Mean:{imgs.mean()} Min:{imgs.min()} Shape:{imgs.shape}  dtype:{imgs.dtype}")

    dir_name = data_file_name.replace('.bz2', '').replace('.json', '') + '_bmps/'
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    namer = (dir_name + '{}.bmp').format

    for l in range(max(labels)+1):
        indices = labels == l
        name = namer(reverse_labels[l])
        imgs_l = imgs[indices]

        if imgs_l.shape[0] == 0:
            print("{}) Skipping {} with 0 images".format(l, name))
        else:
            print("{}) Printing {} with {} images".format(l, name, imgs_l.shape[0]))
            image_flat = tile4(imgs_l)
            im.fromarray(image_flat).save(name)


if __name__ == "__main__":
    main()