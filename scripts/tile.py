import numpy as np
from math import ceil
from scipy.ndimage.interpolation import zoom


def normalize(img, make_white):
    maxx, minn = img.max(), img.min()
    img -= minn
    img /= maxx - minn
    if make_white and np.mean(img) < .5:
        img = 1 - img
    return img


def tile_raster_images(image3d,
                       zm=1,
                       margin_width=1,
                       margin_color=.1,
                       make_white=False,
                       global_normalize=False):
    n_images = image3d.shape[0]
    w = n_images // int(np.sqrt(n_images))
    h = ceil(float(n_images) / w)

    if global_normalize:
        image3d = normalize(image3d, make_white)
    else:
        image3d = [normalize(img, make_white) for img in image3d]

    if zm != 1:
        image3d = zoom(image3d, zoom=(1, zm, zm), order=0)

    pad_axes = (0, h * w - n_images), (0, 1), (0, 1)
    pad_width = (margin_width * np.array(pad_axes)).tolist()
    pad_fill = (margin_color * np.array(pad_axes)).tolist()
    image3d = np.pad(image3d, pad_width, 'constant', constant_values=pad_fill)

    image2d = np.vstack([np.hstack([image3d[i * w + j] for j in range(w)])
                         for i in range(h)])
    image2d = image2d[:-margin_width, :-margin_width]
    image2d = (255 * image2d).astype("uint8")
    return image2d


def tile_zagged_vertical(img_stack, margin=2, gray=127):
    hts, wds = zip(*(a.shape for a in img_stack))
    H = sum(hts) + (len(img_stack) + 1) * margin
    W = max(wds) + 2 * margin
    result = np.full((H, W), gray).astype("uint8")

    at = margin
    for (i, img) in enumerate(img_stack):
        result[at:at + hts[i], margin:wds[i] + margin] = img
        at += hts[i] + margin

    return result


def tile_zagged_horizontal(img_stack, *args, **kwargs):
    return tile_zagged_vertical([a.T for a in img_stack], *args, **kwargs).T


def tile_zagged_columns(img_stack, ncolumns=1, margin=4, gray=0):
    n = len(img_stack)
    subs = []
    for i in range(0, n, ncolumns):
        subs.append(tile_zagged_horizontal(img_stack[i:min(n, i + ncolumns)], margin, gray))
    return tile_zagged_vertical(subs, margin, 0)
