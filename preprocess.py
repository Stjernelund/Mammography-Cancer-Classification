import cv2 as cv
from skimage import measure
from skimage.measure import regionprops, label
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

def resize_images(paths,dim):
    #paths must be a series of paths
    #dim must be a tuple of (width,height), (256,256) for example
    for i in range(len(paths)):
        img = cv.imread(paths.values[i])
        if (type(img) == type(None)):
            pass
        else:
            img = cv.resize(img, dim, interpolation=cv.INTER_AREA)
            cv.imwrite(paths.values[i], img)

def cropBorders(img, l=0.01, r=0.01, u=0.04, d=0.04):
    #Crop borders of full size images
    nrows, ncols = img.shape
    l_crop = int(ncols * l)
    r_crop = int(ncols * (1 - r))
    u_crop = int(nrows * u)
    d_crop = int(nrows * (1 - d))
    cropped_img = img[u_crop:d_crop, l_crop:r_crop]
    return cropped_img

def add_channels(img):
    return np.expand_dims(img, axis=-1)

#remove small areas from image, keep only the largest connected component
def remove_areas(img):
    mask = img > 0 
    mask_labeled = np.vectorize(label, signature = '(n,m)->(n,m)')(mask)
    rps = regionprops(mask_labeled)
    areas = [r.area for r in rps]
    idx = np.argsort(areas)[::-1]
    new_slc = np.zeros_like(mask_labeled)
    i = idx[0]
    new_slc[tuple(rps[i].coords.T)] = i + 1
    img[new_slc == 0] = 0
    return img

def resize_full_image(paths):
    for i in range(len(paths)):
        img = cv.imread(paths.values[i], cv.IMREAD_GRAYSCALE)
        img = cropBorders(img)
        img = remove_areas(img)
        if (type(img) == type(None)):
            pass
        else:
            img = cv.resize(img, (256, 256), interpolation=cv.INTER_LINEAR_EXACT)
            img = add_channels(img)
            cv.imwrite(paths.values[i], img)

def show_grid_of_images(images,ncols,nrows):
    fig, ax = plt.subplots(nrows,ncols, figsize=(15,15))
    for i in range(nrows):
        for j in range(ncols):
            image = cv.imread(images.values[i*ncols+j], cv.IMREAD_GRAYSCALE)
            ax[i, j].imshow(image)
    plt.show()

#make training and test sets
def make_dataset(paths):
    X = np.array([cv.imread(path) for path in paths])
    X = tf.convert_to_tensor(X)
    return X