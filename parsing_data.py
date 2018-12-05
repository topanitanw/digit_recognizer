import PIL as pil
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

############################################################
# helper functions
############################################################
def byte_int(ubyte):
    # convert a bytes object to an int
    # return int
    return int.from_bytes(ubyte, byteorder='big', signed=False)

def read_int(file, nbyte):
    # read nbyte bytes from a file
    # return an int
    return byte_int(file.read(nbyte))

def read_array(file, nbyte, nelement):
    # read nbyte bytes for nelement numbers from a file object
    # return a list of nelement integers
    arr = []
    for _ in range(nelement):
        element = read_int(file, nbyte)
        arr.append(element)
    return arr

def read_narray(file, num, row, col, nbyte):
    # read nbyte bytes for num arrays from a file object
    # return a 2D row x column list of nelement integers
    images = []
    for _ in range(num):
        array = read_array(file, nbyte, row * col)
        images.append(array)
    return images

############################################################
# exported functions
############################################################
def read_images(filename):
    # read the idx3/image files
    # return a dataframe object
    with open(filename, "rb") as ftraindata:
        magic_number = read_int(ftraindata, 4)
        if magic_number != 2051:
            raise IOError("Error reading a wrong file %s with this magic number %d"
                          % (filename, magic_number))

        item_num = read_int(ftraindata, 4)
        item_num = 1
        row_num = read_int(ftraindata, 4)
        col_num = read_int(ftraindata, 4)
        images = read_narray(ftraindata, item_num, 1, row_num * col_num, 1)
    return pd.DataFrame(images, dtype=np.uint8)

def read_labels(filename):
    # idx1 files (label files)
    with open(filename, "rb") as fdata:
        magic_number = read_int(fdata, 4)
        if magic_number != 2049:
            raise IOError("Error reading a wrong file %s with this magic number %d"
                          % (filename, magic_number))

        nitem = read_int(fdata, 4)
        labels = read_array(fdata, 1, nitem)
    return pd.DataFrame({'label': labels}, dtype=np.uint8)

def write_csv(df, filename):
    # write the dataframe to the csv file, and create an image id
    df.to_csv(filename, index=True, index_label="id")
