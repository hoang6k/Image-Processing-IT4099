import numpy as np
import cv2
import glob


def load_pos(path, padding):
    items = glob.glob(path + '*.png')
    print(len(items))
    return [cv2.imread(path, cv2.IMREAD_GRAYSCALE)[padding:-padding, padding:-padding] for path in items]


def random_subimg(img):
    height = img.shape[0]
    width = img.shape[1]
    max_height = height - 128
    max_width = width - 64
    sub_imgs = []
    for i in range(10):
        np.random.seed(i)
        vtc = np.random.randint(0, max_height)
        hzt = np.random.randint(0, max_width)
        sub_imgs.append(img[vtc:vtc + 128, hzt:hzt + 64].reshape((1, 128, 64)))
    return np.concatenate(sub_imgs)


def load_neg(path):
    items = glob.glob(path + '*')
    print(len(items))
    imgs = []
    for path in items:
        imgs.append(random_subimg(cv2.imread(path, cv2.IMREAD_GRAYSCALE)))
    print(np.concatenate(imgs).shape)
    return np.concatenate(imgs)


if __name__ == "__main__":
    # train_pos_path = 'INRIAPerson/train_64x128_H96/pos/'
    # train_pos = load_pos(train_pos_path, 16)
    # np.save('train_pos.npy', train_pos)
    train_pos = np.load('train_pos.npy')

    # train_neg_path = 'INRIAPerson/Train/neg/'
    # train_neg = load_neg(train_neg_path)
    # np.save('train_neg.npy', train_neg)
    train_neg = np.load('train_neg.npy')
