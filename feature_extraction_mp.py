import numpy as np
import sklearn
from multiprocessing import Pool, cpu_count

index_glob = 0


def convolution(img, kernel):
    h = img.shape[0]
    w = img.shape[1]
    img_w_pad = np.zeros((h + 2, w + 2))
    img_w_pad[0, 1:-1] = img[0]
    img_w_pad[-1, 1:-1] = img[-1]
    img_w_pad[1:-1, 0] = img[:, 0]
    img_w_pad[1:-1, -1] = img[:, -1]
    img_w_pad[1:-1, 1:-1] = img

    grad = np.zeros_like(img)
    for i in range(h):
        for j in range(w):
            grad[i, j] = np.sum(kernel * img_w_pad[i:i + 3, j:j + 3])
    return grad


def grad_extract(img, kernel):
    gx = convolution(img, kernel)
    gy = convolution(img, kernel.T)
    grad_mag = np.sqrt(gx ** 2 + gy ** 2)
    grad_angle = np.arctan2(gy, gx)
    grad_angle = np.rad2deg(np.where(grad_angle < 0, grad_angle + np.pi, grad_angle))
    return grad_mag, grad_angle


def block_extract(grad_mag, grad_angle):
    feature = np.zeros(9)
    for i in range(grad_mag.shape[0] ** 2):
        bin_ratio, bin_idx = np.modf((grad_angle.item(i) + 10) / 20)
        bin_idx = int(bin_idx % 9)
        feature[bin_idx - 1] += (1 - bin_ratio) * grad_mag.item(i)
        feature[bin_idx] += bin_ratio * grad_mag.item(i)
    return feature


def block_feature(grad_mag, grad_angle):
    h = grad_mag.shape[0]
    w = grad_mag.shape[1]
    block_histogram = np.zeros((h // 8, w // 8, 9))
    feature_block = np.zeros((h // 8 - 1, w // 8 - 1, 36))
    for i in range(h // 8):
        for j in range(w // 8):
            block_histogram[i, j] = block_extract(grad_mag[8 * i:8 * i + 8, 8 * j:8 * j + 8],
                                                  grad_angle[8 * i:8 * i + 8, 8 * j:8 * j + 8])
    for i in range(h // 8 - 1):
        for j in range(w // 8 - 1):
            feature_block[i, j] = np.linalg.norm(np.concatenate(block_histogram[i:i + 2, j:j + 2], axis=None))
    return np.concatenate(feature_block, axis=None)


def extract(img):
    global index_glob
    print("Xu ly den anh " + str(index_glob))
    index_glob += 1
    k = img.shape[0]
    x_kernel = np.array([[0., 0., 0.],
                         [-1., 0, 1.],
                         [0., 0., 0.]])
    feature = []
    for c in range(k):
        c_img = np.float32(img[c]) / 255.0
        grad_mag, grad_angle = grad_extract(c_img, x_kernel)
        feature.append(block_feature(grad_mag, grad_angle))
    return np.concatenate(feature)


def preprocess_1item(item):
    return extract(item)


def multipreprocess(data, number_process=None):
    if number_process is None:
        number_process = max(cpu_count() - 1, 1)
    else:
        number_process = max(number_process, cpu_count() - 1)
    print('Number of CPU: {}'.format(number_process))
    pool = Pool(number_process)
    data_prep = pool.map(preprocess_1item, data)
    return data_prep


def feature_extraction(imgs, number_process=None):
    data = [img for img in imgs]
    data = multipreprocess(data)

    # data = [extract(img) for img in imgs]
    return np.asarray(data)


if __name__ == "__main__":
    # train_pos = np.load('train_pos_rgb.npy')
    # train_neg = np.load('train_neg_rgb.npy')
    # train_pos = np.load('train_pos.npy')
    # train_neg = np.load('train_neg.npy')

    # data_pos = feature_extraction(train_pos)
    # np.save('feature_pos.npy', data_pos)
    # data_pos = np.load('feature_pos_train.npy')

    # data_neg = feature_extraction(train_neg)
    # np.save('feature_neg.npy', data_neg)
    # data_neg = np.load('feature_neg_train.npy')


    # test_pos = np.load('test_pos.npy')
    # test_neg = np.load('test_neg.npy')

    # data_pos = feature_extraction(test_pos)
    # np.save('feature_pos_test.npy', data_pos)
    # data_pos = np.load('feature_pos_test.npy')

    # data_neg = feature_extraction(test_neg)
    # np.save('feature_neg_test.npy', data_neg)
    # data_neg = np.load('feature_neg_test.npy')
