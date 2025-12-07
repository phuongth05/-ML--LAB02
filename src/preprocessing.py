import numpy as np
import cv2

def normalize_pixels(X):
    return X / 255.0 

def apply_sobel_edge(X):
    
    edges = np.array([
        np.sqrt(cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)**2 + 
                cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)**2)
        for img in X
    ], dtype=np.float32)
    
    edges = np.array([e / (e.max() + 1e-8) for e in edges], dtype=np.float32)
    
    return edges

def apply_canny_edge(X):
    if X.dtype != np.uint8:
        X_uint8 = X.astype(np.uint8)
    else:
        X_uint8 = X
    
    edges = np.array([cv2.Canny(img, 100, 200) for img in X_uint8], dtype=np.float32)
    
    edges = np.array([e / 255.0 for e in edges], dtype=np.float32)

    return edges

def apply_block_averaging(X, block_size=2):
    if isinstance(block_size, int):
        bh = bw = block_size
    else:
        bh, bw = block_size
        
    N, H, W = X.shape

    new_h = H // bh
    new_w = W // bw
    
    valid_h = new_h * bh
    valid_w = new_w * bw

    X_cropped = X[:, :valid_h, :valid_w]

    blocked = X_cropped.reshape(N, new_h, bh, new_w, bw)

    return blocked.mean(axis=(2, 4))

def normalization_Function(x_train, y_train, x_test, y_test):

    x_train_nor = normalize_pixels(x_train)
    x_test_nor  = normalize_pixels(x_test)
    
    x_new_train = x_train_nor.reshape(len(x_train_nor), -1)  # (60000, 784)
    x_new_test  = x_test_nor.reshape(len(x_test_nor), -1)    # (10000, 784)
    
    return x_new_train, y_train, x_new_test, y_test

def edge_normalization_Function(x_train, y_train, x_test, y_test, method='sobel'):
    x_train_nor = normalize_pixels(x_train)
    x_test_nor  = normalize_pixels(x_test)

    x_train_edges = None
    x_test_edges  = None

    if method == 'sobel':
        x_train_edges = apply_sobel_edge(x_train_nor) 
        x_test_edges  = apply_sobel_edge(x_test_nor)

    elif method == 'canny':
        x_train_edges = apply_canny_edge((x_train_nor * 255).astype(np.uint8))
        x_test_edges  = apply_canny_edge((x_test_nor * 255).astype(np.uint8))

    if x_train_edges is not None:
        X_new_train = np.stack([x_train_nor, x_train_edges], axis=-1)
        X_new_test  = np.stack([x_test_nor, x_test_edges], axis=-1)
    else:
        X_new_train = x_train_nor[..., np.newaxis]
        X_new_test  = x_test_nor[..., np.newaxis]

    x_new_train = X_new_train.reshape(X_new_train.shape[0], -1)
    x_new_test  = X_new_test.reshape(X_new_test.shape[0], -1)
    
    return x_new_train, y_train, x_new_test, y_test

def block_averaging_Function(x_train, y_train, x_test, y_test):
    
    x_train_nor = normalize_pixels(x_train)
    x_test_nor  = normalize_pixels(x_test)

    x_train_avg = apply_block_averaging(x_train_nor)
    x_test_avg = apply_block_averaging(x_test_nor)
    x_new_train = x_train_avg.reshape(len(x_train_avg), -1)
    x_new_test = x_test_avg.reshape(len(x_test_avg), -1)
    
    return x_new_train, y_train, x_new_test, y_test

