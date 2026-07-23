import numpy as np
def conv2d_single_filter(image, filt):
    H, W = image.shape
    f, f2 = filt.shape  # assume square filter, f == f2

    out_H = H-f+1  # you already derived this formula
    out_W = W-f+1

    output = np.zeros((out_H, out_W))

    for i in range(out_H):
        for j in range(out_W):
            window_sum = 0.0
            for r in range(f):
                for c in range(f):
                    window_sum += image[i + r, j + c] * filt[r, c]
            output[i, j] = window_sum

    return output

def im2col(image, f):
    H, W = image.shape
    out_H = H - f + 1
    out_W = W - f + 1
    
    cols = np.zeros((out_H * out_W, f * f))
    
    idx = 0
    for i in range(out_H):
        for j in range(out_W):
            patch = image[i:i+f, j:j+f]  # extract the patch
            cols[idx] = patch.flatten()  # flatten it into the row
            idx += 1
    
    return cols

def conv2d_vectorized(image, filt):
    f = filt.shape[0]
    out_H = image.shape[0] - f + 1
    out_W = image.shape[1] - f + 1
    
    cols = im2col(image, f)         # (num_patches, f*f)
    filt_flat = filt.flatten()      # (f*f,)
    
    output_flat = cols @ filt_flat  # (num_patches,) -- one matmul replaces the loops!
    output = output_flat.reshape(out_H, out_W)
    
    return output

def get_im2col_matrix(image, f):
    
    H, W = image.shape
    out_H = H - f + 1
    out_W = W - f + 1
    num_patches = out_H * out_W
    X_col = np.zeros((num_patches, f * f))
    patch_idx = 0
    for i in range(out_H):
        for j in range(out_W):
            patch = image[i : i+f, j : j+f]
            X_col[patch_idx, :] = patch.flatten()
            patch_idx += 1
            
    return X_col

def conv2d_multi_filter(image, filters):
    n_filters, f, _ = filters.shape
    H, W = image.shape
    
    out_H = H - f + 1
    out_W = W - f + 1
    X_col = get_im2col_matrix(image, f)   
    W_col = filters.reshape(n_filters, -1)    
    out_col = W_col @ X_col.T    
    output = out_col.reshape(n_filters, out_H, out_W)
    
    return output

def get_im2col_matrix_multichannel(image, f):
    C, H, W = image.shape
    out_H = H - f + 1
    out_W = W - f + 1
    num_patches = out_H * out_W
    
    X_col = np.zeros((num_patches, C * f * f))
    
    patch_idx = 0
    for i in range(out_H):
        for j in range(out_W):
            patch = image[:, i:i+f, j:j+f]  # now 3D: (C, f, f)
            X_col[patch_idx, :] = patch.flatten()  # flatten this 3D patch
            patch_idx += 1
    
    return X_col

def get_im2col_matrix_3d(image, f):
    C, H, W = image.shape
    out_H = H - f + 1
    out_W = W - f + 1
    
    num_patches = out_H * out_W
    
    patch_size = C * f * f
    
    X_col = np.zeros((num_patches, patch_size))
    
    patch_idx = 0
    for i in range(out_H):
        for j in range(out_W):
            patch = image[:, i : i+f, j : j+f]
            X_col[patch_idx, :] = patch.flatten()
            patch_idx += 1
            
    return X_col

def conv2d_multichannel(image, filters):
    C, H, W = image.shape
    n_filters, filter_C, f, _ = filters.shape

    assert C == filter_C, f"Channel mismatch! Image has {C}, filters expect {filter_C}"

    out_H = H - f + 1
    out_W = W - f + 1

    X_col = get_im2col_matrix_3d(image, f)
    W_col = filters.reshape(n_filters, -1)

    out_col = W_col @ X_col.T

    output = out_col.reshape(n_filters, out_H, out_W)

    return output

def conv2d_multichannel(image, filters, biases=None):
    C, H, W = image.shape
    n_filters, filter_C, f, _ = filters.shape
    
    assert C == filter_C, f"Channel mismatch! Image has {C}, filters expect {filter_C}"
    
    out_H = H - f + 1
    out_W = W - f + 1
    
    X_col = get_im2col_matrix_3d(image, f)
    
    W_col = filters.reshape(n_filters, -1)
    
    out_col = W_col @ X_col.T
    output = out_col.reshape(n_filters, out_H, out_W)
    
    if biases is not None:
        output = output + biases
        
    return output

def conv_backward_single(dOut, image, filt):
    f = filt.shape[0]

    dFilt = conv2d_single_filter(image, dOut)  # is this literally right? think about it

    flipped_filt = filt[::-1, ::-1]
    pad = f - 1
    padded_dOut = np.pad(dOut, ((pad, pad), (pad, pad)), mode='constant')
    dImage = conv2d_single_filter(padded_dOut, flipped_filt)

    return dImage, dFilt

image = np.random.randint(1, 21, size=(4, 26, 26))       # 4-channel, 26x26
filters = np.random.randint(1, 21, size=(6, 4, 3, 3))    # 6 filters, each seeing 4 channels, 3x3

output = conv2d_multichannel(image, filters)
print(output.shape)  # expect (6, 24, 24)

biases = np.array([10, 20, 30, 40, 50, 60]).reshape(6, 1, 1)
output_with_bias = conv2d_multichannel(image, filters, biases)
output_no_bias = conv2d_multichannel(image, filters, None)

# each filter's output should be shifted by exactly its bias, everywhere
diff = output_with_bias - output_no_bias
print(np.allclose(diff[0], 10), np.allclose(diff[1], 20), np.allclose(diff[5], 60))