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

def conv_backward_multi(dOut, image, filters):
    """
    dOut: (n_filters, out_H, out_W) - gradient flowing in
    image: (C, H, W) - cached input from forward pass
    filters: (n_filters, C, f, f) - filters used in forward pass
    Returns: dImage (C,H,W), dFilters (n_filters,C,f,f), dBias (n_filters,1,1)
    """
    n_filters, C, f, _ = filters.shape
    H, W = image.shape[1], image.shape[2]
    pad = f - 1

    dImage = np.zeros((C, H, W))
    for k in range(n_filters):
        padded_dOut_k = np.pad(dOut[k], pad_width=pad, mode='constant')
        flipped_filt_k = filters[k][:, ::-1, ::-1]
        dImage += conv2d_multi_filter(padded_dOut_k, flipped_filt_k)

    dOut_col = dOut.reshape(n_filters, -1)
    X_col = get_im2col_matrix_3d(image, f)
    dFilters = (dOut_col @ X_col).reshape(n_filters, C, f, f)

    dBias = np.sum(dOut, axis=(1, 2), keepdims=True)

    return dImage, dFilters, dBias

class Conv2D:
    def __init__(self, in_channels, n_filters, f):
        self.in_channels = in_channels
        self.n_filters = n_filters
        self.f = f

        fan_in = in_channels * f * f
        self.W = np.random.randn(n_filters, in_channels, f, f) * np.sqrt(2 / fan_in)
        self.b = np.zeros((n_filters, 1, 1))

        self.X_batch = None
        self.dW = None
        self.db = None

    def forward(self, X_batch):
        self.X_batch = X_batch
        m = X_batch.shape[0]
        batch_outputs = []

        for i in range(m):
            image = X_batch[i]
            out = conv2d_multichannel(image, self.W, self.b)
            batch_outputs.append(out)

        return np.stack(batch_outputs, axis=0)

    def backward(self, dOut_batch):
        m = self.X_batch.shape[0]

        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        dX_batch = np.zeros_like(self.X_batch)

        pad = self.f - 1

        for i in range(m):
            image = self.X_batch[i]
            dOut = dOut_batch[i]

            self.db += np.sum(dOut, axis=(1, 2), keepdims=True)

            dOut_col = dOut.reshape(self.n_filters, -1)
            X_col = get_im2col_matrix_3d(image, self.f)
            self.dW += (dOut_col @ X_col).reshape(self.W.shape)

            for k in range(self.n_filters):
                padded_dOut_k = np.pad(dOut[k], pad_width=pad, mode='constant')
                flipped_filt_k = self.W[k, :, ::-1, ::-1]
                dX_batch[i] += conv2d_multi_filter(padded_dOut_k, flipped_filt_k)

        self.dW /= m
        self.db /= m

        return dX_batch