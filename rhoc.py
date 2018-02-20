import numpy as np


def localCorrelation(mat1, mat2, max_distance=2):
    if mat1.shape != mat2.shape:
        error('Matrix size mismatch')
    (rows, cols) = mat1.shape
    mat1pad = np.pad(mat1, max_distance, mode='constant', constant_values=(0))
    mat2pad = np.pad(mat2, max_distance, mode='constant', constant_values=(0))
    corrmat = np.zeros((rows, cols))
    for y in range(0, rows):
        for x in range(0, cols):
            # Extract the sub image in mat2pad and mult by the corresponding value in mat1
            subimage = mat2pad[y:y + 2 * max_distance + 1, x:x + 2 * max_distance + 1]
            prodimage = mat1[y, x] * subimage
            corrmat[y, x] = prodimage.sum()

    return corrmat


def calcHOCs(frames, max_distance=4, threshold=2.5, recursion_order=5):
    (y, x) = frames[0].shape
    Y = np.zeros((recursion_order, recursion_order, y, x))
    thingg = frames[:][1]
    # Store the frames in the first instance of Y (i.e. Y(0)) to allow for recursion
    Y[0,:,:,:] = frames[:][:]

    for time_idx in range(1, recursion_order - 1):
        for i in range(0, recursion_order - time_idx-1):
            unthresholded = localCorrelation(Y[time_idx-1, i, :, :], Y[time_idx-1, i + 1, :, :], max_distance)
            Y[time_idx, i, :, :] = (unthresholded > threshold).astype(int)

    return Y


if __name__ == "__main__":
    x = 5
    y = 5
    frames = []
    frames.append(np.array([[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]))
    frames.append(np.array([[0, 0, 0, 0], [1, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]))
    frames.append(np.array([[0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0], [0, 0, 0, 0]]))
    frames.append(np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]]))
    frames.append(np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1]]))

    paths = calcHOCs(frames, max_distance=3, threshold=.5)

    print(frames)
