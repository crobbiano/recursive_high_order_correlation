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
            for i in range(y - max_distance, y + max_distance + 1):
                for j in range(x - max_distance, x + max_distance + 1):
                    corrmat[y, x] += mat1pad[y + max_distance, x + max_distance] * mat2pad[
                        y - i + max_distance, x - j + max_distance]

    return corrmat


def calcPaths(frames, max_distance=2, threshold=2.5, recursion_order=5):
    Y = []
    Y.append(localCorrelation(frames[0], frames[1], max_distance))
    for i in range(1, recursion_order - 1):
        Y.append(localCorrelation(frames[i], frames[i + 1], max_distance))

    return Y


if __name__ == "__main__":
    x = 5
    y = 5
    frames = []
    frames.append(np.array([[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]))
    frames.append(np.array([[0, 1, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]))
    frames.append(np.array([[0, 1, 1, 0], [0, 0, 1, 0], [0, 0, 0, 0], [0, 0, 0, 0]]))
    frames.append(np.array([[0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]]))
    frames.append(np.array([[0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1]]))

    paths = calcPaths(frames, max_distance=1)

    print(frames)
