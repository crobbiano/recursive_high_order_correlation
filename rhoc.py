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
            for i in range(-max_distance, max_distance + 1):
                for j in range(-max_distance, max_distance + 1):
                    firstterm = mat1pad[y + max_distance, x + max_distance]
                    secondterm = mat2pad[y + max_distance - i, x + max_distance - j]
                    corrmat[y, x] += firstterm * secondterm

    return corrmat


def calcPaths(frames, max_distance=4, threshold=2.5, recursion_order=5):
    Y = []
    for i in range(0, recursion_order-1):
        Y.append(localCorrelation(frames[i], frames[i + 1], max_distance))

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

    paths = calcPaths(frames, max_distance=2)

    print(frames)
