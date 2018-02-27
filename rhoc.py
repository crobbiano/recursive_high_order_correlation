import numpy as np
import cv2
from scipy.sparse import rand

def calcScoreAngle(A, alpha, i1, j1, i2, j2):
    gamma = np.abs(np.arctan(j2 / i2) - np.arctan(j1 / i1))
    score = (A - gamma) * alpha
    return score


def calcScoreMove(A, alpha, i1, j1, i2, j2):
    score = (A - np.abs(i2 - i1) - np.abs(j2 - j1)) * alpha
    return score


def calcScores(frames, mask, max_distance=4, A1=3, alpha1=1, A2=3, alpha2=1):
    # Ignore the first frame, and use the mask to determine which points
    # to start with from the first frame
    maskpad = np.pad(mask, max_distance, mode='constant', constant_values=(0))
    onelocations = np.argwhere(maskpad > 0)
    # FIXME - unhardcode these once things work
    scores1 = []
    scores2 = []
    # for frame_idx in range(1, frames.__len__()):
    for frame_idx in range(1, 2):
        frame1pad = np.pad(frames[frame_idx - 1], max_distance, mode='constant', constant_values=(0))
        frame2pad = np.pad(frames[frame_idx], max_distance, mode='constant', constant_values=(0))
        for (loc_idx, loc) in enumerate(onelocations):
            subim2 = frame2pad[onelocations[loc_idx][0] - max_distance:onelocations[loc_idx][0] + max_distance + 1,
                     onelocations[loc_idx][1] - max_distance:onelocations[loc_idx][1] + max_distance + 1]
            movedlocations = np.argwhere(subim2 > 0)
            for (moveloc_idx, moveloc) in enumerate(movedlocations):
                movescore = calcScoreMove(A1, alpha1, )

            scores1.append(movescore)
            # print(frames[frame_idx])

    return (0, 0)


def findPaths(frames, mask, max_distance=4):
    # Ignore the first frame, and use the mask to determine which points
    # to start with from the first frame
    maskpad = np.pad(mask, max_distance, mode='constant', constant_values=(0))
    onelocations = np.argwhere(maskpad > 0)

    tracks = []
    for i in range(0, onelocations.shape[0]):
        tracks.append(np.zeros_like(maskpad))

    for (loc_idx, loc) in enumerate(onelocations):
        tracks[loc_idx][loc[0], loc[1]] = loc_idx + 1

    # for frame_idx in range(1, 2):
    for frame_idx in range(1, frames.__len__()):
        # print(frames[frame_idx])
        frame2pad = np.pad(frames[frame_idx], max_distance, mode='constant', constant_values=(0))
        for track_idx in range(0, onelocations.shape[0]):
            for (loc_idx, loc) in enumerate(np.argwhere(tracks[track_idx] > 0)):
                subim2 = frame2pad[loc[0] - max_distance:loc[0] + max_distance + 1,
                         loc[1] - max_distance:loc[1] + max_distance + 1]
                for (subloc_idx, subloc) in enumerate(np.argwhere(subim2 > 0)):
                    tracks[track_idx][loc[0] + subloc[0] - 1, loc[1] + subloc[1] - 1] = tracks[track_idx][
                        loc[0], loc[1]]

    return tracks


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


def calcHOCs(frames, max_distance=4, threshold=2.5, recursion_order=5, time_steps=50):
    (y, x) = frames[0].shape
    Y = np.zeros((recursion_order, time_steps, y, x))
    thingg = frames[:][1]
    # Store the frames in the first instance of Y (i.e. Y(0)) to allow for recursion
    Y[0, :, :, :] = frames[0:time_steps][:][:]

    for rec_num in range(1, recursion_order):
        for time_idx in range(0, recursion_order - rec_num):
            unthresholded = localCorrelation(Y[rec_num - 1, time_idx, :, :], Y[rec_num - 1, time_idx + 1, :, :],
                                             max_distance)
            Y[rec_num, time_idx, :, :] = (unthresholded > threshold).astype(int)

    return Y


if __name__ == "__main__":
    # x = 5
    # y = 5
    # frames = []
    # frames.append(np.array([[1, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 1, 1, 0]]))
    # frames.append(np.array([[0, 0, 0, 1], [1, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]))
    # frames.append(np.array([[0, 0, 0, 1], [0, 0, 1, 0], [0, 0, 0, 0], [0, 0, 0, 0]]))
    # frames.append(np.array([[0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]]))
    # frames.append(np.array([[0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1]]))

    N = 300
    tracks = []
    # Generate the initial tracks images
    initial_track = np.zeros((N, N))

    # Make M tracks
    M = 50
    locations = np.array([[112, 124], [46, 34], [234, 231], [98, 0], [100, 200]])
    for loc in locations:
        initial_track[loc[0], loc[1]] = 1
    f4 = np.random.choice([0, 1], size=(N,N), p=[.99, .01])
    initial_track += np.random.choice([0, 1], size=(N,N), p=[.99, .01])

    tracks.append(initial_track)

    for track_idx in range(1, M):
        # cv2.imshow('img', tracks[track_idx - 1])
        # cv2.waitKey()
        next_track = np.zeros_like(initial_track)
        thing = np.where(tracks[track_idx - 1] > 0)
        # for (loc_idx, loc) in enumerate(np.argwhere(tracks[track_idx - 1] > 0)):
        for (loc_idx, loc) in enumerate(locations):
            if loc[0] < N - 1 and loc[1] < N - 1:
                next_track[loc[0] + 1, loc[1] + 1] = 1
                locations[loc_idx][0] += 1
                locations[loc_idx][1] += 1

        next_track += rand(N, N, density=0.01, format='csr')
        tracks.append(next_track)
        # print(next_track)

    full_tracks = np.zeros_like(initial_track)
    for track_idx in range(1, M):
        full_tracks += tracks[track_idx]

    # Find the possible paths for the current time.  RHOCs will contain a 1 in the
    # pixel location from image from the last row (representing the kth order correlation)
    # if there could exist a path between time n and time n+k
    RHOCs = calcHOCs(tracks, max_distance=1, threshold=.5, time_steps=10)

    # For all of the 1's in the last row of RHOCs, calculate the score at each step
    (a, b, c, d) = RHOCs.shape
    last_corr_im = RHOCs[a - 1, 0, :, :]
    # scores = calcScores(frames, last_corr_im, max_distance=1)
    paths = findPaths(tracks, last_corr_im, max_distance=3)

    full_tracks = np.zeros_like(initial_track)
    for track_idx in range(1, M):
        full_tracks += tracks[track_idx]
        cv2.imshow('img', tracks[track_idx])
        cv2.waitKey()

    cv2.imshow('img', full_tracks)
    cv2.waitKey()
