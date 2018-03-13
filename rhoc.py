import numpy as np
import cv2
from scipy.sparse import rand
from PIL import Image

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
    num_tracks = onelocations.shape[0]
    num_frames = frames.__len__()

    # Need to track the tracks here, it at any point the track stops then we don't want to return it at all
    # We can identify tracks that stop if the score of the track prevents new points from being added
    # Need to score based on 3 frames: current, current - 1 and current - 2
    tracks = []
    for i in range(0, num_tracks):
        tracks.append(np.zeros_like(maskpad))

    # This labels the individual start paths with different values for identification
    for (tloc_idx, tloc) in enumerate(onelocations):
        tracks[tloc_idx][tloc[0], tloc[1]] = tloc_idx + 1

    # Can only provide a score for N-2 frames since we need to look into the future/past
    # Look at the first N-2 frames to decide if we should keep the path
    # Start with first frame (frame 0) and look forwards
    for frame_idx in range(num_frames - 1, 1, -1):
        # Get the current frame and pad it
        framepad = np.pad(frames[frame_idx], max_distance, mode='constant', constant_values=(0))
        # Identify all possible single paths going one frame forwards
        for track_idx in range(0, num_tracks):
            # Copy the current track
            temp_track = np.copy(tracks[track_idx])
            # Find where the current track has non zero entries
            for (loc_idx, loc) in enumerate(np.argwhere(tracks[track_idx] > 0)):
                # Extract the subimage around the current point in the current track in the next frame
                framepad1 = np.pad(frames[frame_idx-1], max_distance, mode='constant', constant_values=(0))
                subim1 = framepad1[loc[0] - max_distance:loc[0] + max_distance + 1, loc[1] - max_distance:loc[1] + max_distance + 1]
                # Find where the sub image has non zero points
                for (subloc_idx, subloc) in enumerate(np.argwhere(subim1 > 0)):
                    temp_track[loc[0] + subloc[0] - 1, loc[1] + subloc[1] - 1] = tracks[track_idx][loc[0], loc[1]]
                    # i1 = loc[0] + subloc[0] - 1
                    # j1 = loc[1] + subloc[1] - 1
                    i1 = loc[0] + subloc[0]
                    j1 = loc[1] + subloc[1]
                    # tracks[track_idx][loc[0] + subloc[0] - 1, loc[1] + subloc[1] - 1] = tracks[track_idx][
                    #     loc[0], loc[1]]
                    # Find where the next track has non zero entries
                    for (loc_idx1, loc1) in enumerate(np.argwhere(temp_track > 0)):
                        # Extract the subimage around the current point in the current track in the next frame
                        framepad2 = np.pad(frames[frame_idx-2], max_distance, mode='constant', constant_values=(0))
                        subim2 = framepad2[loc1[0] - max_distance:loc1[0] + max_distance + 1, loc1[1] - max_distance:loc1[1] + max_distance + 1]
                        # Find where the sub image has non zero points
                        for (subloc_idx1, subloc1) in enumerate(np.argwhere(subim2 > 0)):
                            temp_track[loc1[0] + subloc1[0] - 1, loc1[1] + subloc1[1] - 1] = temp_track[loc1[0], loc1[1]]
                            i2 = loc1[0] + subloc1[0] - 1
                            j2 = loc1[1] + subloc1[1] - 1
                            # Score this track
                            move_score = calcScoreMove(9, .5, i1, j1, i2, j2)
                            angle_score = calcScoreAngle(2.356, 10, i1, j1, i2, j2)
                            # print('Angle Score: ', angle_score)
                            if move_score > 3:
                                # tracks[track_idx][loc[0] + subloc[0] - 1, loc[1] + subloc[1] - 1] = tracks[track_idx][loc[0], loc[1]]
                                tracks[track_idx][loc1[0] + subloc[0] - 1, loc1[1] + subloc[1] - 1] = tracks[track_idx][loc[0], loc[1]]
                                print('Move Score: ', move_score, ' Angle Score: ', angle_score)





    # for frame_idx in range(1, num_frames):
    #     # print(frames[frame_idx])
    #     frame2pad = np.pad(frames[frame_idx], max_distance, mode='constant', constant_values=(0))
    #     for track_idx in range(0, num_tracks):
    #         for (loc_idx, loc) in enumerate(np.argwhere(tracks[track_idx] > 0)):
    #             subim2 = frame2pad[loc[0] - max_distance:loc[0] + max_distance + 1,
    #                      loc[1] - max_distance:loc[1] + max_distance + 1]
    #             for (subloc_idx, subloc) in enumerate(np.argwhere(subim2 > 0)):
    #                 tracks[track_idx][loc[0] + subloc[0] - 1, loc[1] + subloc[1] - 1] = tracks[track_idx][
    #                     loc[0], loc[1]]

    # Un-pads the tracks
    real_tracks = []
    for track in tracks:
        track_mask = np.copy(track)
        track_mask[track_mask > 0] = 1
        if np.sum(track_mask) > num_frames - 2:
            real_tracks.append(track[max_distance:-max_distance, max_distance:-max_distance])

    return real_tracks


def localCorrelation(mat1, mat2, mat3, max_distance=2):
    (rows, cols) = mat1.shape
    mat1pad = np.pad(mat1, max_distance, mode='constant', constant_values=(0))
    mat2pad = np.pad(mat2, max_distance, mode='constant', constant_values=(0))
    mat3pad = np.pad(mat3, max_distance, mode='constant', constant_values=(0))
    corrmat = np.zeros_like(mat1)
    for y in range(0, rows):
        for x in range(0, cols):
            if mat1[y, x] == 1:
                # Extract the sub image in mat2pad and mult by the corresponding value in mat1
                subimage2 = mat2pad[y:y + 2 * max_distance + 1, x:x + 2 * max_distance + 1]
                subimage3 = mat3pad[y:y + 2 * max_distance + 1, x:x + 2 * max_distance + 1]
                sub2sum = subimage2.sum()
                sub3sum = subimage3.sum()
                corrmat[y, x] = sub2sum*sub3sum

    return corrmat

# Update to look at current time and calculate backwards in time instead of previous time and to the future
def calcHOCs(frames, max_distance=4, threshold=2.5, recursion_order=5, time_steps=5):
    (y, x) = frames[0].shape
    Y = np.zeros((recursion_order, time_steps, y, x))
    # thingg = frames[:][1]
    # Store the frames in the first instance of Y (i.e. Y(0)) to allow for recursion
    Y[0, :, :, :] = frames[0:time_steps][:][:]

    for rec_num in range(0, recursion_order - 1):
        for time_idx in range(time_steps - 1, rec_num, -1):
            unthresholded = localCorrelation(Y[rec_num, time_idx, :, :], Y[rec_num, time_idx - 1], Y[rec_num, time_idx - 2, :, :], max_distance)
            Y[rec_num + 1, time_idx, :, :] = (unthresholded >= threshold).astype(int)

    return Y[recursion_order - 1, time_steps - 1, :, :]


if __name__ == "__main__":
    # x = 5
    # y = 5
    # frames = []
    # frames.append(np.array([[1, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 1, 1, 0]]))
    # frames.append(np.array([[0, 0, 0, 1], [1, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]))
    # frames.append(np.array([[0, 0, 0, 1], [0, 0, 1, 0], [0, 0, 0, 0], [0, 0, 0, 0]]))
    # frames.append(np.array([[0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]]))
    # frames.append(np.array([[0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1]]))

    # Dimensions of the images
    N = 300
    tracks = []
    og_tracks = []
    #  ----------  Generate the initial tracks images  ------------
    initial_track = np.zeros((N, N))

    # Make M tracks
    M = 10
    locations = np.array([[112, 124], [46, 34], [234, 231], [98, 0], [100, 200]])
    for loc in locations:
        initial_track[loc[0], loc[1]] = 1
    f4 = np.random.choice([0.00, 1.00], size=(N,N), p=[.99, .01])
    initial_track += f4
    # cv2.imshow('f4', f4*255)
    # cv2.imshow('init', initial_track*255)
    # cv2.waitKey(0)

    tracks.append(initial_track)
    og_tracks.append(initial_track)

    for track_idx in range(1, M):
        # cv2.imshow('img', tracks[track_idx - 1])
        # cv2.waitKey()
        next_track = np.zeros_like(initial_track)
        og_next_track = np.zeros_like(initial_track)
        for (loc_idx, loc) in enumerate(locations):
            if loc[0] < N - 1 and loc[1] < N - 1:
                next_track[loc[0] + 1, loc[1] + 1] = 1
                og_next_track[loc[0], loc[1]] = loc_idx + 1
                locations[loc_idx][0] += 1
                locations[loc_idx][1] += 1

        f4 = np.random.choice([0.00, 1.00], size=(N,N), p=[.99, .01])
        # Save the og_track first before adding noise
        og_tracks.append(og_next_track)
        next_track += f4
        next_track[next_track>1] = 1
        tracks.append(next_track)
        # print(next_track)
    #  ---------- End generate the initial tracks images  ------------

    # Find the possible paths for the current time.  RHOCs will contain a 1 in the
    # pixel location from image from the last row (representing the kth order correlation)
    # if there could exist a path between time n and time n+k
    RHOCs = calcHOCs(tracks, max_distance=3, threshold=1, time_steps=10)

    # For all of the 1's in the last row of RHOCs, calculate the score at each step
    # (a, b, c, d) = RHOCs.shape
    # last_corr_im = RHOCs[a - 1, 0, :, :]
    last_corr_im = RHOCs
    # scores = calcScores(frames, last_corr_im, max_distance=1)

    paths = findPaths(tracks, last_corr_im, max_distance=4)


    full_tracks = np.zeros_like(initial_track)
    # full_og_tracks = np.zeros_like(initial_track)

    full_og_tracks = np.zeros([initial_track.shape[0], initial_track.shape[1], 3])

    for track_idx in range(1, M):
        full_tracks += tracks[track_idx]

        full_og_tracks[:,:,0] += np.mod(og_tracks[track_idx]*220, 255)/255.0
        full_og_tracks[:,:,1] += np.mod(og_tracks[track_idx]*110, 255)/255.0
        full_og_tracks[:,:,2] += np.mod(og_tracks[track_idx]*40, 255)/255.0

    # im = np.array(full_og_tracks * 255, dtype = np.uint8)

    full_paths = np.zeros_like(initial_track)
    full_paths = np.zeros([initial_track.shape[0], initial_track.shape[1], 3])
    for path_idx, path in enumerate(paths):
        currpath = np.copy(path)
        currpath[currpath > 0] = path_idx + 1
        full_paths[:,:,0] += np.mod(currpath*220, 255)/255.0
        full_paths[:,:,1] += np.mod(currpath*110, 255)/255.0
        full_paths[:,:,2] += np.mod(currpath*40, 255)/255.0
        # full_paths += path * path_idx

    cv2.imshow('tracks', full_tracks)
    cv2.imshow('og_tracks', full_og_tracks)
    cv2.imshow('paths', full_paths)
    cv2.waitKey()
