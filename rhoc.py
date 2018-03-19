import numpy as np
import cv2
import time
from scipy.sparse import rand
from PIL import Image


class RHOClibs:
    def __init__(self, Amove=5, alphamove=3, Aangle=5, alphaangle=3, max_distance=4, threshold=1, recursion_order=5,
                 time_steps=5):
        self.Amove = Amove
        self.alphamove = alphamove
        self.Aangle = Aangle
        self.alphaangle = alphaangle
        self.max_distance = max_distance
        self.threshold = threshold
        self.recursion_order = recursion_order
        self.time_steps = time_steps

    def calcScoreAngle(self, i1, j1, i2, j2, A=None, alpha=None):
        if A is None:
            A = self.Aangle
        if alpha is None:
            alpha = self.alphaangle

        gamma = np.abs(np.arctan(j2 / i2) - np.arctan(j1 / i1))
        score = (A - gamma) * alpha
        return score

    def calcScoreMove(self, i1, j1, i2, j2, A=None, alpha=None):
        if A is None:
            A = self.Amove
        if alpha is None:
            alpha = self.alphamove

        score = (A - np.abs(i2 - i1) - np.abs(j2 - j1)) * alpha
        return score

    def calcScores(self, frames, mask, max_distance=None, A1=None, alpha1=None, A2=None, alpha2=None):
        if A2 is None:
            A2 = self.Aangle
        if alpha2 is None:
            alpha2 = self.alphaangle
        if A1 is None:
            A1 = self.Amove
        if alpha1 is None:
            alpha1 = self.alphamove
        if max_distance is None:
            max_distance = self.max_distance

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

    def findPaths(self, frames, mask, max_distance=None, use_score=True):
        if max_distance is None:
            max_distance = self.max_distance

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
            tracks[tloc_idx][tloc[0]+1, tloc[1]+1] = tloc_idx + 1

        # Can only provide a score for N-2 frames since we need to look into the future/past
        # Look at the first N-2 frames to decide if we should keep the path
        # Start with first frame (frame 0) and look forwards
        for frame_idx in range(0, num_frames - 2):
            # Extract the subimage around the current point in the current track in the next frame
            framepad1 = np.pad(frames[frame_idx + 1], max_distance, mode='constant', constant_values=(0))
            framepad2 = np.pad(frames[frame_idx + 2], max_distance, mode='constant', constant_values=(0))
            # Identify all possible single paths going one frame forwards
            for track_idx in range(0, num_tracks):
                # Find where the tracks exists (non-zero entries that are coded to the time step)
                for (loc_idx, loc) in enumerate(np.argwhere(tracks[track_idx] == track_idx + 1 + (frame_idx) / 100)):
                    subim1 = framepad1[loc[0] - max_distance:loc[0] + max_distance + 1,
                             loc[1] - max_distance:loc[1] + max_distance + 1]
                    # Find where the sub image has non zero points
                    for (subloc_idx, subloc) in enumerate(np.argwhere(subim1 > 0)):
                        i1 = loc[0] + subloc[0] - max_distance
                        j1 = loc[1] + subloc[1] - max_distance
                        # Find where the next frame (+2) has non zero points around the subloc of (+1) frame
                        subim2 = framepad2[i1 - max_distance:i1 + max_distance + 1,
                                 j1 - max_distance:j1 + max_distance + 1]
                        nonzero = np.argwhere(subim2 > 0)
                        # scores is (move, angle, i, j)
                        scores = np.zeros((nonzero.shape[0], 4))
                        for (loc_idx1, loc1) in enumerate(nonzero):
                            i2 = i1 + loc1[0] - max_distance
                            j2 = j1 + loc1[1] - max_distance
                            # Score this track
                            move_score = self.calcScoreMove(A=self.Amove,
                                                            alpha=self.alphamove,
                                                            i1=i1,
                                                            j1=j1,
                                                            i2=i2,
                                                            j2=j2)
                            angle_score = self.calcScoreAngle(A=self.Aangle,
                                                              alpha=self.alphaangle,
                                                              i1=i1,
                                                              j1=j1,
                                                              i2=i2,
                                                              j2=j2)
                            scores[loc_idx1, :] = [move_score, angle_score, i1, j1]
                            # FIXME - need to collect all scores for possible paths and select the highest valued one
                            # if angle_score >= 23.56 and move_score >= 3.5:
                            # if use_score:
                            #     tracks[track_idx][i1, j1] = track_idx + 1 + (frame_idx + 1) / 100
                            #     # cv2.imshow('fake', tracks[track_idx])
                            #     # cv2.waitKey(0)
                            #     # print('Move Score: ', move_score, ' Angle Score: ', angle_score)
                            #     if frame_idx is num_frames - 2:
                            #         # Add in the last track
                            #         tracks[track_idx][i2, j2] = track_idx + 1 + (frame_idx + 2) / 100
                            # else:
                            #     tracks[track_idx][i1, j1] = track_idx + 1 + (frame_idx + 1) / 100
                            #
                            #     if frame_idx is num_frames - 2:
                            #         # Add in the last track
                            #         tracks[track_idx][i2, j2] = track_idx + 1 + (frame_idx + 2) / 100
                        # find the largest score and add the point
                        if nonzero.size:
                            max_score_idx = np.argmax(scores[:,0])
                            tracks[track_idx][int(scores[max_score_idx, 2]), int(scores[max_score_idx, 3])] = track_idx + 1 + (frame_idx + 1) / 100

        # Un-pads the tracks
        real_tracks = []
        for track in tracks:
            track_mask = np.copy(track)
            track_mask[track_mask > 0] = 1
            if np.sum(track_mask) > 2:
            # if np.sum(track_mask) > num_frames - 2:
                real_tracks.append(track[max_distance:-max_distance, max_distance:-max_distance])

        return real_tracks

    def localCorrelation(self, mat1, mat2, mat3=None, max_distance=None):
        if max_distance is None:
            max_distance = self.max_distance

        (rows, cols) = mat1.shape
        mat1pad = np.pad(mat1, max_distance, mode='constant', constant_values=(0))
        mat2pad = np.pad(mat2, max_distance, mode='constant', constant_values=(0))
        if mat3 is not None:
            mat3pad = np.pad(mat3, 2 * max_distance, mode='constant', constant_values=(0))
        corrmat = np.zeros_like(mat1)
        for y in range(0, rows):
            for x in range(0, cols):
                if mat1[y, x] == 1:
                    # Extract the sub image in mat2pad and mult by the corresponding value in mat1
                    subimage2 = mat2pad[y:y + 2 * max_distance + 1, x:x + 2 * max_distance + 1]
                    sub2sum = subimage2.sum()
                    corrmat[y, x] = sub2sum
                    if mat3 is not None:
                        subimage3 = mat3pad[y:y + 4 * max_distance + 1, x:x + 4 * max_distance + 1]
                        sub3sum = subimage3.sum()
                        corrmat[y, x] *= sub3sum

        return corrmat

    # Update to look at current time and calculate backwards in time instead of previous time and to the future
    def calcHOCs(self, frames, max_distance=None, threshold=None, recursion_order=None, time_steps=None):
        if max_distance is None:
            max_distance = self.max_distance
        if threshold is None:
            threshold = self.threshold
        if recursion_order is None:
            recursion_order = self.recursion_order
        if time_steps is None:
            time_steps = self.time_steps

        (y, x) = frames[0].shape
        Y = np.zeros((recursion_order, time_steps, y, x))
        # thingg = frames[:][1]
        # Store the frames in the first instance of Y (i.e. Y(0)) to allow for recursion
        Y[0, :, :, :] = frames[0:time_steps][:][:]

        rec_idx = 0
        for rec_num in range(0, recursion_order - 1):
            for time_idx in range(rec_num + 1, time_steps):
                # unthresholded = self.localCorrelation(Y[rec_num, time_idx - 2, :, :], Y[rec_num, time_idx - 1],
                #                                       Y[rec_num, time_idx, :, :], max_distance)
                unthresholded = self.localCorrelation(Y[rec_num, time_idx - 1, :, :], Y[rec_num, time_idx, :, :],
                                                      max_distance=max_distance)
                Y[rec_num + 1, time_idx, :, :] = (unthresholded >= threshold).astype(int)

        # return Y[recursion_order - 1, time_steps - 2, :, :]
        return self.localCorrelation(Y[recursion_order - 1, time_steps - 2, :, :],
                                     Y[recursion_order - 1, time_steps - 1, :, :],
                                     max_distance=max_distance)


if __name__ == "__main__":
    rhoc_tools = RHOClibs(Amove=9,
                          alphamove=.5,
                          Aangle=6,
                          alphaangle=1,
                          max_distance=3,
                          threshold=1,
                          recursion_order=5,
                          time_steps=10)

    N = 300
    tracks = []
    og_tracks = []
    #  ----------  Generate the initial tracks images  ------------
    t0 = time.time()
    initial_track = np.zeros((N, N))

    # Make M tracks
    M = 10
    locations = np.array([[112, 124], [46, 34], [234, 231], [98, 0], [100, 200], [94, 5]])
    for loc in locations:
        initial_track[loc[0], loc[1]] = 1
    f4 = np.random.choice([0.00, 1.00], size=(N, N), p=[.995, .005])
    initial_track += f4
    # cv2.imshow('f4', f4*255)
    # cv2.imshow('init', initial_track*255)
    # cv2.waitKey(0)

    tracks.append(initial_track)
    og_tracks.append(initial_track)

    numlocs = locations.shape[0]
    for track_idx in range(1, M):
        # cv2.imshow('img', tracks[track_idx - 1])
        # cv2.waitKey()
        next_track = np.zeros_like(initial_track)
        og_next_track = np.zeros_like(initial_track)
        for (loc_idx, loc) in enumerate(locations):
            if loc[0] < N - 1 and loc[1] < N - 1:
                if loc_idx is not numlocs - 1:
                    locations[loc_idx][0] += 1
                    locations[loc_idx][1] += 1
                else:
                    locations[loc_idx][0] += 2
                    # locations[loc_idx][1] += 1

                next_track[loc[0], loc[1]] = 1
                og_next_track[loc[0], loc[1]] = loc_idx + 1

        f4 = np.random.choice([0.00, 1.00], size=(N, N), p=[.995, .005])
        # Save the og_track first before adding noise
        og_tracks.append(og_next_track)
        next_track += f4
        next_track[next_track > 1] = 1
        tracks.append(next_track)
        # print(next_track)
    t1 = time.time()
    print('Generation done after ', t1 - t0, ' seconds')
    #  ---------- End generate the initial tracks images  ------------

    #  ---------- Calculate expected score -----------
    move_score = rhoc_tools.calcScoreMove(i1=98, j1=0, i2=100, j2=1)
    angle_score = rhoc_tools.calcScoreAngle(i1=98, j1=0, i2=100, j2=1)
    print('TEMP Move Score: ', move_score, ' Angle Score: ', angle_score)

    #  ---------- END Calculate expected score -----------

    # Find the possible paths for the current time.  RHOCs will contain a 1 in the
    # pixel location from image from the last row (representing the kth order correlation)
    # if there could exist a path between time n and time n+k
    max_distance = 3
    reversedtracks = list(reversed(tracks))
    last_corr_im = rhoc_tools.calcHOCs(frames=reversedtracks)
    t2 = time.time()
    print('RHOCs done after ', t2 - t1, ' seconds')

    paths = rhoc_tools.findPaths(frames=reversedtracks, mask=last_corr_im)
    t3 = time.time()
    print('Paths done after ', t3 - t2, ' seconds')

    full_tracks = np.zeros_like(initial_track)
    full_track_horizontal = initial_track.copy()
    full_track_horizontal = np.hstack((full_track_horizontal, np.ones((initial_track.shape[0], 1))))
    full_og_tracks = np.zeros([initial_track.shape[0], initial_track.shape[1], 3])

    for track_idx in range(1, M):
        full_tracks += tracks[track_idx]
        full_track_horizontal = np.hstack((full_track_horizontal, tracks[track_idx]))
        full_track_horizontal = np.hstack((full_track_horizontal, np.ones((initial_track.shape[0], 1))))

        full_og_tracks[:, :, 0] += np.mod(og_tracks[track_idx] * 220, 255) / 255.0
        full_og_tracks[:, :, 1] += np.mod(og_tracks[track_idx] * 110, 255) / 255.0
        full_og_tracks[:, :, 2] += np.mod(og_tracks[track_idx] * 40, 255) / 255.0

    # im = np.array(full_og_tracks * 255, dtype = np.uint8)

    full_paths = np.zeros_like(initial_track)
    full_paths = np.zeros([initial_track.shape[0], initial_track.shape[1], 3])
    for path_idx, path in enumerate(paths):
        currpath = np.copy(path)
        currpath[currpath > 0] = path_idx + 1
        full_paths[:, :, 0] += np.mod(currpath * 220, 255) / 255.0
        full_paths[:, :, 1] += np.mod(currpath * 110, 255) / 255.0
        full_paths[:, :, 2] += np.mod(currpath * 40, 255) / 255.0

    cv2.imshow('tracks', full_track_horizontal)
    cv2.moveWindow('tracks', 200, 200)
    cv2.imshow('og_tracks', full_og_tracks)
    cv2.moveWindow('og_tracks', 200, 700)
    cv2.imshow('paths', full_paths)
    cv2.moveWindow('paths', 200, 1200)
    cv2.waitKey()
