# mk_tracks.py
# Generates M random tracks in an N by N image
import numpy as np
import cv2


if __name__=='__main__':
    N = 1000
    tracks = []
    # Generate the initial tracks images
    initial_track = np.zeros((N, N))

    # Make M tracks
    M = 50
    initial_track[102, 124] = 1
    initial_track[646, 334] = 1
    initial_track[234, 631] = 1
    initial_track[198, 248] = 1
    initial_track[900, 900] = 1
    tracks.append(initial_track)

    for track_idx in range(1, M):
        next_track = np.zeros_like(initial_track)
        thing = np.where(tracks[track_idx-1] > 0)
        for (loc_idx, loc) in enumerate(np.argwhere(tracks[track_idx-1] > 0)):
            next_track[loc[0] + 1, loc[1] + 1] = 1

        tracks.append(next_track)
        # print(next_track)

    full_tracks = np.zeros_like(initial_track)
    for track_idx in range(1, M):
        full_tracks += tracks[track_idx]

    full_img = full_tracks*255
    cv2.imshow('img',full_img)
    cv2.waitKey()
