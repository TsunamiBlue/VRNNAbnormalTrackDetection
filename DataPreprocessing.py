import math
import torch
import config as cfgs
import os
import numpy as np
from collections import defaultdict


def down_sampling(tracks: list, mmsi, interval=600):
    """
    downsample the tracks to 10 minutes(600 seconds) per point by default.

    :param mmsi: the mmsi of a ship which these track points belong to.
    :param interval: sample interval by second
    :param tracks: track info list with each element: [lat,lon,sog,cog,timestamp]
    :return: modified track info list
    """
    print(f"down sampling by {interval} seconds for mmsi:{mmsi}...")
    ans = [tracks[0]]
    idx = 0
    current_timestamp = tracks[0][-1]
    while idx < len(tracks):
        if tracks[idx][-1] >= current_timestamp + interval:
            ans.append(tracks[idx])
            current_timestamp = tracks[idx][-1]
        idx += 1

    print(f"{len(ans)} track points after downsampling.")
    return ans


def delete_bad_tracks(tracks: list, mmsi,period_threshold=14400):
    """
    delete bad tracks: 1) anchored 2) too short tracks 3) value out of bound
    config is expected to detect boundaries.

    :param period_threshold: ignore tracks less than 4 hrs (14400s) by default.
    :param tracks: track info list with each element: [lat,lon,sog,cog,timestamp]
    :param mmsi: the mmsi of a ship which these track points belong to.
    :return: modified track info list
    """
    print(f"delete bad tracks..")
    ans = []
    if tracks[-1][-1] - tracks[0][-1] <= period_threshold:
        return []
    if cfgs.LA


def data_preprocessing(raw_data, unused_attribute=None):
    """
    preprocessing track dataset by mmsi.
    raw_data is initially a list of logs with mmsi, after preprocessing there will be no mmsi info to prevent
    overfitting and logs are combined into tracks by mmsi.

    Firstly, process downsampling to avoid four-hot encoding flooding with massive data in a short period of time.
    Secondly, delete bad tracks: 1) anchored 2) too short tracks 3) value out of bound

    NOTICE
    downsampling is set to 10 mins by default.
    :param unused_attribute: unnecessary attribute "heading" (index=5) for four-hot encoding. can introduce others by list.
    :param raw_data: a numpy ndarray which contains all logs retrieved from ais.
            [ [mmsi,lat,lon,sog,cog,heading,timestamp],.. ]
    :return: Tensor:[attribute size, dataset size,time size]
    """
    if unused_attribute is None:
        unused_attribute = [5]
    raw_data = np.delete(raw_data, unused_attribute, axis=1)

    print("Find tracks by mmsi...")
    track_dict = defaultdict(lambda: [])
    ans = []
    for log in raw_data:
        track_dict[log[0]].append(log[1:])

    for k, _ in track_dict.items():
        track_dict[k].sort(key=lambda x: x[4])
        # 1  down sampling and interval set to 600 seconds by default
        track_dict[k] = down_sampling(track_dict[k], k)
        # 2 delete bad tracks
        track_dict[k] = delete_bad_tracks(track_dict[k],k)
        if not track_dict[k]:
            del track_dict[k]

    return ans
