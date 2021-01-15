import math
import torch
import config as cfgs
import os
import numpy as np
from collections import defaultdict
from sklearn import preprocessing


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


def delete_bad_tracks(tracks: list, mmsi, period_threshold=14400, anchor_ratio=0.8):
    """
    delete bad tracks: 1) anchored 2) too short tracks 3) value out of bound
    config is expected to detect boundaries.

    :param anchor_ratio: if the ratio of anchoring by moving is above this param, we delete this track
    :param period_threshold: ignore tracks less than 4 hrs (14400s) by default.
    :param tracks: track info list with each element: [lat,lon,sog,cog,timestamp]
    :param mmsi: the mmsi of a ship which these track points belong to.
    :return: True if this track is bad enough.
    """

    # track is too short
    if tracks[-1][-1] - tracks[0][-1] <= period_threshold:
        return True
    track_length = len(tracks)
    counter = 0
    for log in tracks:
        if log[2] < cfgs.SOG_MIN:
            counter += 1
        if cfgs.LAT_MAX < log[0] < cfgs.LAT_MIN:
            return True
        if cfgs.LON_MAX < log[1] < cfgs.LON_MIN:
            return True
        if cfgs.COG_MAX < log[3] < cfgs.COG_MIN:
            return True
        if cfgs.SOG_MAX < log[2]:
            return True
    # print(f"{mmsi} has ratio {float(counter / track_length)}")
    if float(counter / track_length) > anchor_ratio:
        return True


def normalization(track_dict, backward_padding=True):
    """
    A standard normalization for tracks. config should have boundary info.

    :param backward_padding: padding by the last log in each track
    :param track_dict: mmsi as key, the array of the rest info as value.
    :return: Tensor:[attribute, dataset,time]
    """
    ans = torch.Tensor()
    track_list = []
    max_len_track = 0
    for each_track in track_dict.values():
        datapoint = []
        for log in each_track:
            lat_normalized = (log[0] - cfgs.LAT_MIN) / (cfgs.LAT_MAX - cfgs.LAT_MIN)
            lon_normalized = (log[1] - cfgs.LON_MIN) / (cfgs.LON_MAX - cfgs.LON_MIN)
            speed_normalized = (log[2] - 0) / (cfgs.SOG_MAX - 0)
            course_normalized = (log[3] - cfgs.COG_MIN) / (cfgs.COG_MAX - cfgs.COG_MIN)
            single = torch.Tensor([lat_normalized, lon_normalized, speed_normalized, course_normalized])
            datapoint.append(single)
        if len(datapoint) > max_len_track:
            max_len_track = len(datapoint)
        track_list.append(torch.stack(tuple(datapoint), dim=0))
    if backward_padding:
        for index, each_track in enumerate(track_list):
            current_len_track = len(each_track)
            if current_len_track != max_len_track:
                patch = each_track[-1].repeat(max_len_track - current_len_track, 1)
                track_list[index] = torch.cat((track_list[index], patch), dim=0)

    ans = torch.stack(tuple(track_list), dim=0)
    print(ans.shape)
    return ans


def four_hot_encoding(track_dict):
    """
    A four hot encoding for tracks. Be careful.
    See concept at https://arxiv.org/abs/1506.02216
    :param track_dict:
    :return: Tensor:[attribute, dataset,time]
    """
    ans = []
    # TODO implement four-hot encoding
    return ans


def data_preprocessing(raw_data, unused_attribute=None, use_four_hot_encoding=False):
    """
    preprocessing track dataset by mmsi.
    raw_data is initially a list of logs with mmsi, after preprocessing there will be no mmsi info to prevent
    overfitting and logs are combined into tracks by mmsi.

    Firstly, process downsampling to avoid four-hot encoding flooding with massive data in a short period of time.
    Secondly, delete bad tracks: 1) anchored 2) too short tracks 3) value out of bound

    NOTICE
    downsampling is set to 10 mins by default.
    :param use_four_hot_encoding: set True to replace normalization with four-hot encoding.
    :param unused_attribute: unnecessary attribute "heading" (index=5) for four-hot encoding. can introduce others by list.
    :param raw_data: a numpy ndarray which contains all logs retrieved from ais.
            [ [mmsi,lat,lon,sog,cog,heading,timestamp],.. ]
    :return: Tensor:[attribute, dataset,time]
    """
    if unused_attribute is None:
        unused_attribute = [5]
    raw_data = np.delete(raw_data, unused_attribute, axis=1)

    print("Find tracks by mmsi...")
    track_dict = defaultdict(lambda: [])
    for log in raw_data:
        track_dict[log[0]].append(log[1:])
    trash_list = []
    for k, _ in track_dict.items():
        track_dict[k].sort(key=lambda x: x[4])
        # 1  down sampling and interval set to 600 seconds by default
        track_dict[k] = down_sampling(track_dict[k], k)
        # 2 delete bad tracks
        del_flag = delete_bad_tracks(track_dict[k], k)
        if del_flag:
            trash_list.append(k)

    for mmsi in trash_list:
        del track_dict[mmsi]

    # the last part of preprocess to generate a valid dataset.
    if use_four_hot_encoding:
        ans = four_hot_encoding(track_dict)
    else:
        ans = normalization(track_dict)

    print(f"there are {len(ans)} tracks left after preprocessing.")

    return ans
