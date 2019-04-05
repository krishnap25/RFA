"""
helper functions for preprocessing shakespeare data
"""

import json
import os
import re


def __txt_to_data(txt_dir, seq_length=80):
    """Parses text file in given directory into data for next-character model.

    Args:
        txt_dir: path to text file
        seq_length: length of strings in X
    """
    with open(txt_dir, 'r') as inf:
        raw_text = inf.read()
    raw_text = raw_text.replace('\n', ' ')
    raw_text = re.sub(r"   *", r' ', raw_text)
    data_x = []
    data_y = []
    for i in range(0, len(raw_text) - seq_length, 1):
        seq_in = raw_text[i:i + seq_length]
        seq_out = raw_text[i + seq_length]
        data_x.append(seq_in)
        data_y.append(seq_out)
    return data_x, data_y


def parse_data_in(data_dir, users_and_plays_path, raw=False, seq_length=80):
    """
    returns dictionary with keys: users, num_samples, user_data
    raw := bool representing whether to include raw text in all_data
    if raw is True, then user_data key
    removes users with no data
    """
    with open(users_and_plays_path, 'r') as inf:
        users_and_plays = json.load(inf)
    files = os.listdir(data_dir)
    users = []
    hierarchies = []
    num_samples = []
    user_data = {}
    for f in files:
        user = f[:-4]
        filename = os.path.join(data_dir, f)
        with open(filename, 'r') as inf:
            passage = inf.read()
        data_x, data_y = __txt_to_data(filename, seq_length=seq_length)
        if len(data_x) > 0:
            users.append(user)
            if raw:
                user_data[user] = {'raw': passage}
            else:
                user_data[user] = {}
            user_data[user]['x'] = data_x
            user_data[user]['y'] = data_y
            hierarchies.append(users_and_plays[user])
            num_samples.append(len(data_y))
    all_data = {
        'users':  users,
        'hierarchies': hierarchies,
        'num_samples': num_samples,
        'user_data': user_data
    }
    return all_data
