import json
import os


def batch_data(data, batch_size, rng=None, shuffle=True, eval_mode=False):
    """
    data is a dict := {'x': [list], 'y': [list]} with optional fields 'y_true': [list], 'x_true' : [list]
    If eval_mode, use 'x_true' and 'y_true' instead of 'x' and 'y', if such fields exist
    returns x, y, which are both lists of size-batch_size lists
    """
    x = data['x_true'] if eval_mode and 'x_true' in data else data['x']
    y = data['y_true'] if eval_mode and 'y_true' in data else data['y']
    raw_x_y = list(zip(x, y))
    if shuffle:
        assert rng is not None
        rng.shuffle(raw_x_y)
    raw_x, raw_y = zip(*raw_x_y)
    batched_x, batched_y = [], []
    for i in range(0, len(raw_x), batch_size):
        batched_x.append(raw_x[i:i + batch_size])
        batched_y.append(raw_y[i:i + batch_size])
    return batched_x, batched_y


def read_data(train_data_dir, test_data_dir):
    """parses data in given train and test data directories

    assumes:
    - the data in the input directories are .json files with 
        keys 'users' and 'user_data'
    - the set of train set users is the same as the set of test set users
    
    Return:
        clients: list of client ids
        groups: list of group ids; empty list if none found
        train_data: dictionary of train data
        test_data: dictionary of test data
    """
    clients = []
    groups = []
    train_data = {}
    test_data = {}

    train_files = os.listdir(train_data_dir)
    train_files = [f for f in train_files if f.endswith('.json')]
    for f in train_files:
        file_path = os.path.join(train_data_dir, f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        clients.extend(cdata['users'])
        if 'hierarchies' in cdata:
            groups.extend(cdata['hierarchies'])
        train_data.update(cdata['user_data'])

    test_files = os.listdir(test_data_dir)
    test_files = [f for f in test_files if f.endswith('.json')]
    for f in test_files:
        file_path = os.path.join(test_data_dir, f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        test_data.update(cdata['user_data'])

    clients = list(train_data.keys())

    return clients, groups, train_data, test_data
