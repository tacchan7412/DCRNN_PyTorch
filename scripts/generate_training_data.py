from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import yaml
import numpy as np
import os
import uuid
import pandas as pd

import sys
import lib.lookup as lt
sys.path.insert(0, './scripts/ssfilterDP')
import mechanism as mc
import filtering


def generate_graph_seq2seq_io_data(df, x_offsets, y_offsets,
                                   add_time_in_day=True, add_day_in_week=False, scaler=None, config=None, alg='', raw=True):
    """
    Generate samples from
    :param df:
    :param x_offsets:
    :param y_offsets:
    :param add_time_in_day:
    :param add_day_in_week:
    :param scaler:
    :return:
    # x: (epoch_size, input_length, num_nodes, input_dim)
    # y: (epoch_size, output_length, num_nodes, output_dim)
    """

    num_samples, num_nodes = df.shape
    data = df.values

    if config:
        T = config['T']
        num_samples = int(T / 0.8)
        data = data[:num_samples]
        if not raw:
            print('adding noise to data')
            if alg == 'gaussian':
                data = mc.gaussian(data, config['eps'], config['delta'], np.sqrt(config['I']))
            elif alg == 'dft':
                for i in range(num_nodes):
                    data[:,i] =  mc.dft_gaussian(data[:,i], config['eps'], config['delta'],
                                                 np.sqrt(config['I']), config['k'])
            elif alg == 'ss':
                for i in range(num_nodes):
                    data[:T,i] = mc.ss_gaussian(data[:T,i], config['eps'], config['delta'],
                                                config['I'], config['k'],
                                                interpolate_kind=config['interpolate_kind'],
                                                smooth=config['smooth'], smooth_window=config['smooth_window'])
            elif alg == 'ssf':
                h = filtering.get_h('gaussian', T, std=config['std'])
                A = filtering.get_circular(h)
                L = sum(h**2)
                sr = mc.srank_circular(h)
                for i in range(num_nodes):
                    data[:T,i] = mc.ssf_gaussian(data[:T,i], A, config['eps'], config['delta'],
                                                 np.sqrt(config['I']), config['k'], sr=sr, L=L,
                                                 interpolate_kind=config['interpolate_kind'],
                                                 smooth=config['smooth'], smooth_window=config['smooth_window'])
            else:
                print('no randomization')

    data = np.expand_dims(data, axis=-1)
    data_list = [data]
    if add_time_in_day:
        time_ind = (df.index.values - df.index.values.astype("datetime64[D]")) / np.timedelta64(1, "D")
        time_in_day = np.tile(time_ind, [1, num_nodes, 1]).transpose((2, 1, 0))
        if config:
            time_in_day = time_in_day[:num_samples]
        data_list.append(time_in_day)
    if add_day_in_week:
        day_in_week = np.zeros(shape=(num_samples, num_nodes, 7))
        day_in_week[np.arange(num_samples), :, df.index.dayofweek] = 1
        if config:
            day_in_week = day_in_week[:num_samples]
        data_list.append(day_in_week)

    data = np.concatenate(data_list, axis=-1)
    # epoch_len = num_samples + min(x_offsets) - max(y_offsets)
    x, y = [], []
    # t is the index of the last observation.
    min_t = abs(min(x_offsets))
    max_t = abs(num_samples - abs(max(y_offsets)))  # Exclusive
    for t in range(min_t, max_t):
        x_t = data[t + x_offsets, ...]
        y_t = data[t + y_offsets, ...]
        x.append(x_t)
        y.append(y_t)
    x = np.stack(x, axis=0)
    y = np.stack(y, axis=0)
    return x, y


def generate_train_val_test(traffic_df_filename, config):
    param_config = config['param']
    df = pd.read_hdf(traffic_df_filename)
    # 0 is the latest observed sample.
    x_offsets = np.sort(
        # np.concatenate(([-week_size + 1, -day_size + 1], np.arange(-11, 1, 1)))
        np.concatenate((np.arange(-11, 1, 1),))
    )
    # Predict the next one hour
    y_offsets = np.sort(np.arange(1, 13, 1))
    # x: (num_samples, input_length, num_nodes, input_dim)
    # y: (num_samples, output_length, num_nodes, output_dim)
    x, y = generate_graph_seq2seq_io_data(
        df.copy(),
        x_offsets=x_offsets,
        y_offsets=y_offsets,
        add_time_in_day=True,
        add_day_in_week=False,
        config=param_config,
        alg=config['alg'],
        raw=False,
    )
    raw_x, raw_y = generate_graph_seq2seq_io_data(
        df.copy(),
        x_offsets=x_offsets,
        y_offsets=y_offsets,
        add_time_in_day=True,
        add_day_in_week=False,
        config=param_config,
        raw=True,
    )

    print("x shape: ", x.shape, ", y shape: ", y.shape)
    # Write the data into npz file.
    # num_test = 6831, using the last 6831 examples as testing.
    # for the rest: 7/8 is used for training, and 1/8 is used for validation.
    num_samples = x.shape[0]
    num_test = round(num_samples * 0.2)
    num_train = round(num_samples * 0.7)
    num_val = num_samples - num_test - num_train

    # train
    x_train, y_train = x[:num_train], y[:num_train]
    # val
    x_val, y_val = (
        x[num_train: num_train + num_val],
        y[num_train: num_train + num_val],
    )
    # test (not randomized)
    x_test, y_test = raw_x[-num_test:], raw_y[-num_test:]

    id_str = str(uuid.uuid4())
    save_dir = os.path.join(config['output_dir'], id_str)
    os.makedirs(save_dir, exist_ok=True)
    param_config['id'] = id_str
    lt.add_row(config['alg'], param_config)

    for cat in ["train", "val", "test"]:
        _x, _y = locals()["x_" + cat], locals()["y_" + cat]
        print(cat, "x: ", _x.shape, "y:", _y.shape)
        np.savez_compressed(
            os.path.join(save_dir, "%s.npz" % cat),
            x=_x,
            y=_y,
            x_offsets=x_offsets.reshape(list(x_offsets.shape) + [1]),
            y_offsets=y_offsets.reshape(list(y_offsets.shape) + [1]),
        )


def main(args):
    print("Generating training data")
    with open(args.config_filename) as f:
        config = yaml.load(f)
    print(config.keys())
    if args.rep:
        config['param']['rep'] = args.rep
        print('overwrite rep parameter with argument')
    generate_train_val_test(args.traffic_df_filename, config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--traffic_df_filename",
        type=str,
        default="data/metr-la.h5",
        help="Raw traffic readings.",
    )
    parser.add_argument('--config_filename', default=None, type=str,
                        help='Configuration filename for generating the data.')
    parser.add_argument('--rep', default=None, type=int, help='trial No.')
    args = parser.parse_args()
    main(args)
