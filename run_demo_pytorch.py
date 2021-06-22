import argparse
import numpy as np
import os
import sys
import yaml

from lib.utils import load_graph_data
from lib.lookup import search_id
from model.pytorch.dcrnn_supervisor import DCRNNSupervisor


def run_dcrnn(args):
    with open(args.config_filename) as f:
        supervisor_config = yaml.load(f)

        graph_pkl_filename = supervisor_config['data'].get('graph_pkl_filename')
        sensor_ids, sensor_id_to_ind, adj_mx = load_graph_data(graph_pkl_filename)

        id_str = search_id(supervisor_config['alg'], supervisor_config['param'])
        print(id_str)
        model_dir = supervisor_config['train']['model_dir']
        supervisor_config['train']['model_dir'] = os.path.join(model_dir, id_str)
        dset_dir = supervisor_config['data']['dataset_dir']
        supervisor_config['data']['dataset_dir'] = os.path.join(dset_dir, id_str)

        assert supervisor_config['train']['epoch'] == -1

        supervisor = DCRNNSupervisor(adj_mx=adj_mx, **supervisor_config)
        mean_score, outputs = supervisor.evaluate('test')
        output_dir = os.path.join(args.output_dir, id_str)
        os.makedirs(output_dir, exist_ok=True)
        output_filename = os.path.join(output_dir, 'dcrnn_predictions.npz')
        np.savez_compressed(output_filename, **outputs)
        print("MAE : {}".format(mean_score))
        print('Predictions saved as {}.'.format(output_filename))


if __name__ == '__main__':
    sys.path.append(os.getcwd())
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_cpu_only', default=False, type=str, help='Whether to run tensorflow on cpu.')
    parser.add_argument('--config_filename', default='data/model/pretrained/METR-LA/config.yaml', type=str,
                        help='Config file for pretrained model.')
    parser.add_argument('--output_dir', default='data')
    args = parser.parse_args()
    run_dcrnn(args)
