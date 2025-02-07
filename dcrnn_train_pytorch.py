from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import yaml
import os

from lib.utils import load_graph_data
from lib.lookup import search_id
from model.pytorch.dcrnn_supervisor import DCRNNSupervisor


def main(args):
    with open(args.config_filename) as f:
        supervisor_config = yaml.load(f)
        if args.rep:
            supervisor_config['param']['rep'] = args.rep
            print('overwrite rep parameter with argument')

        graph_pkl_filename = supervisor_config['data'].get('graph_pkl_filename')
        sensor_ids, sensor_id_to_ind, adj_mx = load_graph_data(graph_pkl_filename)

        id_str = search_id(supervisor_config['alg'], supervisor_config['param'])
        model_dir = supervisor_config['train']['model_dir']
        supervisor_config['train']['model_dir'] = os.path.join(model_dir, id_str)
        dset_dir = supervisor_config['data']['dataset_dir']
        supervisor_config['data']['dataset_dir'] = os.path.join(dset_dir, id_str)

        supervisor = DCRNNSupervisor(adj_mx=adj_mx, **supervisor_config)

        supervisor.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_filename', default=None, type=str,
                        help='Configuration filename for restoring the model.')
    parser.add_argument('--use_cpu_only', default=False, type=bool, help='Set to true to only use cpu.')
    parser.add_argument('--rep', default=None, type=int, help='trial No.')
    args = parser.parse_args()
    main(args)
