export CUDA_VISIBLE_DEVICES=0

python run_demo_pytorch.py --config_filename data/model/pretrained/PEMS-BAY/flow_eps1.yaml --output_filename data/results/PEMS-BAY/flow_predictions_eps1.npz
python run_demo_pytorch.py --config_filename data/model/pretrained/PEMS-BAY/flow_eps01.yaml --output_filename data/results/PEMS-BAY/flow_predictions_eps01.npz
python run_demo_pytorch.py --config_filename data/model/pretrained/PEMS-BAY/flow_eps001.yaml --output_filename data/results/PEMS-BAY/flow_predictions_eps001.npz
python run_demo_pytorch.py --config_filename data/model/pretrained/PEMS-BAY/flow_eps0001.yaml --output_filename data/results/PEMS-BAY/flow_predictions_eps0001.npz
python run_demo_pytorch.py --config_filename data/model/pretrained/PEMS-BAY/flow_eps00001.yaml --output_filename data/results/PEMS-BAY/flow_predictions_eps00001.npz
