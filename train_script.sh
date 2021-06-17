# train models
echo "don't forget to update yaml"

export CUDA_VISIBLE_DEVICES=0
python dcrnn_train_pytorch.py --config_filename=data/model/dcrnn_bay_flow/dft.yaml
python dcrnn_train_pytorch.py --config_filename=data/model/dcrnn_bay_flow/gaussian.yaml
python dcrnn_train_pytorch.py --config_filename=data/model/dcrnn_bay_flow/ss.yaml
python dcrnn_train_pytorch.py --config_filename=data/model/dcrnn_bay_flow/ssf.yaml
