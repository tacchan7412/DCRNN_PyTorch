# evaluation script
echo "don't forget to update yaml, esp. epoch number"
echo "don't forget to update output_filename argument"
export CUDA_VISIBLE_DEVICES=0

python run_demo_pytorch.py --config_filename data/model/pretrained/PEMS-BAY-FLOW/raw.yaml --output_dir /tmp3/tacchan7412/DCRNN_PyTorch/predictions/raw
python run_demo_pytorch.py --config_filename data/model/pretrained/PEMS-BAY-FLOW/dft.yaml --output_dir /tmp3/tacchan7412/DCRNN_PyTorch/predictions/dft/1
python run_demo_pytorch.py --config_filename data/model/pretrained/PEMS-BAY-FLOW/gaussian.yaml --output_dir /tmp3/tacchan7412/DCRNN_PyTorch/predictions/gaussian/1
python run_demo_pytorch.py --config_filename data/model/pretrained/PEMS-BAY-FLOW/ss.yaml --output_dir /tmp3/tacchan7412/DCRNN_PyTorch/predictions/ss/1
python run_demo_pytorch.py --config_filename data/model/pretrained/PEMS-BAY-FLOW/ssf.yaml --output_dir /tmp3/tacchan7412/DCRNN_PyTorch/predictions/ssf/1
