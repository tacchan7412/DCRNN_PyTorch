# data generation
echo "don't forget to update the yaml"
python -m scripts.generate_training_data --config_filename=data/data_generation/dft.yaml --traffic_df_filename=/tmp2/PEMS/pems-bay-flow.h5
python -m scripts.generate_training_data --config_filename=data/data_generation/gaussian.yaml --traffic_df_filename=/tmp2/PEMS/pems-bay-flow.h5
python -m scripts.generate_training_data --config_filename=data/data_generation/ss.yaml --traffic_df_filename=/tmp2/PEMS/pems-bay-flow.h5
python -m scripts.generate_training_data --config_filename=data/data_generation/ssf.yaml --traffic_df_filename=/tmp2/PEMS/pems-bay-flow.h5
