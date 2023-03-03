# processing data
matlab -nodesktop -nosplash -r data_CAVE


# train 
python train.py --cuda --gpus 0  --ex GSSR --datasetName CAVE --show 

# test
python test.py --cuda --gpus 0 --model_name path