echo Task 2
echo ------------------------------------
mkdir ./t2; python train.py --mode regression --train-csv ~/cnt/sdsj2018-automl/data/check_2_r/train.csv --model-dir ./t2
python predict.py --test-csv ~/cnt/sdsj2018-automl/data/check_2_r/test.csv --prediction-csv ./t2/prediction.csv --model-dir ./t2
echo ----------***************-----------
