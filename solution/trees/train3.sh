echo Task 3
echo ------------------------------------
mkdir ./t3; python train.py --mode regression --train-csv ~/cnt/sdsj2018-automl/data/check_3_r/train.csv --model-dir ./t3
python predict.py --test-csv ~/cnt/sdsj2018-automl/data/check_3_r/test.csv --prediction-csv ./t3/prediction.csv --model-dir ./t3
echo ----------***************-----------
