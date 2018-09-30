echo Task 1
echo ------------------------------------
mkdir ./t1; python train.py --mode regression --train-csv ~/cnt/sdsj2018-automl/data/check_1_r/train.csv --model-dir ./t1
python predict.py --test-csv ~/cnt/sdsj2018-automl/data/check_1_r/test.csv --prediction-csv ./t1/prediction.csv --model-dir ./t1
echo ----------***************-----------
