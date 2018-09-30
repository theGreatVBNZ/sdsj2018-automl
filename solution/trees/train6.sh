echo Task 6
echo ------------------------------------
mkdir ./t6; python train.py --mode classification --train-csv ~/cnt/sdsj2018-automl/data/check_6_c/train.csv --model-dir ./t6
python predict.py --test-csv ~/cnt/sdsj2018-automl/data/check_6_c/test.csv --prediction-csv ./t6/prediction.csv --model-dir ./t6
echo ----------***************-----------
