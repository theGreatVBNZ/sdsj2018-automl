echo Task 4
echo ------------------------------------
mkdir ./t4; python train.py --mode classification --train-csv ~/cnt/sdsj2018-automl/data/check_4_c/train.csv --model-dir ./t4
python predict.py --test-csv ~/cnt/sdsj2018-automl/data/check_4_c/test.csv --prediction-csv ./t4/prediction.csv --model-dir ./t4
echo ----------***************-----------
