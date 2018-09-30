echo Task 8
echo ------------------------------------
mkdir ./t8; python train.py --mode classification --train-csv ~/cnt/sdsj2018-automl/data/check_8_c/train.csv --model-dir ./t8
python predict.py --test-csv ~/cnt/sdsj2018-automl/data/check_8_c/test.csv --prediction-csv ./t8/prediction.csv --model-dir ./t8
echo ----------***************-----------
