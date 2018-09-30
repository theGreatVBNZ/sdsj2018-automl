echo Task 7
echo ------------------------------------
mkdir ./t7; python train.py --mode classification --train-csv ~/cnt/sdsj2018-automl/data/check_7_c/train.csv --model-dir ./t7
python predict.py --test-csv ~/cnt/sdsj2018-automl/data/check_7_c/test.csv --prediction-csv ./t7/prediction.csv --model-dir ./t7
echo ----------***************-----------
