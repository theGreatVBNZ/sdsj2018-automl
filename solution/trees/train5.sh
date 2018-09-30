echo Task 5
echo ------------------------------------
mkdir ./t5; python train.py --mode classification --train-csv ~/cnt/sdsj2018-automl/data/check_5_c/train.csv --model-dir ./t5
python predict.py --test-csv ~/cnt/sdsj2018-automl/data/check_5_c/test.csv --prediction-csv ./t5/prediction.csv --model-dir ./t5
echo ----------***************-----------
