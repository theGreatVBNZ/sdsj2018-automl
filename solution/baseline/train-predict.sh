echo Task 1
echo ------------------------------------
mkdir ./t1; python train.py --mode regression --train-csv ~/cnt/sdsj2018-automl/data/check_1_r/train.csv --model-dir ./t1
python predict.py --test-csv ~/cnt/sdsj2018-automl/data/check_1_r/test.csv --prediction-csv ./t1/prediction.csv --model-dir ./t1
echo ----------***************-----------

echo Task 2
echo ------------------------------------
mkdir ./t2; python train.py --mode regression --train-csv ~/cnt/sdsj2018-automl/data/check_2_r/train.csv --model-dir ./t2
python predict.py --test-csv ~/cnt/sdsj2018-automl/data/check_2_r/test.csv --prediction-csv ./t2/prediction.csv --model-dir ./t2
echo ----------***************-----------

echo Task 3
echo ------------------------------------
mkdir ./t3; python train.py --mode regression --train-csv ~/cnt/sdsj2018-automl/data/check_3_r/train.csv --model-dir ./t3
python predict.py --test-csv ~/cnt/sdsj2018-automl/data/check_3_r/test.csv --prediction-csv ./t3/prediction.csv --model-dir ./t3
echo ----------***************-----------

echo Task 4
echo ------------------------------------
mkdir ./t4; python train.py --mode classification --train-csv ~/cnt/sdsj2018-automl/data/check_4_c/train.csv --model-dir ./t4
python predict.py --test-csv ~/cnt/sdsj2018-automl/data/check_4_c/test.csv --prediction-csv ./t4/prediction.csv --model-dir ./t4
echo ----------***************-----------

echo Task 5
echo ------------------------------------
mkdir ./t5; python train.py --mode classification --train-csv ~/cnt/sdsj2018-automl/data/check_5_c/train.csv --model-dir ./t5
python predict.py --test-csv ~/cnt/sdsj2018-automl/data/check_5_c/test.csv --prediction-csv ./t5/prediction.csv --model-dir ./t5
echo ----------***************-----------

echo Task 6
echo ------------------------------------
mkdir ./t6; python train.py --mode classification --train-csv ~/cnt/sdsj2018-automl/data/check_6_c/train.csv --model-dir ./t6
python predict.py --test-csv ~/cnt/sdsj2018-automl/data/check_6_c/test.csv --prediction-csv ./t6/prediction.csv --model-dir ./t6
echo ----------***************-----------

echo Task 7
echo ------------------------------------
mkdir ./t7; python train.py --mode classification --train-csv ~/cnt/sdsj2018-automl/data/check_7_c/train.csv --model-dir ./t7
python predict.py --test-csv ~/cnt/sdsj2018-automl/data/check_7_c/test.csv --prediction-csv ./t7/prediction.csv --model-dir ./t7
echo ----------***************-----------

echo Task 8
echo ------------------------------------
mkdir ./t8; python train.py --mode classification --train-csv ~/cnt/sdsj2018-automl/data/check_8_c/train.csv --model-dir ./t8
python predict.py --test-csv ~/cnt/sdsj2018-automl/data/check_8_c/test.csv --prediction-csv ./t8/prediction.csv --model-dir ./t8
echo ----------***************-----------