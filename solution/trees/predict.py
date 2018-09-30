import argparse
import os
import pandas as pd
import pickle
import time

from utils import transform_data, make_predictions, root_mean_squared_error, pprint, predict

# use this to stop the algorithm before time limit exceeds
TIME_LIMIT = int(os.environ.get('TIME_LIMIT', 5*60))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-csv', type=argparse.FileType('r'), required=True)
    parser.add_argument('--prediction-csv', type=argparse.FileType('w'), required=True)
    parser.add_argument('--model-dir', required=True)
    args = parser.parse_args()

    start_time = time.time()

    # load model
    pipeline_path = os.path.join(args.model_dir, 'pipeline.pkl')
    with open(pipeline_path, 'rb') as fin:
        pipeline = pickle.load(fin)

    # read dataset
    df = pd.read_csv(args.test_csv)
    print(f'Dataset read, shape {df.shape}')

    df['prediction'] = predict(df, pipeline)
    df[['line_id', 'prediction']].to_csv(args.prediction_csv, index=False)

    print('Prediction time: {}'.format(time.time() - start_time))
