__author__ = "Jakob Aungiers"
__copyright__ = "Jakob Aungiers 2018"
__version__ = "2.0.0"
__license__ = "MIT"

import os
import json
import time
import math
import matplotlib.pyplot as plt
from core.data_processor import DataLoader
from core.model import Model


def plot_results(predicted_data, true_data):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    plt.plot(predicted_data, label='Prediction')
    plt.legend()
    plt.show()


def plot_results_multiple(predicted_data, true_data, prediction_len):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
	# Pad the list of predictions to shift it in the graph to it's correct start
    for i, data in enumerate(predicted_data):
        padding = [None for p in range(i * prediction_len)]
        plt.plot(padding + data, label='Prediction')
        plt.legend()
    plt.show()


def main():
    configs = json.load(open('config.json', 'r'))
    if not os.path.exists(configs['model']['save_dir']): os.makedirs(configs['model']['save_dir'])
    # saved_models dir for saving models
    data = DataLoader(
        os.path.join('data', configs['data']['filename']),   # filename for input data - SP500
        configs['data']['train_test_split'],                 # 0.85 split
        configs['data']['columns']                           # columns
    )

    model = Model()   # init -> self.model = Sequential()
    model.build_model(configs)
    x, y = data.get_train_data(
        seq_len=configs['data']['sequence_length'],       # sequence length 50
        normalise=configs['data']['normalise']            # normalize = true
    )


    '''
	# in-memory training
	model.train(
		x,
		y,
		epochs = configs['training']['epochs'],
		batch_size = configs['training']['batch_size'],
		save_dir = configs['model']['save_dir']
	)
	'''
    # out-of memory generative training
    steps_per_epoch = math.ceil((data.len_train - configs['data']['sequence_length']) / configs['training']['batch_size'])
    model.train_generator(
        data_gen=data.generate_train_batch(
            seq_len=configs['data']['sequence_length'], # 50
            batch_size=configs['training']['batch_size'], # 32
            normalise=configs['data']['normalise']        # true
        ),
        epochs=configs['training']['epochs'],            # 2
        batch_size=configs['training']['batch_size'],    #32
        steps_per_epoch=steps_per_epoch,                 # 124
        save_dir=configs['model']['save_dir']            # /saved_models
    )

    x_test, y_test = data.get_test_data(
        seq_len=configs['data']['sequence_length'],      #50
        normalise=configs['data']['normalise']          # true
    )
    #print("x_test: " , x_test)
    #print("y_test: " , y_test)
    #predictions = model.predict_sequences_multiple(x_test, configs['data']['sequence_length'], configs['data']['sequence_length'])
    #predictions = model.predict_sequence_full(x_test, configs['data']['sequence_length'])
    print("Before predictions - x_test:", x_test)

    predictions = model.predict_point_by_point(x_test)
    print("Predictions: ", predictions)
    #plot_results_multiple(predictions, y_test, configs['data']['sequence_length'])
    plot_results(predictions, y_test)


if __name__ == '__main__':
    main()