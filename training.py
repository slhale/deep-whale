import argparse
import keras
import numpy as np
import time

parser = argparse.ArgumentParser(description='Train a sequential neural '
                                             'network on generated data.',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('-l', '--load-data', type=str, default=None,
            help='File to load data from. Expects an hdf5 file with keys '
                '"training_data", "training_labels", "validation_data", and '
                    '"validation_labels".')

parser.add_argument('-m', '--load-model', type=str, default=None,
            help='File to load neural network model from. Expects a Keras json.')

parser.add_argument('-s', '--save-file', action="store_true",
            help='If you want to save the produced model.')

parser.add_argument('-e', '--epochs', type=int, default=3,
            help='Number of epochs to train with.')
parser.add_argument('-b', '--batch-size', type=int, default=1000,
            help='Size of batches to load.')

args = parser.parse_args()

model_file = args.load_model#'timeseries_feedforward_model.json'

# Load the model
with open(model_file, 'r') as json_file:
        model = keras.models.model_from_json(json_file.read())
        
# Compile the model
model.compile(optimizer='adadelta', loss='categorical_crossentropy',
                      metrics=['accuracy'])


data_file = args.load_data#'whale_training_samples19200train4800val6000test_shape4000.hdf5'

# Define normalizer for loading data
def normalize_long_axis(x):
    # Assuming data format (num_samples, time_series, channels)
    # and we want to normalize over the time series
    return keras.utils.normalize(x, axis=1)

# Load training and validataion data
# Data is normalized upon loading
training_data = keras.utils.io_utils.HDF5Matrix(
        data_file, 'training_data', normalizer=normalize_long_axis)
validation_data = keras.utils.io_utils.HDF5Matrix(
        data_file, 'validation_data', normalizer=normalize_long_axis)
training_labels = keras.utils.io_utils.HDF5Matrix(
        data_file, 'training_labels')
validation_labels = keras.utils.io_utils.HDF5Matrix(
        data_file, 'validation_labels')


keras.backend.set_image_data_format('channels_last')


# Callbacks
callbacks = []

batch_size = args.batch_size#100
epochs = args.epochs#10
# Save training log
callbacks.append(keras.callbacks.CSVLogger("training_log_"+time.strftime("%Y%m%d")+"_"
                                           +str(batch_size)+"batch_"
                                           +str(epochs)+"epochs_"
                                           +model_file[:-5]+".csv"))

# Train
model.fit(training_data, training_labels, epochs=epochs,
          batch_size=batch_size, verbose=2, shuffle='batch',
          validation_data=(validation_data, validation_labels),
          callbacks=[])

# Save the model
if args.save_file:
    model.save("trained_"+time.strftime("%Y%m%d")+"_"
               +str(batch_size)+"batch_"
               +str(epochs)+"epochs_"
               +model_file[:-5]+".h5")


