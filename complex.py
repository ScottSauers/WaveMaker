import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv1D, LSTM, Dense, Input, MaxPooling1D, Flatten, Reshape
from tensorflow.keras.optimizers import Adam
from scipy.io import wavfile
import matplotlib.pyplot as plt

def load_waveform(file_path):
    sample_rate, data = wavfile.read(file_path)
    if data.ndim > 1:  # Convert stereo to mono by averaging channels
        data = np.mean(data, axis=1)
    data = data.astype(np.float32) / np.max(np.abs(data))  # Normalize data
    return data, sample_rate

file_path = '/Users/scott/Downloads/wave/sample.wav'
data, sample_rate = load_waveform(file_path)

sequence_length = 512  # Reduce sequence length for faster training
prediction_length = sequence_length  # Set prediction length equal to sequence length for continuity
batch_size = 8  # Batch size fits the data

def create_sequences(data, sequence_length, prediction_length):
    X, Y = [], []
    for start in range(0, len(data) - sequence_length - prediction_length, prediction_length):
        end = start + sequence_length
        X.append(data[start:end])
        Y.append(data[end:end + prediction_length])
    return np.array(X), np.array(Y)

x_train, y_train = create_sequences(data, sequence_length, prediction_length)

# Ensure the number of samples is divisible by batch size
num_samples = (len(x_train) // batch_size) * batch_size
x_train, y_train = x_train[:num_samples], y_train[:num_samples]

# Define the model using the Input layer for dynamic input shape
input_layer = Input(batch_shape=(batch_size, sequence_length, 1))
conv_layer = Conv1D(filters=32, kernel_size=3, activation='relu')(input_layer)
pool_layer = MaxPooling1D(pool_size=2)(conv_layer)
flatten_layer = Flatten()(pool_layer)
reshaped_layer = Reshape((flatten_layer.shape[1], 1))(flatten_layer)
lstm_layer = LSTM(32, stateful=True, return_sequences=False)(reshaped_layer)
output_layer = Dense(prediction_length, activation='linear')(lstm_layer)

model = Model(inputs=input_layer, outputs=output_layer)
model.compile(optimizer=Adam(learning_rate=0.01), loss='mse')

# Callback to reset states and save predictions/plots
class CustomCallback(tf.keras.callbacks.Callback):
    def __init__(self, x_train, y_train, sample_rate, sequence_length, prediction_length):
        super(CustomCallback, self).__init__()
        self.x_train = x_train
        self.y_train = y_train
        self.sample_rate = sample_rate
        self.sequence_length = sequence_length
        self.prediction_length = prediction_length

    def on_epoch_end(self, epoch, logs=None):
        # Reset states of all LSTM layers
        for layer in self.model.layers:
            if hasattr(layer, 'reset_states'):
                layer.reset_states()
        initial_seq = self.x_train[0].reshape(1, self.sequence_length, 1)
        predicted_waveform = predict_full_waveform(self.model, initial_seq, len(self.y_train.flatten()), self.sequence_length)
        plt.figure(figsize=(10, 4))
        plt.plot(predicted_waveform, label='Predicted Waveform')
        plt.plot(self.y_train.flatten(), label='True Waveform', linestyle='--')
        plt.title(f'Waveform Prediction - Epoch {epoch+1}')
        plt.legend()
        plt.savefig(f'waveform_epoch_{epoch+1}.png')
        plt.close()
        wavfile.write(f'predicted_waveform_epoch_{epoch+1}.wav', self.sample_rate, np.array(predicted_waveform, dtype=np.float32))

# Function to predict the entire waveform based on the trained model
def predict_full_waveform(model, initial_seq, total_length, sequence_length):
    predicted_output = []
    current_input = initial_seq
    while len(predicted_output) < total_length:
        prediction = model.predict(current_input)
        prediction = prediction.reshape(-1)  # Flatten the prediction
        predicted_output.extend(prediction)
        current_input = np.roll(current_input, -sequence_length, axis=1)
        current_input[:, -sequence_length:, 0] = prediction[:sequence_length]
        for layer in model.layers:
            if hasattr(layer, 'reset_states'):
                layer.reset_states()  # Reset state after each prediction sequence
    return predicted_output[:total_length]

# Training loop with state reset after each epoch
model.fit(x_train.reshape(-1, sequence_length, 1), y_train, epochs=10, batch_size=batch_size, shuffle=False, callbacks=[CustomCallback(x_train, y_train, sample_rate, sequence_length, prediction_length)], verbose=1)

# Initial sequence for the full waveform prediction
initial_seq = data[:sequence_length].reshape(1, sequence_length, 1)
predicted_waveform = predict_full_waveform(model, initial_seq, len(data), sequence_length)

# Save the final predicted waveform
wavfile.write('predicted_full_waveform.wav', sample_rate, np.array(predicted_waveform, dtype=np.float32))
