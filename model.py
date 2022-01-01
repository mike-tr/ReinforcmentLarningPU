import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
import cv2
# Configuration paramaters for the whole setup
seed = 42
gamma = 0.99  # Discount factor for past rewards
epsilon = 1.0  # Epsilon greedy parameter
epsilon_min = 0.1  # Minimum epsilon greedy parameter
epsilon_max = 1.0  # Maximum epsilon greedy parameter
epsilon_interval = (
    epsilon_max - epsilon_min
)  # Rate at which to reduce chance of random action being taken
batch_size = 32  # Size of batch taken from replay buffer
max_steps_per_episode = 10000


def combine_states(states):
    # output = np.zeros((11*2, 11*2, 3), dtype="uint8")
    # output[0:11, 0:11] = states[0]
    # output[0:11, 11:22] = states[1]
    # output[11:22, 11:22] = states[2]
    # output[11:22, 0:11] = states[3]
    return np.concatenate(states, axis=2)


def create_maze_solving_network(image_size=64, num_actions=4):
    model = keras.models.Sequential()
    inputShape = (image_size, image_size, 12)
    model.add(layers.Conv2D(32, 3, strides=2, input_shape=inputShape,
              padding='same', activation='relu'))
    # model.add(layers.MaxPooling2D())
    model.add(layers.Conv2D(64, 3, strides=2,
              activation='relu', padding='same'))
    # model.add(layers.MaxPooling2D())
    model.add(layers.Conv2D(128, 3, strides=1,
              activation='relu', padding='same'))
    # model.add(layers.MaxPooling2D())
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(num_actions, activation='linear'))
    model.compile(loss='mse', optimizer='adam')

    return model


def masked_mse(args):
    y_true, y_pred, mask = args
    loss = (y_true - y_pred)**2
    loss *= mask
    return K.sum(loss, axis=-1)


def add_rl_loss_to_network(model):
    num_actions = model.output.shape[1]
    y_pred = model.output
    y_true = layers.Input(name='y_true', shape=(num_actions,))
    mask = layers.Input(name='mask', shape=(num_actions,))
    loss_out = layers.Lambda(masked_mse, output_shape=(
        1,), name='loss')([y_true, y_pred, mask])
    trainable_model = Model(inputs=[model.input, y_true, mask],
                            outputs=loss_out)
    trainable_model.compile(optimizer=Adam(), loss=lambda yt, yp: yp)
    return trainable_model


def predict_on_model(net_input, model, return_raw):
    o = model.predict(np.array([net_input]))
    if return_raw:
        return o[0]
    return np.argmax(o[0])


def preprocess_image(im, image_size=64):
    im = cv2.resize(im, (image_size, image_size))/255.0

    return im


def transfer_weights_partially(copy_from_model, copy_to_model, learning_rate=0.5):
    target_weights = copy_from_model.get_weights()
    learning_model_weights = copy_to_model.get_weights()

    for i in range(len(target_weights)):
        learning_model_weights[i] = learning_rate * target_weights[i] + \
            (1-learning_rate) * learning_model_weights[i]
    copy_to_model.set_weights(learning_model_weights)
