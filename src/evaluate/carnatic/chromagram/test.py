import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models, layers
from tensorflow.keras.utils import plot_model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import json
import sys, os

sys.path.append(os.path.join('../', '../'))
from utils import load_data

feature, dataset = 'chromagram', 'carnatic'


def prepare_dataset(test_size):
    # load data
    data = load_data(feature, dataset)
    x, y = data['chroma'], data['label']
    x = x.astype('float32') / 255
    # create train/test split
    return train_test_split(x, y, test_size=test_size, shuffle=True, random_state=42)


def plot_history(number, history):
    with open('history'+str(number)+'.json', 'w') as fp:
        json.dump(history.history, fp, indent=4)

    fig, ax = plt.subplots(2, figsize=(10,8))

    # create accuracy subplot
    ax[0].plot(history.history['acc'], label='train accuracy')
    ax[0].plot(history.history['val_acc'], label='test accuracy')
    ax[0].set_ylabel('Accuracy')
    ax[0].legend(loc='lower right')
    ax[0].set_title('Accuracy eval')

    # create error subplot
    ax[1].plot(history.history['loss'], label='train error')
    ax[1].plot(history.history['val_loss'], label='test error')
    ax[1].set_ylabel('Error')
    ax[1].set_xlabel('Epochs')
    ax[1].legend(loc='upper right')
    ax[1].set_title('Error eval')

    plt.savefig('history' + str(number) +  '.png')


def build_model(number, shape, output):
    if number == 1:
        inputs = layers.Input(shape=shape)
        # layer 1
        x = layers.Reshape((shape[0], shape[1], 1))(inputs)
        x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(x)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)
        x = layers.Dropout(0.1)(x)
        # layer 2
        x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)
        x = layers.Dropout(0.1)(x)
        # layer 3
        x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)
        x = layers.Dropout(0.1)(x)
        # last layer
        x = layers.Flatten()(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dropout(0.1)(x)
        x = layers.Dense(256, activation='relu')(x)
        outputs = layers.Dense(output, activation='softmax')(x)
        return keras.Model(inputs=inputs, outputs=outputs)
    elif number == 2:
        inputs = keras.Input(shape=shape)
        z = layers.Reshape((shape[0], shape[1], 1))(inputs)
        # BRNN
        x = layers.MaxPooling2D((1, 2), (1, 2), padding='same')(z)
        x = layers.Embedding(input_dim=shape[0], output_dim=128)(x)
        x = layers.Bidirectional(layers.GRU(64, return_sequences=True))(inputs)
        y = layers.Dropout(0.5)(x)
        y = layers.Bidirectional(layers.GRU(64, return_sequences=True))(y)
        y = layers.Flatten()(y)
        y = layers.Dense(256, activation='relu')(y)

        # Attention (PCNNA)

        # layer 1
        z = layers.Conv2D(16, 3, 1, padding='same', activation='relu')(z)
        z = layers.Dropout(0.2)(z)
        z = layers.BatchNormalization()(z)
        z = layers.MaxPooling2D(2, 2, padding='same')(z)
        # layer 2
        z = layers.Conv2D(32, 3, 1, padding='same', activation='relu')(z)
        z = layers.Dropout(0.2)(z)
        z = layers.BatchNormalization()(z)
        z = layers.MaxPooling2D(2, 2, padding='same')(z)
        # layer 3
        z = layers.Conv2D(64, 3, 1, padding='same', activation='relu')(z)
        z = layers.Dropout(0.2)(z)
        z = layers.BatchNormalization()(z)
        z = layers.MaxPooling2D(2, 1, padding='same')(z)
        # layer 4
        z = layers.Conv2D(128, 3, 1, padding='same', activation='relu')(z)
        z = layers.Dropout(0.2)(z)
        z = layers.BatchNormalization()(z)
        z = layers.MaxPooling2D(4, 4, padding='same')(z)
        # layer 5
        z = layers.Conv2D(64, 3, 1, padding='same', activation='relu')(z)
        z = layers.Dropout(0.2)(z)
        z = layers.BatchNormalization()(z)
        z = layers.MaxPooling2D(4, 4, padding='same')(z)
        # Dense layer
        z = layers.Flatten()(z)
        z = layers.Dense(256, activation='relu', name='second_block')(z)


        # Parallel combining BRNN and RCNNA blocks

        outputs = layers.add([y, z])
        outputs = layers.Dense(output, activation='softmax')(outputs)

        model = keras.Model(inputs=inputs, outputs=outputs)
        return model
    elif number == 3:
        inputs = keras.Input(shape=shape)

        z = layers.Reshape((shape[0], shape[1], 1))(inputs)

        # layer 1
        z = layers.Conv2D(128, 4, padding='same', activation='relu')(z)
        z = layers.Dropout(0.2)(z)
        z = layers.BatchNormalization()(z)
        z = layers.MaxPooling2D(2, 1, padding='same')(z)
        # layer 2
        z = layers.Conv2D(128, 4, padding='same', activation='relu')(z)
        z = layers.Dropout(0.2)(z)
        z = layers.BatchNormalization()(z)
        z = layers.MaxPooling2D(2, 1, padding='same')(z)
        # layer 3
        z = layers.Conv2D(256, 4, padding='same', activation='relu')(z)
        z = layers.Dropout(0.2)(z)
        z = layers.BatchNormalization()(z)
        # Parallel pooling
        x = layers.MaxPooling2D(26, 1, padding='same')(z)
        y = layers.AveragePooling2D(26, 1, padding='same')(z)
        x = layers.Flatten()(x)
        y = layers.Flatten()(y)

        z = layers.add([x, y])
        # Dense layers
        z = layers.Dense(300, activation='relu')(z)
        z = layers.Dense(150, activation='relu')(z)
        outputs = layers.Dense(output, activation='softmax')(z)

        return keras.Model(inputs=inputs, outputs=outputs)
    elif number == 4:
        inputs = keras.Input(shape=shape)

        z = layers.Reshape((shape[0], shape[1], 1))(inputs)

        # layer 1
        first = layers.Conv2D(256, 4, padding='same', activation='relu')(z)
        z = layers.Dropout(0.2)(first)
        z = layers.BatchNormalization()(z)
        z = layers.MaxPooling2D(2, 1, padding='same')(z)
        # layer 2
        z = layers.Conv2D(256, 4, padding='same', activation='relu')(z)
        z = layers.Dropout(0.2)(z)
        z = layers.BatchNormalization()(z)
        z = layers.MaxPooling2D(2, 1, padding='same')(z)
        # layer 3
        z = layers.Conv2D(256, 4, padding='same', activation='relu')(z)
        z = layers.Dropout(0.2)(z)
        z = layers.BatchNormalization()(z)
        # Residual block
        z = layers.add([first, z])
        # Parallel pooling
        x = layers.MaxPooling2D(125, 1, padding='same')(z)
        y = layers.AveragePooling2D(125, 1, padding='same')(z)
        x = layers.Flatten()(x)
        y = layers.Flatten()(y)

        z = layers.add([x, y])
        # Dense layers
        z = layers.Dense(300, activation='relu')(z)
        z = layers.Dense(150, activation='relu')(z)
        outputs = layers.Dense(output, activation='softmax')(z)

        return keras.Model(inputs=inputs, outputs=outputs)
    elif number == 5:
        inputs = keras.Input(shape=shape)

        z = layers.Reshape((shape[0], shape[1], 1))(inputs)
        mobilenet = keras.applications.MobileNetV2(weights=None, input_tensor=z, include_top=False)
        z = mobilenet(z)
        z = layers.GlobalAveragePooling2D()(z)

        outputs = layers.Dense(output, activation='softmax')(z)

        return keras.Model(inputs=inputs, outputs=outputs)
    elif number == 6:
        inputs = keras.Input(shape=shape)

        z = layers.Reshape((shape[0], shape[1], 1))(inputs)
        effnet = keras.applications.EfficientNetB7(weights=None, input_tensor=z, include_top=False)
        z = effnet(z)
        z = layers.GlobalAveragePooling2D()(z)

        outputs = layers.Dense(output, activation='softmax')(z)

        return keras.Model(inputs=inputs, outputs=outputs)
    elif number == 7:
        inputs = keras.Input(shape=shape)
        x = layers.Reshape((shape[0], shape[1], 1))(inputs)
        x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(x)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)
        x = layers.Dropout(0.1)(x)
        # layer 2
        x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)
        x = layers.Dropout(0.1)(x)
        # layer 3
        x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)
        x = layers.Dropout(0.1)(x)
        # last layer
        x = layers.Flatten()(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.1)(x)
        z = layers.Dense(64, activation='relu')(x)

        outputs = layers.Dense(output, activation='softmax')(z)

        return keras.Model(inputs=inputs, outputs=outputs)


def main():
    x_train, x_test, y_train, y_test = prepare_dataset(0.20)
    train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train = train.shuffle(buffer_size=1000).batch(128)
    val = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(128)
    for i in range(7, 8):
        model = build_model(i, (x_train.shape[1], x_train.shape[2]), 40)
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',   metrics=['acc'])
        history = model.fit(train, validation_data=val,
                    epochs=100,
                    #callbacks= [
                    #keras.callbacks.TensorBoard(log_dir='logs',
                    #                            histogram_freq=1,
                    #                            embeddings_freq=1)]
                            )
        plot_history(i, history)
        plot_model(model, show_shapes=True, to_file='model' + str(i) + '.png')


if __name__ == '__main__':
    main()
