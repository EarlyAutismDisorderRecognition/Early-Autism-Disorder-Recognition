from keras.src.layers import (Input, Dense, Dropout, Conv1D, MaxPooling1D, Bidirectional, LSTM, Layer, Activation)
from keras.src.models import Model
import keras.src.backend as K
import numpy as np
from Classificaltion_Evaluation import ClassificationEvaluation


# ------------------ Attention Layer ------------------
class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight",
                                 shape=(input_shape[-1], 1),
                                 initializer="normal")
        self.b = self.add_weight(name="att_bias",
                                 shape=(input_shape[1], 1),
                                 initializer="zeros")
        super(AttentionLayer, self).build(input_shape)

    def call(self, x):
        e = K.tanh(K.dot(x, self.W) + self.b)
        a = K.softmax(e, axis=1)
        output = x * a
        return K.sum(output, axis=1)


# ------------------ Model ------------------
def Model_CNN_BiLSTM_attention(train_data, Train_Target, test_data, Test_Target, EP=None):
    if EP is None:
        EP = 10
    IMG_SIZE = 255

    # ------------------ Preprocessing ------------------
    Train_X = np.zeros((train_data.shape[0], IMG_SIZE * IMG_SIZE))
    for i in range(train_data.shape[0]):
        temp = np.resize(train_data[i], (IMG_SIZE * IMG_SIZE))
        Train_X[i] = np.reshape(temp, (IMG_SIZE * IMG_SIZE))

    trainX = np.reshape(Train_X, (Train_X.shape[0], Train_X.shape[1], 1))

    Test_X = np.zeros((test_data.shape[0], IMG_SIZE * IMG_SIZE))
    for i in range(test_data.shape[0]):
        temp = np.resize(test_data[i], (IMG_SIZE * IMG_SIZE))
        Test_X[i] = np.reshape(temp, (IMG_SIZE * IMG_SIZE))

    testX = np.reshape(Test_X, (Test_X.shape[0], Test_X.shape[1], 1))

    # ------------------ Input ------------------
    inputs = Input(shape=(trainX.shape[1], 1))

    # ------------------ CNN ------------------
    x = Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(inputs)
    x = MaxPooling1D(pool_size=2)(x)

    x = Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')(x)
    x = MaxPooling1D(pool_size=2)(x)

    # ------------------ BiLSTM ------------------
    x = Bidirectional(LSTM(64, return_sequences=True))(x)

    # ------------------ Attention ------------------
    x = AttentionLayer()(x)

    # ------------------ Fully Connected ------------------
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)

    outputs = Dense(Train_Target.shape[1], activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(trainX, Train_Target, epochs=EP, batch_size=32)
    pred = model.predict(testX)
    avg = np.mean(pred)
    pred[pred >= avg] = 1
    pred[pred < avg] = 0
    pred = np.asarray(pred).astype('int')
    Eval = ClassificationEvaluation(Test_Target, pred)
    return Eval, pred