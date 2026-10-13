from keras.src.layers import (Input, Dense, Dropout, Bidirectional, LSTM,
                              Layer, Multiply, Concatenate,
                              GlobalAveragePooling1D)
from keras.src.models import Model
import keras.src.backend as K
import numpy as np
from keras.src.optimizers import Adam

from Classificaltion_Evaluation import ClassificationEvaluation
from sklearn.model_selection import train_test_split


# ------------------ Multi-Head Graph Attention ------------------
class MultiHeadGraphAttention(Layer):
    def __init__(self, num_heads=4, output_dim=32, **kwargs):
        super(MultiHeadGraphAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.output_dim = output_dim

    def build(self, input_shape):
        self.W = []
        self.a = []

        for i in range(self.num_heads):
            self.W.append(self.add_weight(shape=(input_shape[-1], self.output_dim),
                                          initializer='glorot_uniform',
                                          name=f'W_{i}'))
            self.a.append(self.add_weight(shape=(2 * self.output_dim, 1),
                                          initializer='glorot_uniform',
                                          name=f'a_{i}'))
        super().build(input_shape)

    def call(self, X):
        heads = []

        for i in range(self.num_heads):
            Wh = K.dot(X, self.W[i])

            # Attention scores
            e = K.tanh(K.dot(K.concatenate([Wh, Wh], axis=-1), self.a[i]))
            alpha = K.softmax(e, axis=1)

            h = alpha * Wh
            heads.append(h)

        return Concatenate()(heads)


# ------------------ sigGate ------------------
class SigGate(Layer):
    def build(self, input_shape):
        self.W = self.add_weight(shape=(input_shape[-1], input_shape[-1]),
                                 initializer='glorot_uniform')
        self.b = self.add_weight(shape=(input_shape[-1],),
                                 initializer='zeros')
        super().build(input_shape)

    def call(self, x):
        gate = K.sigmoid(K.dot(x, self.W) + self.b)
        return Multiply()([x, gate])


# ------------------ Preprocessing ------------------
def preprocess(data, IMG_SIZE=255):
    X = np.zeros((data.shape[0], IMG_SIZE * IMG_SIZE))
    for i in range(data.shape[0]):
        temp = np.resize(data[i], (IMG_SIZE * IMG_SIZE))
        X[i] = temp
    return np.reshape(X, (X.shape[0], X.shape[1], 1))


# ------------------ Model ------------------
def Model_MGAFAsBRN(Feat_1, Feat_2, Feat_3, Target, sol=None, EP=None):

    if sol is None:
        sol = [5, 0.01, 5]
    if EP is None:
        EP = int(sol[1])
    train_f1, test_f1, train_f2, test_f2, train_f3, test_f3, Train_Target, Test_Target = train_test_split(Feat_1,
                                                                                                          Feat_2,
                                                                                                          Feat_3,
                                                                                                          Target,
                                                                                                          test_size=0.25,
                                                                                                          random_state=42)

    # ------------------ Prepare Data ------------------
    trainX1 = preprocess(train_f1)
    trainX2 = preprocess(train_f2)
    trainX3 = preprocess(train_f3)

    testX1 = preprocess(test_f1)
    testX2 = preprocess(test_f2)
    testX3 = preprocess(test_f3)

    # ------------------ Inputs ------------------
    input1 = Input(shape=(trainX1.shape[1], 1))
    input2 = Input(shape=(trainX2.shape[1], 1))
    input3 = Input(shape=(trainX3.shape[1], 1))

    # ------------------ Feature Fusion ------------------
    merged = Concatenate(axis=1)([input1, input2, input3])

    # ------------------ Multihead Graph Attention ------------------
    x = MultiHeadGraphAttention(num_heads=4, output_dim=32)(merged)

    # ------------------ Fusion Dense ------------------
    x = Dense(int(sol[0]), activation='relu')(x)

    # ------------------ Bi-RNN ------------------
    x = Bidirectional(LSTM(64, return_sequences=True))(x)

    # ------------------ sigGate ------------------
    x = SigGate()(x)

    # ------------------ Global Aggregation ------------------
    x = GlobalAveragePooling1D()(x)

    # ------------------ Classification ------------------
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)

    outputs = Dense(Train_Target.shape[1], activation='softmax')(x)

    # ------------------ Model ------------------
    model = Model(inputs=[input1, input2, input3], outputs=outputs)

    model.compile(optimizer=Adam(learning_rate=sol[1]), loss='categorical_crossentropy', metrics=['accuracy'])

    # ------------------ Train ------------------
    model.fit([trainX1, trainX2, trainX3], Train_Target, epochs=EP, batch_size=32)
    pred = model.predict([testX1, testX2, testX3])

    avg = np.mean(pred)
    pred[pred >= avg] = 1
    pred[pred < avg] = 0
    pred = pred.astype('int')
    Eval = ClassificationEvaluation(Test_Target, pred)
    return Eval, pred
