from keras.src.layers import Input, Dense, Dropout, Add
from keras.src.layers import BatchNormalization, Activation, Conv1D
from keras.src.models import Model
import numpy as np
from Classificaltion_Evaluation import ClassificationEvaluation


# Deep GCN Block with Residual Connection
def DeepGCN_block(inputs, feature_dims, activation):
    x = BatchNormalization()(inputs)
    x = Activation(activation)(x)
    x = Conv1D(feature_dims, 1, use_bias=False)(x)

    x = BatchNormalization()(x)
    x = Activation(activation)(x)
    x = Conv1D(feature_dims, 1, use_bias=False)(x)

    # Residual connection
    if inputs.shape[-1] == feature_dims:
        x = Add()([inputs, x])

    return x


def Model_Deep_GCN(train_data, Train_Target, test_data, Test_Target, EP=None):
    if EP is None:
        EP = 10

    IMG_SIZE = 255

    # ------------------ Preprocessing ------------------
    Train_X = np.zeros((train_data.shape[0], IMG_SIZE * IMG_SIZE))
    for i in range(train_data.shape[0]):
        temp = np.resize(train_data[i], (IMG_SIZE * IMG_SIZE))
        Train_X[i] = np.reshape(temp, (IMG_SIZE * IMG_SIZE))

    trainX = np.reshape(Train_X, (Train_X.shape[0], 1, Train_X.shape[1]))

    Test_X = np.zeros((test_data.shape[0], IMG_SIZE, IMG_SIZE, 1))
    for i in range(test_data.shape[0]):
        temp = np.resize(test_data[i], (IMG_SIZE * IMG_SIZE, 1))
        Test_X[i] = np.reshape(temp, (IMG_SIZE, IMG_SIZE, 1))

    testX = np.reshape(Test_X, (Test_X.shape[0], 1, Test_X.shape[1]))

    # ------------------ Model ------------------
    inputs = Input(shape=(1, trainX.shape[2]))

    x = Conv1D(64, 1, use_bias=False)(inputs)
    # Stack many DeepGCN blocks
    for _ in range(6):   # You can increase depth (e.g., 10, 20)
        x = DeepGCN_block(x, 64, 'relu')
        x = Dropout(0.3)(x)

    # Output layer
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    outputs = Conv1D(Train_Target.shape[1], 1, activation='softmax')(x)

    # ------------------ Compile ------------------
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