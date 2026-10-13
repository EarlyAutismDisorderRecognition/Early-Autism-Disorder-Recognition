from keras.src.layers import (Input, Dense, Dropout, Add, LayerNormalization, MultiHeadAttention, Conv1D, Activation, BatchNormalization, Concatenate)
from keras.src.models import Model
import numpy as np
from Classificaltion_Evaluation import ClassificationEvaluation


# ------------------ Deep GCN Block ------------------
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


# ------------------ Transformer Block ------------------
def Transformer_block(x, num_heads, key_dim, ff_dim):
    # Self-attention
    attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)(x, x)
    x = Add()([x, attn_output])
    x = LayerNormalization()(x)

    # Feed Forward Network
    ffn = Dense(ff_dim, activation='relu')(x)
    ffn = Dense(x.shape[-1])(ffn)
    x = Add()([x, ffn])
    x = LayerNormalization()(x)
    return x


# ------------------ Model ------------------
def Model_GCN_dual_transformer(train_data, Train_Target, test_data, Test_Target, EP=None):
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
    # ------------------ Input ------------------
    inputs = Input(shape=(1, trainX.shape[2]))
    # ------------------ Deep GCN Backbone ------------------
    x = Conv1D(64, 1, use_bias=False)(inputs)
    for _ in range(6):
        x = DeepGCN_block(x, 64, 'relu')
        x = Dropout(0.3)(x)
    # ------------------ Dual Transformer ------------------
    # 🔹 Transformer Branch 1 (Local Attention)
    t1 = Transformer_block(x, num_heads=4, key_dim=32, ff_dim=128)
    # 🔹 Transformer Branch 2 (Global Attention - deeper)
    t2 = Transformer_block(x, num_heads=8, key_dim=32, ff_dim=256)
    t2 = Transformer_block(t2, num_heads=8, key_dim=32, ff_dim=256)
    # ------------------ Fusion ------------------
    x = Concatenate()([t1, t2])
    # ------------------ Output ------------------
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    outputs = Conv1D(Train_Target.shape[1], 1, activation='softmax')(x)
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