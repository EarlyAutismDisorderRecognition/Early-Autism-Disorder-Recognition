import os
import numpy as np
import pandas as pd
import nibabel as nib
import cv2 as cv
from numpy import matlib
from scipy.io import loadmat
import librosa
from sklearn.manifold import TSNE
from ARP import ARP
from BCO import BCO
from BOA import BOA
from GTO import GTO
from Global_vars import Global_vars
from Model_CNN_BiLSTM_attention import Model_CNN_BiLSTM_attention
from Model_Deep_GCN import Model_Deep_GCN
from Model_GCN_dual_transformer import Model_GCN_dual_transformer
from Model_MGAFAsBRN import Model_MGAFAsBRN
from OOA import OOA
from Obj_fun import objfun_feat, objfun_cls
from PROPOSED import PROPOSED
from Plot_results import *


def convert_to_numeric(df, columns):
    for col in columns:
        unique_values = df[col].unique()
        value_map = {val: idx for idx, val in enumerate(unique_values)}
        df[col] = df[col].map(value_map)
    return df


# Read the Dataset
an = 0
if an == 1:
    Dataset = './nilearn_data/ABIDE_pcp/cpac/nofilt_noglobal'
    file = './nilearn_data/ABIDE_pcp/Dataset.txt'
    path = os.listdir(Dataset)  # Directory of the dataset
    IMAGE = []
    Target = []
    uni = []
    Tar = []
    df = pd.read_csv(file, sep=" ")
    Values = df.values
    for t in range(len(Values)):
        print(t, len(Values))
        val = Values[t].astype(str)
        targ = val[0].split('\t')
        Tar.append(targ[2])
    Tar = np.asarray(Tar)
    for i in range(len(path)):
        print(i)
        fold_1 = Dataset + '/' + path[i]
        if '.gz' in fold_1:
            images_fold = nib.load(str(fold_1))
            image_file = images_fold.get_fdata()
            img = image_file[:, :, :, 77]
            for j in range(32, 40):
                print(i, len(path), j, img.shape)
                imge = img[:, :, j]
                imae = np.uint8(imge)
                uni.append(len(np.unique(imae)))
                image = cv.resize(imae, (512, 512))

                IMAGE.append(image)
                Target.append((Tar[i]).astype(int))

    Image = np.asarray(IMAGE)
    Img_Target = (np.asarray(Target)).reshape(-1, 1)

    # Data
    Datas_path = './Datasets/Autism_Data.arff'
    data = pd.read_csv(Datas_path)
    data.head()
    print(list(data.columns))
    columns_to_convert = ['age', 'gender', 'ethnicity', 'jundice', 'austim', 'contry_of_res', 'used_app_before',
                          'age_desc', 'relation', 'Class/ASD']
    Datas = convert_to_numeric(data, columns_to_convert)
    Target = Datas['Class/ASD']
    Datas = Datas.drop('Class/ASD', axis=1)
    Datas = np.asarray(Datas)
    Target = np.asarray(Target)
    text_target = Target.reshape(-1, 1)

    # Audio
    Audio_path = './Datasets/data/'
    adu_dir = os.listdir(Audio_path)
    Audio = []
    len_aud = []
    for i in range(len(adu_dir)):
        Aud_dir = Audio_path + adu_dir[i]
        data = loadmat(Aud_dir)
        audio_signal = data['ISIs']  # <-- key from your image
        # Remove extra dimension (900,1) → (900,)
        audio_signal = np.squeeze(audio_signal)
        Audio.append(audio_signal)
        len_aud.append(len(audio_signal))

    Aud_Data = np.zeros((len(Audio), np.max(len_aud)))
    for i in range(len(Audio)):
        Aud_Data[i, :len(Audio[i])] = Audio[i][:]
    Aud_Data = np.nan_to_num(Aud_Data)
    data_len = np.min((Aud_Data.shape[0], Datas.shape[0], Img_Target.shape[0]))
    # Resize
    Image_reshape = Image[:data_len]
    Img_Target_reshape = Img_Target[:data_len]
    Datas_reshape = Datas[:data_len]
    text_target_reshape = text_target[:data_len]
    Aud_Data_reshape = Aud_Data[:data_len]

    uniq, Count = np.unique(text_target, return_counts=True)

    np.save('Abide_Images.npy', Image_reshape)
    np.save('Abide_Targets.npy', Img_Target_reshape)
    np.save('Abide_Text.npy', Datas_reshape)
    np.save('Abide_Audio.npy', Aud_Data_reshape)


# optimization for Feature
an = 0
if an == 1:
    Feat = np.load('Abide_Images.npy', allow_pickle=True)  # Load the Feat
    Target = np.load('Abide_Targets.npy', allow_pickle=True)  # Load the Target
    Global_vars.Feat = Feat
    Global_vars.Target = Target
    Npop = 10
    Chlen = 3  # Angular range, Total number of cells, Maximum radius
    xmin = matlib.repmat(np.asarray([0, 10, 50]), Npop, 1)
    xmax = matlib.repmat(np.asarray([360, 1000, 150]), Npop, 1)
    fname = objfun_feat
    initsol = np.zeros((Npop, Chlen))
    for p1 in range(initsol.shape[0]):
        for p2 in range(initsol.shape[1]):
            initsol[p1, p2] = np.random.uniform(xmin[p1, p2], xmax[p1, p2])
    Max_iter = 50

    print("GTO...")
    [bestfit1, fitness1, bestsol1, time1] = GTO(initsol, fname, xmin, xmax, Max_iter)  # Gorilla Troops Optimizer

    print("BCO...")
    [bestfit2, fitness2, bestsol2, time2] = BCO(initsol, fname, xmin, xmax, Max_iter)  # Border Collie Optimization

    print("BOA...")
    [bestfit3, fitness3, bestsol3, time3] = BOA(initsol, fname, xmin, xmax, Max_iter)  # Barber Optimization Algorithm

    print("OOA...")
    [bestfit4, fitness4, bestsol4, time4] = OOA(initsol, fname, xmin, xmax, Max_iter)  # Orangutan Optimization Algorithm

    print("PROPOSED...")
    [bestfit5, fitness5, bestsol5, time5] = PROPOSED(initsol, fname, xmin, xmax, Max_iter)  # Improved Orangutan Optimization Algorithm

    BestSol = [bestsol1.squeeze(), bestsol2.squeeze(), bestsol3.squeeze(), bestsol4.squeeze(), bestsol5.squeeze()]
    FITNESS = [fitness1.squeeze(), fitness2.squeeze(), fitness3.squeeze(), fitness4.squeeze(), fitness5.squeeze()]

    np.save('BestSol_Feat.npy', np.asarray(BestSol))  # Best sol
    np.save('Fitness_Feat.npy', np.asarray(FITNESS))  # Fitness


# Feature Extraction
# Angular Radial Partitioning (ARP)
an = 0
if an == 1:
    Images = np.load('Abide_Images.npy', allow_pickle=True) # Load the Images
    BestSol = np.load('BestSol_Feat.npy', allow_pickle=True)  # Load the BestSolution
    Feature = ARP(Images, sol=BestSol[-1, :])
    np.save('ARP_feature.npy', Feature)


# 1. Mel-Frequency Cepstral Coefficients (MFCC)
an = 0
if an == 1:
    Audios = np.load('Abide_Audio.npy', allow_pickle=True)
    spectral = []
    for i in range(len(Audios)):
        print(i, len(Audios))
        Audio = Audios[i]
        mfccs = librosa.feature.mfcc(y=Audio, sr=1000, n_mfcc=128)
        spectral.append(mfccs.flatten())
    spectral = np.asarray(spectral)
    np.save('MFCC_feature.npy', spectral)


# t-distributed Stochastic Neighbor Embedding (Tsne) Feature
an = 0
if an == 1:
    data = np.load('Abide_Text.npy', allow_pickle=True)  # Load the Dataset
    tsne = TSNE(n_components=3)
    tsne_feature = tsne.fit_transform(data)
    np.save('TSNE_feature.npy', tsne_feature)  # Save the Feat 3


# Optimization for Weighted Feature Extraction
an = 0
if an == 1:
    Feat_1 = np.load('ARP_feature.npy', allow_pickle=True)  # Load the Feat 1
    Feat_2 = np.load('MFCC_feature.npy', allow_pickle=True)  # Load the Feat 2
    Feat_3 = np.load('TSNE_feature.npy', allow_pickle=True)  # Load the Feat 3
    Target = np.load('Target.npy', allow_pickle=True)
    Global_vars.Feat_1 = Feat_1
    Global_vars.Feat_2 = Feat_2
    Global_vars.Feat_3 = Feat_3
    Global_vars.Target = Target
    Npop = 10
    Chlen = 3  # Hidden Neuron Count, Learning Rate, No of Epochs from sigGate Bi-RNN Network
    xmin = matlib.repmat(np.asarray([5, 0.01, 5]), Npop, 1)
    xmax = matlib.repmat(np.asarray([255, 0.99, 50]), Npop, 1)
    fname = objfun_cls
    initsol = np.zeros((Npop, Chlen))
    for p1 in range(initsol.shape[0]):
        for p2 in range(initsol.shape[1]):
            initsol[p1, p2] = np.random.uniform(xmin[p1, p2], xmax[p1, p2])
    Max_iter = 50

    print("GTO...")
    [bestfit1, fitness1, bestsol1, time1] = GTO(initsol, fname, xmin, xmax, Max_iter)  # Gorilla Troops Optimizer

    print("BCO...")
    [bestfit2, fitness2, bestsol2, time2] = BCO(initsol, fname, xmin, xmax, Max_iter)  # Border Collie Optimization

    print("BOA...")
    [bestfit3, fitness3, bestsol3, time3] = BOA(initsol, fname, xmin, xmax, Max_iter)  # Barber Optimization Algorithm

    print("OOA...")
    [bestfit4, fitness4, bestsol4, time4] = OOA(initsol, fname, xmin, xmax, Max_iter)  # Orangutan Optimization Algorithm

    print("PROPOSED...")
    [bestfit5, fitness5, bestsol5, time5] = PROPOSED(initsol, fname, xmin, xmax, Max_iter)  # Improved Orangutan Optimization Algorithm

    BestSol = [bestsol1.squeeze(), bestsol2.squeeze(), bestsol3.squeeze(), bestsol4.squeeze(), bestsol5.squeeze()]
    FITNESS = [fitness1.squeeze(), fitness2.squeeze(), fitness3.squeeze(), fitness4.squeeze(), fitness5.squeeze()]

    np.save('BestSol.npy', np.asarray(BestSol))  # Best solution
    np.save('Fitness.npy', np.asarray(FITNESS))  # Fitness


# Classification
an = 0
if an == 1:
    Feat_1 = np.load('ARP_feature.npy', allow_pickle=True)  # Load the Feat 1
    Feat_2 = np.load('MFCC_feature.npy', allow_pickle=True)  # Load the Feat 2
    Feat_3 = np.load('TSNE_feature.npy', allow_pickle=True)  # Load the Feat 3
    Target = np.load('Targets.npy', allow_pickle=True)  # Load the Target
    BestSol = np.load('BestSol.npy', allow_pickle=True)  # Load the BestSolution
    Feat = np.concatenate((Feat_1, Feat_2, Feat_3), axis=1)
    k_fold = 5
    Per = 1 / k_fold
    EVAL = []
    Perc = round(Feat.shape[0] * Per)
    for i in range(k_fold):
        Test_Data = Feat[i * Perc: ((i + 1) * Perc), :]
        Test_Target = Target[i * Perc: ((i + 1) * Perc), :]
        test_index = np.arange(i * Perc, ((i + 1) * Perc))
        total_index = np.arange(Feat.shape[0])
        train_index = np.setdiff1d(total_index, test_index)
        Train_Data = Feat[train_index, :]
        Train_Target = Target[train_index, :]
        Eval = np.zeros((10, 25))
        for j in range(BestSol.shape[0]):
            print(j)
            sol = BestSol[j, :]
            Eval[j, :], pred0 = Model_MGAFAsBRN(Feat_1, Feat_2, Feat_3, Target, sol=sol)
        Eval[5, :], pred1 = Model_Deep_GCN(Train_Data, Train_Target, Test_Data, Test_Target)
        Eval[6, :], pred2 = Model_GCN_dual_transformer(Train_Data, Train_Target, Test_Data, Test_Target)
        Eval[7, :], pred3 = Model_CNN_BiLSTM_attention(Train_Data, Train_Target, Test_Data, Test_Target)
        Eval[8, :], pred4 = Model_MGAFAsBRN(Feat_1, Feat_2, Feat_3, Target)
        Eval[9, :] = Eval[4, :]
        EVAL.append(Eval)
    np.save('Eval_ALL_Fold.npy', np.asarray(EVAL))

plot_conv()
plot_conv_Feat()
ROC_curve()
Plot_Confusion()
Plot_Activation()
Plot_Kfold()
Plot_Step_per_epoch()


