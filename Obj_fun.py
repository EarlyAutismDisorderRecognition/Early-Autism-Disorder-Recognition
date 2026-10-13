import numpy as np
from ARP import ARP
from Global_vars import Global_vars
from Model_MGAFAsBRN import Model_MGAFAsBRN
from RELIEF import reliefF


def objfun_feat(Soln):
    Feat = Global_vars.Feat
    Tar = Global_vars.Target
    Fitn = np.zeros(Soln.shape[0])
    dimension = len(Soln.shape)
    if dimension == 2:
        for i in range(Soln.shape[0]):
            sol = Soln[i, :]
            Feature = ARP(Feat, sol=sol)
            score = reliefF(np.asarray(Feature), np.ravel(Tar))
            Fitn[i] = 1 / score
        return Fitn
    else:
        sol = Soln
        Feature = ARP(Feat, sol=sol)
        score = reliefF(np.asarray(Feature), np.ravel(Tar))
        Fitn = 1 / score
        return Fitn

def objfun_cls(Soln):
    Feat_1 = Global_vars.Feat_1
    Feat_2 = Global_vars.Feat_2
    Feat_3 = Global_vars.Feat_3
    Tar = Global_vars.Target
    Fitn = np.zeros(Soln.shape[0])
    dimension = len(Soln.shape)
    if dimension == 2:
        for i in range(Soln.shape[0]):
            sol = np.round(Soln[i, :]).astype(np.int16)
            Eval, predict = Model_MGAFAsBRN(Feat_1, Feat_2, Feat_3, Tar, sol=sol)
            Fitn[i] = (1 / (Eval[4] + Eval[13])) + Eval[14]  # (1 / (Accuracy + MCC)) + FOR
        return Fitn
    else:
        sol = np.round(Soln).astype(np.int16)
        Eval, predict = Model_MGAFAsBRN(Feat_1, Feat_2, Feat_3, Tar, sol=sol)
        Fitn = (1 / (Eval[4] + Eval[13])) + Eval[14]  # (1 / (Accuracy + MCC)) + FOR
        return Fitn