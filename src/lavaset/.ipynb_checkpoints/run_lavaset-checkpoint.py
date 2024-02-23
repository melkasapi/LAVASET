from lavaset_new import LAVASET 
import numpy as np
import pandas as pd 
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import time
import csv
import sys
import os
from sklearn.model_selection import ParameterGrid


# nmr_peaks = pd.read_csv('~/Documents/IBS/NMR_data/IBS_HNMR_data_n267.csv')
# X = np.array(nmr_peaks.iloc[:, 3:])
# y = np.array(nmr_peaks.iloc[:, 1], dtype=np.double)

# mtbls1 = pd.read_csv('~/Documents/lavaset_local/mtbls_results/MTBLS1.csv')
# mtbls24 = pd.read_csv('~/Documents/lavaset_local/mtbls_results/MTBLS24.csv')

vcg_data = pd.read_excel('~/Documents/lavaset_local/PTBDB_MI_VCG_data.xlsx')

X = np.array(vcg_data.iloc[:, 5:])
y = np.array(vcg_data.iloc[:, 3], dtype=np.double) # acuteMyocardialInfarction

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=180)

# # y = pd.read_csv('~/Documents/cmr_rf/LAVASET/lavaset-cpp/formate-testing/formate_cluster_labels.txt', header=None).iloc[:, 0].to_numpy(dtype=np.double)
# # y = pd.read_excel('ethanol-uracil-testing/simulated_groups.xlsx').iloc[:, 1]
# if np.unique(y).any() != 0:
# #     y = np.where(y == 1, 0, 1).astype(np.double)
# param_grid = {    
#     'n_neigh': [*range(2, 21), 25, 30, 35, 40, 45, 50],
#     'n_trees': [100, 500, 1000, 1500, 2000, 5000],

# }

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=180)

# st = time.time()
# param_combinations = list(ParameterGrid(param_grid))
results=[]
for i in range(0,100, 5):
    model = LAVASET(ntrees=1000, n_neigh=2, nvartosample='sqrt', nsamtosample=int(X_train.shape[0]*0.9), oobe=True) # 425taking 1/3 of samples for bootstrapping
    knn = model.knn_calculation(X)
    print(knn[0].shape)
    sys.exit()
    lavaset = model.fit_lavaset(X_train, y_train, knn, random_state=i)
    y_preds, votes, oobe = model.predict_lavaset(X_test, lavaset)
    accuracy = accuracy_score(y_test, np.array(y_preds, dtype=int))
    precision = precision_score(y_test, np.array(y_preds, dtype=int))
    recall = recall_score(y_test, np.array(y_preds, dtype=int))
    f1 = f1_score(y_test, np.array(y_preds, dtype=int))
    result = {'random_state': i, 'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1 Score': f1, 'oobe': oobe}
    fields = ['random_state', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'oobe']
    # pd.DataFrame(model.feature_evaluation(X_train, lavaset)).to_csv(f'~/Documents/lavaset_local/vcg_results/lavaset_feature_impo_1000T2nn_VCG_acutemi_rs{i}.csv')
    # result = {'Random State': i, 'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1 Score': f1, 'oobe': oobe}
    # results.append(result)
    # with open(f'lavaset_metrics_1000T2nn_VCG_acutemi_rs.txt', 'a', newline='') as file:
    #     writer = csv.DictWriter(file, fieldnames=fields)
    #     if file.tell() == 0:  # Check if the file is empty
    #         writer.writeheader()  # Write header
    #     writer.writerow(result)

