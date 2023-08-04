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

mtbls1 = pd.read_csv('~/Documents/lavaset_local/mtbls_results/MTBLS1.csv')
mtbls24 = pd.read_csv('~/Documents/lavaset_local/mtbls_results/MTBLS24.csv')

X = np.array(mtbls1.iloc[:, 1:])
y = np.array(mtbls1.iloc[:, 0], dtype=np.double)

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
for i in range(0,100, 5):
    model = LAVASET(ntrees=10, n_neigh=10, nvartosample='sqrt', nsamtosample=95, oobe=True) # 425taking 1/3 of samples for bootstrapping
    knn = model.knn_calculation(mtbls1.columns[1:])
    lavaset = model.fit_lavaset(X_train, y_train, knn, random_state=i)
    y_preds, votes, oobe = model.predict_lavaset(X_test, lavaset)
    accuracy = accuracy_score(y_test, np.array(y_preds, dtype=int))
    precision = precision_score(y_test, np.array(y_preds, dtype=int))
    recall = recall_score(y_test, np.array(y_preds, dtype=int))
    f1 = f1_score(y_test, np.array(y_preds, dtype=int))
    result = {'parameters': i, 'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1 Score': f1, 'oobe': oobe}
    fields = ['parameters', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'oobe']
    print(result)
    # with open(f'lavaset_metrics_impo_mtbls1_nsamtosample95_1000t.csv', 'a', newline='') as file:
    #     writer = csv.DictWriter(file, fieldnames=fields)
    #     if file.tell() == 0:  # Check if the file is empty
    #         writer.writeheader()  # Write header
    #     writer.writerow(result)

