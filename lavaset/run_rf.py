from rf import StochasticBosque
import numpy as np
import pandas as pd 
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import time
import csv 

# nmr_peaks = pd.read_csv('~/Documents/IBS/NMR_data/IBS_HNMR_data_n267.csv')
# X = np.array(nmr_peaks.iloc[:, 3:])
# y = np.array(nmr_peaks.iloc[:, 1], dtype=np.double)

# cmr = pd.read_csv('~/Documents/cmr_rf/RBHHCM_HC_cmr_1273.csv')
# X = np.array(cmr.iloc[:, :-1])
# y = np.array(cmr.loc[:, 'phen'], dtype=np.double)
# # y = pd.read_csv('~/Documents/cmr_rf/LAVASET/lavaset-cpp/formate-testing/formate_cluster_labels.txt', header=None).iloc[:, 0].to_numpy(dtype=np.double)
# # y = pd.read_excel('ethanol-uracil-testing/simulated_groups.xlsx').iloc[:, 1]
# if np.unique(y).any() != 0:
#     y = np.where(y == 1, 0, 1).astype(np.double)
# nn = knn_calculation(nmr_peaks.columns[3:], 1s0)

# mtbls1 = pd.read_csv('MTBLS1.csv')
# mtbls24 = pd.read_csv('MTBLS24.csv')
vcg_data = pd.read_excel('~/Documents/lavaset_local/PTBDB_MI_VCG_data.xlsx')

X = np.array(vcg_data.iloc[:, 5:])
y = np.array(vcg_data.iloc[:, 3], dtype=np.double) # acuteMyocardialInfarction

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=180)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

st = time.time()

results = []
for i in range(0, 100,5):
    print('random_state', i)
    model = StochasticBosque(ntrees=1000, nvartosample='sqrt', nsamtosample=int(X_train.shape[0]*0.9), oobe=True) # 425taking 1/3 of samples for bootstrapping
    rf = model.fit_sb(X_train, y_train, random_state=i)
    y_preds, votes, oobe = model.predict_sb(X_test, rf)
    accuracy = accuracy_score(y_test, np.array(y_preds, dtype=int))
    precision = precision_score(y_test, np.array(y_preds, dtype=int))
    recall = recall_score(y_test, np.array(y_preds, dtype=int))
    f1 = f1_score(y_test, np.array(y_preds, dtype=int))

    result = {'Random State': i, 'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1 Score': f1, 'oobe': oobe}
    results.append(result)
    # pd.DataFrame(model.feature_evaluation(X_train, rf)).to_csv(f'~/Documents/lavaset_local/vcg_results/classicRF_feature_impo_1000T_VCG_acutemi_rs{i}.csv')

print(results)
fields = ['Random State', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'oobe']
en = time.time()
print('time', en-st)
with open(f'~/Documents/lavaset_local/vcg_results/classicRF_1000T_VCG_acutemi.txt', 'w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=fields)
    writer.writeheader()  # Write header
    writer.writerows(results)  # Write multiple rows
# en = time.time()
# print('time to run 1000 trees', en-st)