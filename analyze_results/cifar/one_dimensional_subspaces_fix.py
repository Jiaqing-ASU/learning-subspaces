##
# Author: John Kevin Cava
# Date: May 24, 2023
##
import matplotlib.pyplot as plt
import sys
sys.path.append("viz")
import utils as utils
# from config import fs_helper
# from config import helper
import numpy as np

##
# Gather the Results
##
one_dim_subspace_data = utils.read_csv_files(
    [
        "learning-subspaces-results/cifar/eval-one-dimesnional-subspaces/results.csv",
    ],
    ["curr_acc1", "ensemble_acc", "m0_acc"],
)

ensemble_data = utils.read_csv_files(
    ["learning-subspaces-results/cifar/eval-ensemble/results.csv"],
    ["curr_acc1"],
)

dic_ = {}
for id_ in one_dim_subspace_data:
    acc = float(one_dim_subspace_data[id_]['curr_acc1'])
    alpha0 = float(id_.split('alpha0=')[1].split('+')[0])
    alpha1 = float(id_.split('alpha1=')[1].split('+')[0])
    # print(alpha0,alpha1,acc)
    # break
    if alpha0 not in dic_.keys():
        dic_[alpha0] = [acc]
    else:
        dic_[alpha0].append(acc)

key_list = list(dic_.keys())
# sort keys
key_list.sort()
y = [np.mean(item) for item in [dic_[key] for key in key_list]]
x = key_list

fig = plt.figure()
plt.plot(x,y)
plt.savefig("one_dimensional_subspaces.png")