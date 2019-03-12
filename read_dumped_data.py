import pickle
import matplotlib.pyplot as plt
from operator import add

import numpy

pcfg_dict_fpr = []
pcfg_dict_tpr = []
pcfg_ipv4_fpr = []
pcfg_ipv4_tpr = []
pcfg_dict_num_fpr = []
pcfg_dict_num_tpr = []
pcfg_ipv4_num_fpr = []
pcfg_ipv4_num_tpr = []
pcfg_dict_roc_auc = []
pcfg_ipv4_num_roc_auc = []
pcfg_dict_num_roc_auc = []
pcfg_ipv4_roc_auc = []


def average_list(item_list):
    for i in range(0, len(item_list)):
        item_list[i] = item_list[i]/count


def pad_zeros(list1, list2):
    if len(list1) > len(list2):
        list2.extend([0] * (len(list1) - len(list2)))
        # list1 = list1[:len(list2)]
    else:
        list1.extend([0] * (len(list2) - len(list1)))
        # list2 = list2[:len(list1)]'''
    return list1, list2


with open('/home/ashiq/Pictures/Thesis_data/Thesis_data', 'rb') as f:
    count = 0
    while True:
        try:
            print(count)

            pcfg_dict_fpr, itemlist = pad_zeros(pcfg_dict_fpr, pickle.load(f).tolist())
            pcfg_dict_fpr = list(map(add, pcfg_dict_fpr, itemlist))

            pcfg_dict_tpr, itemlist = pad_zeros(pcfg_dict_tpr, pickle.load(f).tolist())
            pcfg_dict_tpr = list(map(add, pcfg_dict_tpr, itemlist))

            pcfg_dict_roc_auc.append(pickle.load(f).tolist())

            pcfg_ipv4_fpr, itemlist = pad_zeros(pcfg_ipv4_fpr, pickle.load(f).tolist())
            pcfg_ipv4_fpr = list(map(add, pcfg_ipv4_fpr, itemlist))

            pcfg_ipv4_tpr, itemlist = pad_zeros(pcfg_ipv4_tpr, pickle.load(f).tolist())
            pcfg_ipv4_tpr = list(map(add, pcfg_ipv4_tpr, itemlist))

            pcfg_ipv4_roc_auc.append(pickle.load(f).tolist())

            pcfg_ipv4_num_fpr, itemlist = pad_zeros(pcfg_ipv4_num_fpr, pickle.load(f).tolist())
            pcfg_ipv4_num_fpr = list(map(add, pcfg_ipv4_num_fpr, itemlist))

            pcfg_ipv4_num_tpr, itemlist = pad_zeros(pcfg_ipv4_num_tpr, pickle.load(f).tolist())
            pcfg_ipv4_num_tpr = list(map(add, pcfg_ipv4_num_tpr, itemlist))

            pcfg_ipv4_num_roc_auc.append(pickle.load(f).tolist())

            pcfg_dict_num_fpr, itemlist = pad_zeros(pcfg_dict_num_fpr, pickle.load(f).tolist())
            pcfg_dict_num_fpr = list(map(add, pcfg_dict_num_fpr, itemlist))

            pcfg_dict_num_tpr, itemlist = pad_zeros(pcfg_dict_num_tpr, pickle.load(f).tolist())
            pcfg_dict_num_tpr = list(map(add, pcfg_dict_num_tpr, itemlist))

            pcfg_dict_num_roc_auc.append(pickle.load(f).tolist())

            count = count + 1

        except Exception as e:
            print(e)
            break


average_list(pcfg_ipv4_fpr)
average_list(pcfg_ipv4_tpr)
average_list(pcfg_dict_num_tpr)
average_list(pcfg_dict_num_fpr)
average_list(pcfg_dict_fpr)
average_list(pcfg_dict_tpr)
average_list(pcfg_ipv4_num_fpr)
average_list(pcfg_ipv4_num_tpr)
pcfg_dict_num_roc_auc = sum(pcfg_dict_num_roc_auc)/len(pcfg_dict_num_roc_auc)
pcfg_dict_roc_auc = sum(pcfg_dict_roc_auc)/len(pcfg_dict_roc_auc)
pcfg_ipv4_num_roc_auc = sum(pcfg_ipv4_num_roc_auc)/len(pcfg_ipv4_num_roc_auc)
pcfg_ipv4_roc_auc = sum(pcfg_ipv4_roc_auc)/len(pcfg_ipv4_roc_auc)


fig, ax = plt.subplots()
plt.title('ROC curve for PCFG BOTNETs, 20 times, 2000 data')
plt.plot(pcfg_dict_fpr, pcfg_dict_tpr, 'b', label='pcfg_dict, AUC = %0.2f' % pcfg_dict_roc_auc)
plt.plot(pcfg_ipv4_fpr, pcfg_ipv4_tpr, 'g', label='pcfg_ipv4, AUC = %0.2f' % pcfg_ipv4_roc_auc)
plt.plot(pcfg_ipv4_num_fpr, pcfg_ipv4_num_tpr, 'r', label='pcfg_ipv4_num, AUC = %0.2f' % pcfg_ipv4_num_roc_auc)
plt.plot(pcfg_dict_num_fpr, pcfg_dict_num_tpr, 'c', label='pcfg_dict_num, AUC = %0.2f' % pcfg_dict_num_roc_auc)
# common part for all files
plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([-0.1, 1.1])
plt.ylim([-0.1, 1.1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
fig.savefig('/home/ashiq/Pictures/Thesis_image/PCFG_US.eps', format='eps')
plt.show()