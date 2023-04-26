import numpy as np
import pandas as pd
from scoit import sc_multi_omics
import time

def load_data():
    expression_data = np.loadtxt("data/scNMT/expression_data_300.csv")
    promoter_methy_data = np.loadtxt("data/scNMT/promoter_methy_data_300.csv")
    promoter_acc_data = np.loadtxt("data/scNMT/promoter_acc_data_300.csv")

    cell_stage = np.array(pd.read_csv("data/scNMT/cell_stage.csv", header=None))

    labels = []
    for each in cell_stage:
        if each == "E5.5":
            labels.append(0)
        if each == "E6.5":
            labels.append(1)
        if each == "E7.5":
            labels.append(2)
    labels = np.array(labels)


    return expression_data, promoter_methy_data, promoter_acc_data, labels


if __name__ == "__main__":

    start_time = time.time()
    expression_data, promoter_methy_data, promoter_acc_data, labels = load_data()
    data = [expression_data, promoter_methy_data, promoter_acc_data]
    print(data[0].shape)
    print(data[1].shape)
    print(data[2].shape)

    sc_model = sc_multi_omics()
    predict_data = sc_model.fit_list(data, normalization=False, dist="gaussian", lr=1e-3, n_epochs=1000)
    
    np.savetxt("cell_embeddings.csv", sc_model.C, delimiter = ',')
    np.savetxt("predict_data_expression.csv", predict_data[0])
    np.savetxt("predict_data_promoter_methy.csv", predict_data[1])
    np.savetxt("predict_data_promoter_acc.csv", predict_data[2])
    print(time.time() - start_time)
