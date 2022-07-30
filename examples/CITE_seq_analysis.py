import numpy as np
import pandas as pd
from scot import sc_multi_omics
import time

def load_data():
    expression_data = np.array(pd.read_csv("data/CITE_seq/expression_data.csv", index_col=0))
    idx = np.argwhere(np.count_nonzero(expression_data, axis=0) <862)
    expression_data = np.delete(expression_data, idx, axis=1)
    expression_data = np.log(expression_data+1)

    protein_data = np.array(pd.read_csv("data/CITE_seq/protein_data.csv", index_col=0))
    cell_stage = np.array(pd.read_csv("data/CITE_seq/cell_type.csv", header=None))[0]
    labels = []
    for each in cell_stage:
        if each == "Unclassified":
            labels.append(0)
        elif each == "B_cell":
            labels.append(1)
        elif each == "CD4_T_cell":
            labels.append(2)
        elif each == "CD8_T_cell":
            labels.append(3)
        elif each == "NK":
            labels.append(4)
        elif each == "Monocytes":
            labels.append(5)
        elif each == "DC":
            labels.append(6)
        elif each == "HSC":
            labels.append(7)

    return expression_data, protein_data, labels


def data_normalization(expression_data, protein_data):

    # min-max normalization
    expression_data = (expression_data - expression_data.min()) / (expression_data.max() - expression_data.min()) + 0.1
    protein_data = (protein_data - protein_data.min()) / (protein_data.max() - protein_data.min()) + 0.1

    expression_data = np.array(expression_data)
    protein_data = np.array(protein_data)

    data = [expression_data, protein_data]

    return data


if __name__ == "__main__":

    start_time = time.time()
    expression_data, protein_data, labels = load_data()
    data = data_normalization(expression_data, protein_data)
    print(data[0].shape)
    print(data[1].shape)

    sc_model = sc_multi_omics(K1=30, K2=30, K3=30)
    predict_data = sc_model.fit_list_complete(data, opt="Adam", dist="gaussian", lr=1e-3, n_epochs=1500, lambda_C_regularizer=0.01, lambda_G_regularizer=0.01, lambda_O_regularizer=[0.01, 0.01], device="cuda:1")
    
    np.savetxt("cell_embeddings.csv", sc_model.C, delimiter = ',')
    np.savetxt("gene_embeddings.csv", sc_model.G[0], delimiter = ',')
    np.savetxt("protein_embeddings.csv", sc_model.G[1], delimiter = ',')
    np.savetxt("omics_embeddings.csv", sc_model.O, delimiter = ',')
    np.savetxt("predict_data_expression.csv", predict_data[0])
    np.savetxt("predict_data_protein.csv", predict_data[1])
    print(time.time() - start_time)
