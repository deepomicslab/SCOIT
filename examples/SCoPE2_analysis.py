import numpy as np
import pandas as pd
from scot import sc_multi_omics
import time

def load_data():
    expression_data = pd.read_csv("data/SCoPE2/expression_data.csv", index_col=0)
    protein_data = pd.read_csv("data/SCoPE2/protein_data.csv", index_col=0)
    cell_stage = np.array(pd.read_csv("data/SCoPE2/cell_stage.csv", header=None))[0]
    labels = []
    for each in cell_stage:
        if each == "sc_m0":
            labels.append(0)
        elif each == "sc_u":
            labels.append(1)

    return expression_data, protein_data, labels


def data_normalization(expression_data, protein_data):

    # min-max normalization
    expression_data = (expression_data - expression_data.min()) / (expression_data.max() - expression_data.min()) + 0.1
    protein_data = (protein_data - protein_data.min()) / (protein_data.max() - protein_data.min()) + 0.1

    expression_data = np.array(expression_data)
    protein_data = np.array(protein_data)
    data = np.array([expression_data, protein_data])
    data[np.isnan(data)] = 0.1

    return data


if __name__ == "__main__":

    start_time = time.time()
    expression_data, protein_data, labels = load_data()
    data = data_normalization(expression_data, protein_data)
    print(data.shape)

    sc_model = sc_multi_omics(K1=30, K2=30, K3=30)
    predict_data = sc_model.fit_complete(data, opt="Adam", dist="gaussian", lr=1e-3, n_epochs=7000, lambda_C_regularizer=0.01, lambda_G_regularizer=0.01, lambda_O_regularizer=[0.01, 0.01], lambda_OC_regularizer=[2, 2], lambda_OG_regularizer=[2, 2]) 
    
    np.savetxt("cell_embeddings.csv", sc_model.C, delimiter = ',')
    np.savetxt("gene_embeddings.csv", sc_model.G, delimiter = ',')
    np.savetxt("omics_embeddings.csv", sc_model.O, delimiter = ',')
    np.savetxt("predict_data_expression.csv", predict_data[0])
    np.savetxt("predict_data_protein.csv", predict_data[1])
    print(time.time() - start_time)
