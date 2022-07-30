import numpy as np
import pandas as pd
from scot import sc_multi_omics
import time

def load_data():
    expression_data = pd.read_csv("data/sc_GEM/expression_data.csv", index_col=0)
    methylation_data = pd.read_csv("data/sc_GEM/methylation_data.csv", index_col=0)
    cell_stage = np.array(pd.read_csv("data/sc_GEM/cell_stage.csv", header=None))[0]
    labels = []
    for each in cell_stage:
        if each == "BJ":
            labels.append(0)
        if each == "d8":
            labels.append(1)
        if each == "d16T-" or each == "d16T+":
            labels.append(2)
        if each == "d24T-" or each == "d24T+":
            labels.append(3)
        if each == "IPS":
            labels.append(4)
        if each == "ES":
            labels.append(5)
                       
    return expression_data, methylation_data, labels


def data_normalization(expression_data, methylation_data):

    # min-max normalization
    expression_data = (expression_data - expression_data.min()) / (expression_data.max() - expression_data.min()) + 0.1
    methylation_data = methylation_data + 0.1

    expression_data = np.array(expression_data)
    methylation_data = np.array(methylation_data)

    data = np.array([expression_data, methylation_data])

    return data

if __name__ == "__main__":

    start_time = time.time()
    expression_data, methylation_data, labels = load_data()
    data = data_normalization(expression_data, methylation_data)
    print(data.shape)

    sc_model = sc_multi_omics(K1=30, K2=30, K3=30)
    predict_data = sc_model.fit(data, opt="Adam", dist="negative_bionomial", n_epochs=260, lambda_C_regularizer=0.01, lambda_G_regularizer=0.01, lambda_O_regularizer=[0.01, 0.01], lambda_OC_regularizer=[1, 1], lambda_OG_regularizer=[1, 1], batch_size=256, device="cpu")
    
    np.savetxt("cell_embeddings.csv", sc_model.C, delimiter = ',')
    np.savetxt("gene_embeddings.csv", sc_model.G, delimiter = ',')
    np.savetxt("omics_embeddings.csv", sc_model.O, delimiter = ',')
    np.savetxt("predict_data_expression.csv", predict_data[0])
    np.savetxt("predict_data_methylation.csv", predict_data[1])
    print(time.time() - start_time)
