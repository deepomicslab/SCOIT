import numpy as np
import pandas as pd
from scot import sc_multi_omics
import time

def load_data():
    expression_data = pd.read_csv("data/PEA_STA/expression_data.csv", index_col=0)
    protein_data = pd.read_csv("data/PEA_STA/protein_data.csv", index_col=0)
    cell_stage = np.array(pd.read_csv("data/PEA_STA/cell_stage.csv", header=None))[0]
    labels = []
    for each in cell_stage:
        day = each.split("_")[1]
        treat = each.split("_")[2]
        if day == "0h":
            labels.append(0)
        elif day == "6d":
            if treat == "contol":
                labels.append(1)
            elif treat == "BMP4":
                labels.append(2)

    return expression_data, protein_data, labels


def data_normalization(expression_data, protein_data):

    # min-max normalization
    expression_data = (expression_data - expression_data.min()) / (expression_data.max() - expression_data.min()) + 0.1
    protein_data = (protein_data - protein_data.min()) / (protein_data.max() - protein_data.min()) + 0.1

    expression_data = np.array(expression_data)
    protein_data = np.array(protein_data)

    data = np.array([expression_data, protein_data])

    return data


if __name__ == "__main__":

    start_time = time.time()
    expression_data, protein_data, labels = load_data()
    data = data_normalization(expression_data, protein_data)
    print(data.shape)

    sc_model = sc_multi_omics(K1=10, K2=10, K3=10)
    predict_data = sc_model.fit(data, opt="Adam", dist="gaussian", n_epochs=850, lambda_C_regularizer=0.01, lambda_G_regularizer=1, lambda_O_regularizer=[1, 1], lambda_OC_regularizer=[1, 1], lambda_OG_regularizer=[0.01, 0.01], batch_size=256, device="cpu")
    
    np.savetxt("cell_embeddings.csv", sc_model.C, delimiter = ',')
    np.savetxt("local_gene_embeddings.csv", sc_model.OG, delimiter = ',')
    np.savetxt("predict_data_expression.csv", predict_data[0])
    np.savetxt("predict_data_protein.csv", predict_data[1])
    print(time.time() - start_time)
