import numpy as np
import pandas as pd
from scoit import sc_multi_omics
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


if __name__ == "__main__":

    start_time = time.time()
    expression_data, protein_data, labels = load_data()
    data =np.array([expression_data, protein_data])
    print(data.shape)

    sc_model = sc_multi_omics()
    predict_data = sc_model.fit_complete(data, dist="gaussian", lr=1e-3, n_epochs=5000) 
    
    np.savetxt("cell_embeddings.csv", sc_model.C, delimiter = ',')
    np.savetxt("gene_embeddings.csv", sc_model.G, delimiter = ',')
    np.savetxt("omics_embeddings.csv", sc_model.O, delimiter = ',')
    np.savetxt("predict_data_expression.csv", predict_data[0])
    np.savetxt("predict_data_protein.csv", predict_data[1])
    print(time.time() - start_time)
