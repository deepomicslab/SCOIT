import numpy as np
import pandas as pd
from scoit import sc_multi_omics
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


if __name__ == "__main__":

    start_time = time.time()
    expression_data, protein_data, labels = load_data()
    data = [expression_data, protein_data]
    print(data[0].shape)
    print(data[1].shape)

    sc_model = sc_multi_omics()
    predict_data = sc_model.fit_list_complete(data, dist="gaussian", lr=1e-3, n_epochs=5000)
    
    np.savetxt("cell_embeddings.csv", sc_model.C, delimiter = ',')
    np.savetxt("gene_embeddings.csv", sc_model.G[0], delimiter = ',')
    np.savetxt("protein_embeddings.csv", sc_model.G[1], delimiter = ',')
    np.savetxt("omics_embeddings.csv", sc_model.O, delimiter = ',')
    np.savetxt("predict_data_expression.csv", predict_data[0])
    np.savetxt("predict_data_protein.csv", predict_data[1])
    print(time.time() - start_time)
