import numpy as np
import pandas as pd
from scot import sc_multi_omics
import time

def load_data():

    expression_data = pd.read_csv("data/SNARE_seq_adult_mouse/RNA_pca.csv", index_col=0, na_filter=False).to_numpy()

    ATAC_data = pd.read_csv("data/SNARE_seq_adult_mouse/ATAC_lsi.csv", index_col=0, na_filter=False).to_numpy()[:, 1:]

    labels = np.loadtxt("data/SNARE_seq_adult_mouse/label.txt")
    
    return expression_data, ATAC_data, labels



if __name__ == "__main__":

    start_time = time.time()
    expression_data, ATAC_data, labels = load_data()
    data = [expression_data, ATAC_data]
    print(data[0].shape)
    print(data[1].shape)

    sc_model = sc_multi_omics(K1=20, K2=20, K3=20)
    predict_data = sc_model.fit_list_complete(data, opt="Adam", dist="gaussian", lr=1e-3, n_epochs=600, lambda_C_regularizer=0.01, lambda_G_regularizer=0.01, lambda_O_regularizer=[0.01, 0.01])

    np.savetxt("cell_embeddings.csv", sc_model.C, delimiter = ',')
    np.savetxt("gene_embeddings.csv", sc_model.G[0], delimiter = ',')
    np.savetxt("loc_embeddings.csv", sc_model.G[1], delimiter = ',')
    np.savetxt("omics_embeddings.csv", sc_model.O, delimiter = ',')
    np.savetxt("predict_data_expression.csv", predict_data[0])
    np.savetxt("predict_data_loc.csv", predict_data[1])
    print(time.time() - start_time)
