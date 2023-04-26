import numpy as np
import pandas as pd
from scoit import sc_multi_omics
import time

def load_data():
    expression_data = np.array(pd.read_csv("data/sc_GEM/expression_data.csv", index_col=0))
    methylation_data = np.array(pd.read_csv("data/sc_GEM/methylation_data.csv", index_col=0))
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



if __name__ == "__main__":

    start_time = time.time()
    expression_data, methylation_data, labels = load_data()
    data = np.array([expression_data, methylation_data])
    print(data.shape)

    sc_model = sc_multi_omics()
    predict_data = sc_model.fit(data, dist="negative_bionomial", n_epochs=1000, device="cpu")
    
    np.savetxt("cell_embeddings.csv", sc_model.C, delimiter = ',')
    np.savetxt("gene_embeddings.csv", sc_model.G, delimiter = ',')
    np.savetxt("omics_embeddings.csv", sc_model.O, delimiter = ',')
    np.savetxt("predict_data_expression.csv", predict_data[0])
    np.savetxt("predict_data_methylation.csv", predict_data[1])
    print(time.time() - start_time)
