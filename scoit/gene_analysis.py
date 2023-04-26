import numpy as np
import umap
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import scipy
from scipy import stats


def pearson_correlation(feature1_embedding, feature2_embedding):

    return scipy.stats.pearsonr(feature1_embedding, feature2_embedding)[0]


def feature_projection(feature_embedding, cell_embeddings, figure_name="feature_projections.png", umap_epochs=100, dimension=30):
    
    projections = np.dot(cell_embeddings[:, :dimension], feature_embedding[:dimension])
    # plot
    umap_model = umap.UMAP(random_state=123, n_epochs=umap_epochs)
    new_data = umap_model.fit_transform(cell_embeddings)
    plt.figure(figsize=(4, 4))
    plt.scatter(new_data[:, 0], new_data[:, 1], c=projections, marker='o', s=4, cmap="Oranges")
    plt.xlabel("UMAP_1")
    plt.ylabel("UMAP_2")
    plt.tight_layout()
    plt.scatter(new_data[:, 0], new_data[:, 1], c=projections, marker='o', s=4, cmap="Oranges")
    plt.savefig(figure_name, dpi=600)
    plt.close()    

