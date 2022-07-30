# SCOIT
SCOIT is an implementation of a probabilistic tensor decomposition framework for single-cell multi-omics data integration. SCOIT accepts the input of datasets from multiple omics, with missing values allowed.

![image](https://github.com/deepomicslab/SCOIT/blob/main/framework.png)

# Getting started

## Prerequisite
+ numpy
+ pytorch 1.9.0

## Install
```
pip install SCOIT
```

## Examples
This is an example of multiple datasets when features have corresponding information.
```Python
from scoit import sc_multi_omics

data = np.array([expression_data, methylation_data])
sc_model = sc_multi_omics()
predict_data = sc_model.fit(data) # the imputed data
np.savetxt("global_cell_embeddings.csv", sc_model.C, delimiter = ',') # global cell embeddings
np.savetxt("global_gene_embeddings.csv", sc_model.G, delimiter = ',') # global gene embeddings
np.savetxt("local_cell_embeddings.csv", sc_model.C, delimiter = ',') # omics-specific cell embeddings
np.savetxt("local_gene_embeddings.csv", sc_model.G, delimiter = ',') # omics-specific gene embeddings

# imputation
imputed_expression_data = predict_data[0]
imputed_methylation_data = predict_data[1]
```
When the features of different omics do not have corresponding information, please use the ```fit_list``` function, which accepts the input as a list of matrices.
```Python
from scoit import sc_multi_omics

data = [expression_data, protein_data]
sc_model = sc_multi_omics()
predict_data = sc_model.fit_list(data)
```
If the input does not contain missing values ("NA"), we provide ```fit_complete``` and ```fit_list_complete``` functions to accelerate the optimization since they take advantage of matrix operations.
```Python
from scoit import sc_multi_omics

data = np.array([expression_data, methylation_data])
sc_model = sc_multi_omics()
predict_data = sc_model.fit_complete(data) # the imputed data
```
```Python
from scoit import sc_multi_omics

data = [expression_data, protein_data]
sc_model = sc_multi_omics()
predict_data = sc_model.fit_list_complete(data)
```

We put the complete scripts for the analysis described in the manuscript under ```examples/``` directory for detailed usage examples and reproduction. The example data can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1F_WBwNsHggjTqgFfTm6IugNKpb0xJTje?usp=sharing).

## Parameters
###  ```sc_multi_omics```
> + ```K1```: The local element-wise product parameter, see the manuscript for details (default=20).
> + ```K2```: The local element-wise product parameter (default=20).
> + ```K3```: The local element-wise product parameter (default=20).
> + ```random_seed```: The random seed used in optimization (default=111).

###  ```fit```
> + ```opt```: The optimization algorithm for gradient descent, including SGD, Adam, Adadelta, Adagrad, AdamW, SparseAdam, Adamax, ASGD, LBFGS (default="Adam").
> + ```dist```:The distribution used for modeling, including gaussian, poisson, negative_bionomial (default="gaussian").
> + ```lr```: The learning rate for gradient descent (default=1e-2).
> + ```n_epochs```: The number of optimization epochs (default=1000).
> + ```lambda_C_regularizer```: The coefficient for the penalty term of global cell embeddings (default=0.01).
> + ```lambda_G_regularizer```: The coefficient for the penalty term of global gene embeddings (default=0.01).
> + ```lambda_O_regularizer```: The coefficient list for the penalty term of global omics embeddings; the length of the list should be the same with the number of omics (default=[0.01, 0.01]).
> + ```lambda_OC_regularizer```: The coefficient list for the penalty term of omics-specific cell embeddings; the length of the list should be the same with the number of omics, not avaiable for complete functions (default=[1, 1]).
> + ```lambda_OG_regularizer```: The coefficient list for the penalty term of omics-specific gene embeddings, the length of the list should be the same with the number of omics, not avaiable for list functions (default=[1, 1]).
> + ```batch_size```: The batch size used for gradient descent, not avaiable for complete functions (default=1000).
> + ```device```: CPU or GPU (default='cuda' if torch.cuda.is_available() else 'cpu').
> + ```verbose```: Whether to print loss for each epoch (default=True).

### Maintainer
WANG Ruohan ruohawang2-c@my.cityu.edu.hk


