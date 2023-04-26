import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, Adam
from torch.utils.data import Dataset, DataLoader, TensorDataset
from scoit import matrix_model, matrix_model_complete, matrix_list_model, matrix_list_model_complete

def min_max_normalization(data, zero_impute=True):
    contain_na = False
    for omics_data in data:
        if np.isnan(omics_data).any():
            contain_na = True
    # min-max normalization
    if contain_na:
        new_data = []
        omics_min = []
        omics_max = []
        for omics_data in data:
            omics_data = pd.DataFrame(omics_data)
            min_ = omics_data.min()
            max_ = omics_data.max()
            omics_data = (omics_data - min_) / (max_ - min_) + 0.1
            omics_data = np.array(omics_data)
        
            if zero_impute:
                omics_data[np.isnan(omics_data)] = 0.1
            
            new_data.append(omics_data)
            omics_min.append(np.array(min_))
            omics_max.append(np.array(max_))
    else:
        new_data = []
        omics_min = []
        omics_max = []
        for omics_data in data:
            min_ = omics_data.min()
            max_ = omics_data.max()
            omics_data = (omics_data - min_) / (max_ - min_) + 0.1
        
            new_data.append(np.array(omics_data))
            omics_min.append([min_])
            omics_max.append([max_])

    return new_data, omics_min, omics_max


class sc_multi_omics:
    def __init__(self, K1=30, K2=30, K3=30, random_seed=123):
        self.K1 = K1
        self.K2 = K2
        self.K3 = K3
        self.random_seed = random_seed

    def cal_corr(self, data):
        if data.shape[0] == 1:
            pearsonr = 0
        else:
            choose_col = (~np.isnan(data[0]).any(axis=0)) & (~np.isnan(data[1]).any(axis=0))
            for i in range(2, data.shape[0]):
                choose_col = choose_col & (~np.isnan(data[i]).any(axis=0))
            if (np.sum(choose_col)) == 0:
                print("There is no overlapping genes detected in the dataset.")
                pearsonr = 0
            else:
                pearsonr = np.min(np.corrcoef(data[:, :, choose_col].reshape((data.shape[0], -1))))
        return abs(pearsonr)

    def KNN_impute(self, matrix):
        data = np.hstack((matrix))
        imputer = KNNImputer()
        KNN_impute_data = imputer.fit_transform(data)
        if data.shape != KNN_impute_data.shape:
            print("Skip KNN preimpute because of the all-zero features.")    
            
            return matrix

        split_list = [0]
        for each in matrix:
            split_list.append(each.shape[1] + split_list[-1])
        matrix = np.split(KNN_impute_data, split_list[1:-1], axis=1) 
        
        return matrix
    
    def fit(self, matrix, normalization=True, pre_impute=False, opt="Adam", dist="gaussian", lr=1e-2, n_epochs=1000, lambda_C_regularizer=0, lambda_G_regularizer=0, lambda_O_regularizer=[0, 0], lambda_OC_regularizer=[0, 0], lambda_OG_regularizer=[0, 0], batch_size=256, earlystopping=True, device='cuda' if torch.cuda.is_available() else 'cpu', verbose=True):
        # tune the coefficient
        if (lambda_C_regularizer==0) or (lambda_G_regularizer==0) or (0 in lambda_O_regularizer) or (0 in lambda_OC_regularizer) or (0 in lambda_OG_regularizer):
            print("Automatically tune the coefficients for the penalty terms.")
            if self.cal_corr(matrix) > 0.4:
                print("High correlation between the omics detected.")
                lambda_C_regularizer = 0.01
                lambda_G_regularizer = 0.01
                lambda_O_regularizer = [0.01] * matrix.shape[0]
                lambda_OC_regularizer = [1] * matrix.shape[0]
                lambda_OG_regularizer = [1] * matrix.shape[0]
            else:
                print("Low correlation between the omics detected.")
                lambda_C_regularizer = 0.01
                lambda_G_regularizer = 1
                lambda_O_regularizer = [1] * matrix.shape[0]
                lambda_OC_regularizer = [1] * matrix.shape[0]
                lambda_OG_regularizer = [0.01] * matrix.shape[0]
        
        # min-max normizaltion 
        if normalization:
            matrix, omics_min, omics_max = min_max_normalization(data=matrix, zero_impute=False)
            matrix = np.array(matrix)
        
        # KNN preimpute
        if pre_impute:
            matrix = self.KNN_impute(matrix)
            matrix = np.array(matrix)

        # Gradient descent
        if earlystopping:
            self.threshold = (np.log10(lr * 1e3) * 3 + 1.39) * 1e-5
            loss_list = [1e8]
            loss_change_list = [1e8]
        
        self.L, self.N, self.M = matrix.shape
        self.matrix = torch.from_numpy(matrix).float().to(device)
        index = []
        for l in range(self.matrix.shape[0]):
            for n in range(self.matrix.shape[1]):
                for m in range(self.matrix.shape[2]):
                    if not torch.isnan(self.matrix[l, n, m]):
                        index.append([l, n, m])
        index = torch.from_numpy(np.array(index)).long()
        lambda_O_regularizer = torch.from_numpy(np.array(lambda_O_regularizer).reshape((len(lambda_O_regularizer), 1))).float().to(device)
        lambda_OC_regularizer = torch.from_numpy(np.array(lambda_OC_regularizer).reshape((len(lambda_OC_regularizer), 1))).float().to(device)
        lambda_OG_regularizer = torch.from_numpy(np.array(lambda_OG_regularizer).reshape((len(lambda_OG_regularizer), 1))).float().to(device)
        dataset = TensorDataset(index)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        model = matrix_model(self.K1, self.K2, self.K3, self.L, self.N, self.M, self.random_seed).to(device)
        optimizer = eval(opt)(model.parameters(), lr)
        for epoch in range(n_epochs):
            running_loss = 0.0
            for i, (x,) in enumerate(dataloader):
                matrix_hat = model()
                loss = eval("model." + dist + "_loss")(self.matrix, matrix_hat, x.T.numpy(), lambda_C_regularizer, lambda_G_regularizer, lambda_O_regularizer, lambda_OC_regularizer, lambda_OG_regularizer)
                running_loss += loss.detach().item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            if verbose:
                print("Epoch {}| Loss: {:.4f}".format(epoch, running_loss))
            
            # detecting earlystoping:
            if earlystopping:
                loss_list.append(running_loss)
                loss_change_list.append(abs(loss_list[-2]-loss_list[-1])/loss_list[-1])
                if loss_change_list[-1] < self.threshold and loss_change_list[-2] < self.threshold:
                    print("Early stop!")
                    break
        self.C = np.hstack((model.C1.cpu().detach().numpy(), model.C2.cpu().detach().numpy()))
        self.G = np.hstack((model.G1.cpu().detach().numpy(), model.G2.cpu().detach().numpy()))
        self.O = np.hstack((model.O1.cpu().detach().numpy(), model.O2.cpu().detach().numpy()))
        self.OC = model.OC.cpu().detach().numpy()
        self.OG = model.OG.cpu().detach().numpy()
        
        matrix_hat = matrix_hat.cpu().detach().numpy()
        if dist == "poisson" or dist == "negative_bionomial":
            matrix_hat = np.exp(matrix_hat)

        if normalization:
            matrix_hat = matrix_hat - 0.1
            for i in range(len(matrix_hat)):
                if np.isnan(omics_max[i]).any() or np.isnan(omics_min[i]).any():
                    imputer = KNNImputer()
                    omics_max[i] = imputer.fit_transform(np.vstack((matrix_hat[i], omics_max[i].reshape((1, -1)))).T)[:, -1]
                    omics_min[i] = imputer.fit_transform(np.vstack((matrix_hat[i], omics_min[i].reshape((1, -1)))).T)[:, -1]
                matrix_hat[i] = matrix_hat[i] * (omics_max[i] - omics_min[i]) + omics_min[i]

        return matrix_hat
    
    
    def fit_complete(self, matrix, normalization=True, pre_impute=True, opt="Adam", dist="gaussian", lr=1e-2, n_epochs=1000, lambda_C_regularizer=0, lambda_G_regularizer=0, lambda_O_regularizer=[0, 0], lambda_OC_regularizer=[0, 0], lambda_OG_regularizer=[0, 0], batch_size=256, earlystopping=True, device='cuda' if torch.cuda.is_available() else 'cpu', verbose=True):
        # tune the coefficent
        if (lambda_C_regularizer==0) or (lambda_G_regularizer==0) or (0 in lambda_O_regularizer) or (0 in lambda_OC_regularizer) or (0 in lambda_OG_regularizer):
            print("Automatically tune the coefficients for the penalty terms.")
            if np.min(self.cal_corr(matrix)) > 0.4:
                print("High correlation between the omics detected.")
                lambda_C_regularizer = 0.01
                lambda_G_regularizer = 0.01
                lambda_O_regularizer = [0.01] * matrix.shape[0]
                lambda_OC_regularizer = [1] * matrix.shape[0]
                lambda_OG_regularizer = [1] * matrix.shape[0]
            else:
                print("Low correlation between the omics detected.")
                lambda_C_regularizer = 0.01
                lambda_G_regularizer = 1
                lambda_O_regularizer = [1] * matrix.shape[0]
                lambda_OC_regularizer = [1] * matrix.shape[0]
                lambda_OG_regularizer = [0.01] * matrix.shape[0]
        
        # min-max normalization
        if normalization:
            matrix, omics_min, omics_max = min_max_normalization(data=matrix, zero_impute=True)
            matrix = np.array(matrix)
        
        # KNN preimpute
        if pre_impute:
            matrix = self.KNN_impute(matrix)
            matrix = np.array(matrix)

        # Gradient descent
        if earlystopping:
            self.threshold = (np.log10(lr * 1e3) * 3 + 1.39) * 1e-5
            loss_list = [1e8]
            loss_change_list = [1e8]
        
        self.L, self.N, self.M = matrix.shape
        self.matrix = torch.from_numpy(matrix).float().to(device)
        lambda_O_regularizer = torch.from_numpy(np.array(lambda_O_regularizer).reshape((len(lambda_O_regularizer), 1))).float().to(device)
        lambda_OC_regularizer = torch.from_numpy(np.array(lambda_OC_regularizer).reshape((len(lambda_OC_regularizer), 1))).float().to(device)
        lambda_OG_regularizer = torch.from_numpy(np.array(lambda_OG_regularizer).reshape((len(lambda_OG_regularizer), 1))).float().to(device)
        model = matrix_model_complete(self.K1, self.K2, self.K3, self.L, self.N, self.M, self.random_seed).to(device)
        optimizer = eval(opt)(model.parameters(), lr)
        for epoch in range(n_epochs):
            matrix_hat = model()
            loss = eval("model." + dist + "_loss")(self.matrix, matrix_hat, lambda_C_regularizer, lambda_G_regularizer, lambda_O_regularizer, lambda_OC_regularizer, lambda_OG_regularizer)
            running_loss = loss.detach().item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if verbose:
                print("Epoch {}| Loss: {:.4f}".format(epoch, running_loss))
            
            # detecting earlystoping:
            if earlystopping:
                loss_list.append(running_loss)
                loss_change_list.append(abs(loss_list[-2]-loss_list[-1])/loss_list[-1])
                if loss_change_list[-1] < self.threshold and loss_change_list[-2] < self.threshold:
                    print("Early stop!")
                    break
        self.C = np.hstack((model.C1.cpu().detach().numpy(), model.C2.cpu().detach().numpy()))
        self.G = np.hstack((model.G1.cpu().detach().numpy(), model.G2.cpu().detach().numpy()))
        self.O = np.hstack((model.O1.cpu().detach().numpy(), model.O2.cpu().detach().numpy()))
        self.OC = model.OC.cpu().detach().numpy()
        self.OG = model.OG.cpu().detach().numpy()
        
        matrix_hat = matrix_hat.cpu().detach().numpy()
        if dist == "poisson" or dist == "negative_bionomial":
            matrix_hat = np.exp(matrix_hat)
        
        if normalization:
            matrix_hat = matrix_hat - 0.1
            for i in range(len(matrix_hat)):
                if np.isnan(omics_max[i]).any() or np.isnan(omics_min[i]).any():
                    imputer = KNNImputer()
                    omics_max[i] = imputer.fit_transform(np.vstack((matrix_hat[i], omics_max[i].reshape((1, -1)))).T)[:, -1]
                    omics_min[i] = imputer.fit_transform(np.vstack((matrix_hat[i], omics_min[i].reshape((1, -1)))).T)[:, -1]
                matrix_hat[i] = matrix_hat[i] * (omics_max[i] - omics_min[i]) + omics_min[i]
        
        return matrix_hat


    def fit_list(self, matrix_list, normalization=True, pre_impute=False, opt="Adam", dist="gaussian", lr=1e-2, n_epochs=1000, lambda_C_regularizer=0, lambda_G_regularizer=0, lambda_O_regularizer=[0, 0], batch_size=256, earlystopping=True, device='cuda' if torch.cuda.is_available() else 'cpu', verbose=True):
        # tune the coefficient
        if (lambda_C_regularizer==0) or (lambda_G_regularizer==0) or (0 in lambda_O_regularizer):
            print("Automatically tune the coefficients for the penalty terms.")
            lambda_C_regularizer = 0.01
            lambda_G_regularizer = 0.01
            lambda_O_regularizer = [0.01] * len(matrix_list)
        if earlystopping:
            self.threshold = (np.log10(lr * 1e3) * 3 + 1.39) * 1e-5
            loss_list = [1e8]
            loss_change_list = [1e8]
        self.L = len(matrix_list)
        self.N = matrix_list[0].shape[0]
        self.M_list = [matrix_list[i].shape[1] for i in range(self.L)]
        
        # min-max normalization
        if normalization:
            matrix_list, omics_min, omics_max = min_max_normalization(data=matrix_list, zero_impute=False)
        
        # KNN preimpute
        if pre_impute:
            matrix_list = self.KNN_impute(matrix_list)

        # Gradient descent
        self.matrix_list = [torch.from_numpy(matrix_list[i]).float().to(device) for i in range(self.L)]
        index = []
        for l in range(self.L):
            for n in range(self.N):
                for m in range(self.M_list[l]):
                    if not torch.isnan(self.matrix_list[l][n, m]):
                        index.append([l, n, m])
        index = torch.from_numpy(np.array(index)).long()
        lambda_O_regularizer = torch.from_numpy(np.array(lambda_O_regularizer)).float().to(device)
        dataset = TensorDataset(index)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        model = matrix_list_model(self.K1, self.K2, self.K3, self.L, self.N, self.M_list, self.random_seed).to(device)
        optimizer = eval(opt)(model.parameters(), lr)
        for epoch in range(n_epochs):
            running_loss = 0.0
            for i, (x,) in enumerate(dataloader):
                matrix_list_hat = model()
                loss = eval("model." + dist + "_loss")(self.matrix_list, matrix_list_hat, x.T.numpy(), lambda_C_regularizer, lambda_G_regularizer, lambda_O_regularizer)
                running_loss += loss.detach().item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            if verbose:
                print("Epoch {}| Loss: {:.4f}".format(epoch, running_loss))
            
            # detecting earlystoping:
            if earlystopping:
                loss_list.append(running_loss)
                loss_change_list.append(abs(loss_list[-2]-loss_list[-1])/loss_list[-1])
                if loss_change_list[-1] < self.threshold and loss_change_list[-2] < self.threshold:
                    print("Early stop!")
                    break
        self.C = np.hstack((model.C1.cpu().detach().numpy(), model.C2.cpu().detach().numpy()))
        self.G = [np.hstack((model.G1[i].cpu().detach().numpy(), model.G2[i].cpu().detach().numpy())) for i in range(self.L)]
        self.O = np.hstack((model.O1.cpu().detach().numpy(), model.O2.cpu().detach().numpy()))

        matrix_list_hat = [matrix_list_hat[i].cpu().detach().numpy() for i in range(self.L)]
        if dist == "poisson" or dist == "negative bionomial":
            matrix_list_hat = [np.exp(matrix_list_hat[i]) for i in range(self.L)]
        
        if normalization:
            matrix_list_hat = matrix_list_hat - 0.1
            for i in range(len(matrix_list_hat)):
                if np.isnan(omics_max[i]).any() or np.isnan(omics_min[i]).any():
                    imputer = KNNImputer()
                    omics_max[i] = imputer.fit_transform(np.vstack((matrix_list_hat[i], omics_max[i].reshape((1, -1)))).T)[:, -1]
                    omics_min[i] = imputer.fit_transform(np.vstack((matrix_list_hat[i], omics_min[i].reshape((1, -1)))).T)[:, -1]
                matrix_list_hat[i] = matrix_list_hat[i] * (omics_max[i] - omics_min[i]) + omics_min[i]

        return matrix_list_hat


    def fit_list_complete(self, matrix_list, normalization=True, pre_impute=True, opt="Adam", dist="gaussian", lr=1e-2, n_epochs=1000, lambda_C_regularizer=0, lambda_G_regularizer=0, lambda_O_regularizer=[0, 0], batch_size=256, earlystopping=True, device='cuda' if torch.cuda.is_available() else 'cpu', verbose=True):
        # Tune the coefficient
        if (lambda_C_regularizer==0) or (lambda_G_regularizer==0) or (0 in lambda_O_regularizer):
            print("Automatically tune the coefficients for the penalty terms.")
            lambda_C_regularizer = 0.01
            lambda_G_regularizer = 0.01
            lambda_O_regularizer = [0.01] * len(matrix_list)
        if earlystopping:
            self.threshold = (np.log10(lr * 1e3) * 3 + 1.39) * 1e-5
            loss_list = [1e8]
            loss_change_list = [1e8]
        self.L = len(matrix_list)
        self.N = matrix_list[0].shape[0]
        self.M_list = [matrix_list[i].shape[1] for i in range(self.L)]
        
        # min-max normalization
        if normalization:
            matrix_list, omics_min, omics_max = min_max_normalization(data=matrix_list, zero_impute=True)

        # KNN preimpute
        if pre_impute:
            matrix_list = self.KNN_impute(matrix_list)
        
        # Gradient descent
        self.matrix_list = [torch.from_numpy(matrix_list[i]).float().to(device) for i in range(self.L)]
        lambda_O_regularizer = torch.from_numpy(np.array(lambda_O_regularizer)).float().to(device)
        model = matrix_list_model_complete(self.K1, self.K2, self.K3, self.L, self.N, self.M_list, self.random_seed).to(device)
        optimizer = eval(opt)(model.parameters(), lr)
        for epoch in range(n_epochs):
            matrix_list_hat = model()
            loss = eval("model." + dist + "_loss")(self.matrix_list, matrix_list_hat, lambda_C_regularizer, lambda_G_regularizer, lambda_O_regularizer)
            running_loss = loss.detach().item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if verbose:
                print("Epoch {}| Loss: {:.4f}".format(epoch, running_loss))
            
            # detecting earlystoping:
            if earlystopping:
                loss_list.append(running_loss)
                loss_change_list.append(abs(loss_list[-2]-loss_list[-1])/loss_list[-1])
                if loss_change_list[-1] < self.threshold and loss_change_list[-2] < self.threshold:
                    print("Early stop!")
                    break
        self.C = np.hstack((model.C1.cpu().detach().numpy(), model.C2.cpu().detach().numpy()))
        self.G = [np.hstack((model.G1[i].cpu().detach().numpy(), model.G2[i].cpu().detach().numpy())) for i in range(self.L)]
        self.O = np.hstack((model.O1.cpu().detach().numpy(), model.O2.cpu().detach().numpy()))
        
        matrix_list_hat = [matrix_list_hat[i].cpu().detach().numpy() for i in range(self.L)]
        if dist == "poisson" or dist == "negative bionomial":
            matrix_list_hat = [np.exp(matrix_list_hat[i]) for i in range(self.L)]
        
        if normalization:
            matrix_list_hat = matrix_list_hat - 0.1
            for i in range(len(matrix_list_hat)):
                if np.isnan(omics_max[i]).any() or np.isnan(omics_min[i]).any():
                    imputer = KNNImputer()
                    omics_max[i] = imputer.fit_transform(np.vstack((matrix_list_hat[i], omics_max[i].reshape((1, -1)))).T)[:, -1]
                    omics_min[i] = imputer.fit_transform(np.vstack((matrix_list_hat[i], omics_min[i].reshape((1, -1)))).T)[:, -1]
                matrix_list_hat[i] = matrix_list_hat[i] * (omics_max[i] - omics_min[i]) + omics_min[i]

        return matrix_list_hat


