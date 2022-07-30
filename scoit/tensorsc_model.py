import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, Adam
from torch.utils.data import Dataset, DataLoader, TensorDataset
from scot import matrix_model, matrix_model_complete, matrix_list_model, matrix_list_model_complete

class sc_multi_omics:
    def __init__(self, K1=20, K2=20, K3=20, random_seed=111):
        self.K1 = K1
        self.K2 = K2
        self.K3 = K3
        self.random_seed = random_seed

    
    def fit(self, matrix, opt="Adam", dist="gaussian", lr=1e-2, n_epochs=1000, lambda_C_regularizer=0.01, lambda_G_regularizer=0.01, lambda_O_regularizer=[0.01, 0.01], lambda_OC_regularizer=[1, 1], lambda_OG_regularizer=[1, 1], batch_size=100, device='cuda' if torch.cuda.is_available() else 'cpu', verbose=True):
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

        self.C = np.hstack((model.C1.cpu().detach().numpy(), model.C2.cpu().detach().numpy()))
        self.G = np.hstack((model.G1.cpu().detach().numpy(), model.G2.cpu().detach().numpy()))
        self.O = np.hstack((model.O1.cpu().detach().numpy(), model.O2.cpu().detach().numpy()))
        self.OC = model.OC.cpu().detach().numpy()
        self.OG = model.OG.cpu().detach().numpy()
        
        matrix_hat = matrix_hat.cpu().detach().numpy()
        if dist == "poisson" or dist == "negative_bionomial":
            matrix_hat = np.exp(matrix_hat)

        return matrix_hat
    
    
    def fit_complete(self, matrix, opt="Adam", dist="gaussian", lr=1e-2, n_epochs=1000, lambda_C_regularizer=0.01, lambda_G_regularizer=0.01, lambda_O_regularizer=[0.01, 0.01], lambda_OC_regularizer=[1, 1], lambda_OG_regularizer=[1, 1], batch_size=100, device='cuda' if torch.cuda.is_available() else 'cpu', verbose=True):
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

        self.C = np.hstack((model.C1.cpu().detach().numpy(), model.C2.cpu().detach().numpy()))
        self.G = np.hstack((model.G1.cpu().detach().numpy(), model.G2.cpu().detach().numpy()))
        self.O = np.hstack((model.O1.cpu().detach().numpy(), model.O2.cpu().detach().numpy()))
        self.OC = model.OC.cpu().detach().numpy()
        self.OG = model.OG.cpu().detach().numpy()
        
        matrix_hat = matrix_hat.cpu().detach().numpy()
        if dist == "poisson" or dist == "negative_bionomial":
            matrix_hat = np.exp(matrix_hat)
        
        return matrix_hat


    def fit_list(self, matrix_list, opt="Adam", dist="gaussian", lr=1e-2, n_epochs=1000, lambda_C_regularizer=0.01, lambda_G_regularizer=0.01, lambda_O_regularizer=[0.01, 0.01], batch_size=100, device='cuda' if torch.cuda.is_available() else 'cpu', verbose=True):
        self.L = len(matrix_list)
        self.N = matrix_list[0].shape[0]
        self.M_list = [matrix_list[i].shape[1] for i in range(self.L)]
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

        self.C = np.hstack((model.C1.cpu().detach().numpy(), model.C2.cpu().detach().numpy()))
        self.G = [np.hstack((model.G1[i].cpu().detach().numpy(), model.G2[i].cpu().detach().numpy())) for i in range(self.L)]
        self.O = np.hstack((model.O1.cpu().detach().numpy(), model.O2.cpu().detach().numpy()))

        matrix_list_hat = [matrix_list_hat[i].cpu().detach().numpy() for i in range(self.L)]
        if dist == "poisson" or dist == "negative bionomial":
            matrix_list_hat = [np.exp(matrix_list_hat[i]) for i in range(self.L)]

        return matrix_list_hat


    def fit_list_complete(self, matrix_list, opt="Adam", dist="gaussian", lr=1e-2, n_epochs=1000, lambda_C_regularizer=0.01, lambda_G_regularizer=0.01, lambda_O_regularizer=[0.01, 0.01], batch_size=100, device='cuda' if torch.cuda.is_available() else 'cpu', verbose=True):
        self.L = len(matrix_list)
        self.N = matrix_list[0].shape[0]
        self.M_list = [matrix_list[i].shape[1] for i in range(self.L)]
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

        self.C = np.hstack((model.C1.cpu().detach().numpy(), model.C2.cpu().detach().numpy()))
        self.G = [np.hstack((model.G1[i].cpu().detach().numpy(), model.G2[i].cpu().detach().numpy())) for i in range(self.L)]
        self.O = np.hstack((model.O1.cpu().detach().numpy(), model.O2.cpu().detach().numpy()))
        
        matrix_list_hat = [matrix_list_hat[i].cpu().detach().numpy() for i in range(self.L)]
        if dist == "poisson" or dist == "negative bionomial":
            matrix_list_hat = [np.exp(matrix_list_hat[i]) for i in range(self.L)]

        return matrix_list_hat


