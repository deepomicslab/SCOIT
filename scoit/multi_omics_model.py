import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, Adam, Adadelta, Adagrad, AdamW, SparseAdam, Adamax, ASGD, LBFGS
import random

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


class matrix_model(nn.Module):
    def __init__(self, K1, K2, K3, L, N, M, random_seed=111):
        super().__init__()
        setup_seed(random_seed)
        self.L = L
        self.N = N
        self.M = M

        self.C1 = nn.Parameter(torch.normal(0, 0.1, size=(self.N, K1), requires_grad=True, dtype=torch.float)) 
        self.C2 = nn.Parameter(torch.normal(0, 0.1, size=(self.N, K2), requires_grad=True, dtype=torch.float)) 

        self.G1 = nn.Parameter(torch.normal(0, 0.1, size=(self.M, K1), requires_grad=True, dtype=torch.float)) 
        self.G2 = nn.Parameter(torch.normal(0, 0.1, size=(self.M, K3), requires_grad=True, dtype=torch.float)) 

        self.O1 = nn.Parameter(torch.normal(0, 0.1, size=(self.L, K2), requires_grad=True, dtype=torch.float)) 
        self.O2 = nn.Parameter(torch.normal(0, 0.1, size=(self.L, K3), requires_grad=True, dtype=torch.float)) 
        
        self.OC = nn.Parameter(torch.normal(0, 0.1, size=(self.L * self.N, K1+K3), requires_grad=True, dtype=torch.float)) 
        self.OG = nn.Parameter(torch.normal(0, 0.1, size=(self.L * self.M, K1+K2), requires_grad=True, dtype=torch.float)) 
        
        self.beta_global = nn.Parameter(torch.normal(0, 0.1, size=(1, self.M), requires_grad=True, dtype=torch.float))
        self.beta_local = nn.Parameter(torch.normal(0, 0.1, size=(self.L, self.M), requires_grad=True, dtype=torch.float))
    
    def forward(self):
        
        return torch.matmul(self.C1, self.G1.T).reshape(1, self.N, self.M).expand(self.L, self.N, self.M) + torch.matmul(self.O1, self.C2.T).reshape(self.L, self.N, 1).expand(self.L, self.N, self.M) + torch.matmul(self.O2, self.G2.T).reshape(self.L, 1, self.M).expand(self.L, self.N, self.M) + torch.matmul(torch.cat([self.C1, self.C2], dim=1), self.OG.T).T.reshape(self.L, self.M, self.N).permute(0, 2, 1) + torch.matmul(torch.cat([self.G1, self.G2], dim=1), self.OC.T).T.reshape(self.L, self.N, self.M)

    def poisson_loss(self, matrix, matrix_hat, x, lamda_C, lamda_G, lamda_O, lamda_OC, lamda_OG):
        
        lamda_O_ = torch.sqrt(lamda_O)
        lamda_OC_ = torch.sqrt(lamda_OC.repeat(1, self.N).reshape((-1,1)))
        lamda_OG_ = torch.sqrt(lamda_OG.repeat(1, self.M).reshape((-1,1)))
        loss_ = torch.sum(torch.exp(matrix_hat[x]) - torch.mul(matrix[x], matrix_hat[x])) + lamda_C * (torch.sum(torch.square(self.C1[x[1]])) + torch.sum(torch.square(self.C2[x[1]]))) + lamda_G * (torch.sum(torch.square(self.G1[x[2]])) + torch.sum(torch.square(self.G2[x[2]]))) + torch.sum(torch.square((lamda_O_ * self.O1)[x[0]])) + torch.sum(torch.square((lamda_O_ * self.O2)[x[0]])) + torch.sum(torch.square((lamda_OC_ * self.OC)[x[0] * self.N + x[1]])) + torch.sum(torch.square((lamda_OG_ * self.OG)[x[0] * self.M + x[2]]))
        
        return loss_

    def gaussian_loss(self, matrix, matrix_hat, x, lamda_C, lamda_G, lamda_O, lamda_OC, lamda_OG, sigma=0.5):
        
        lamda_O_ = torch.sqrt(lamda_O)
        lamda_OC_ = torch.sqrt(lamda_OC.repeat(1, self.N).reshape((-1,1)))
        lamda_OG_ = torch.sqrt(lamda_OG.repeat(1, self.M).reshape((-1,1)))
        loss_ = torch.sum(torch.square(matrix_hat[x]-matrix[x]))/(2*sigma) + lamda_C * (torch.sum(torch.square(self.C1[x[1]])) + torch.sum(torch.square(self.C2[x[1]]))) + lamda_G * (torch.sum(torch.square(self.G1[x[2]])) + torch.sum(torch.square(self.G2[x[2]]))) + torch.sum(torch.square((lamda_O_ * self.O1)[x[0]])) + torch.sum(torch.square((lamda_O_ * self.O2)[x[0]])) + torch.sum(torch.square((lamda_OC_ * self.OC)[x[0] * self.N + x[1]])) + torch.sum(torch.square((lamda_OG_ * self.OG)[x[0] * self.M + x[2]]))

        return loss_

    def negative_bionomial_loss(self, matrix, matrix_hat, x, lamda_C, lamda_G, lamda_O, lamda_OC, lamda_OG, alpha=0):

        lamda_O_ = torch.sqrt(lamda_O)
        lamda_OC_ = torch.sqrt(lamda_OC.repeat(1, self.N).reshape((-1,1)))
        lamda_OG_ = torch.sqrt(lamda_OG.repeat(1, self.M).reshape((-1,1)))
        beta_global = (self.beta_global ** 2).reshape(1, 1, self.M).expand(self.L, self.N, self.M)
        beta_local = (self.beta_local ** 2).reshape(self.L, 1, self.M).expand(self.L, self.N, self.M)
        beta = alpha * beta_global + (1 - alpha) * beta_local
        mu_beta = torch.mul(torch.exp(matrix_hat[x]), beta[x])

        loss_ = torch.sum(-torch.mul(mu_beta, torch.log(beta[x])) + torch.lgamma(mu_beta) - torch.lgamma(matrix[x] + mu_beta) + torch.mul((matrix[x] + mu_beta), torch.log(1 + beta[x]))) + lamda_C * (torch.sum(torch.square(self.C1[x[1]])) + torch.sum(torch.square(self.C2[x[1]]))) + lamda_G * (torch.sum(torch.square(self.G1[x[2]])) + torch.sum(torch.square(self.G2[x[2]]))) + torch.sum(torch.square((lamda_O_ * self.O1)[x[0]])) + torch.sum(torch.square((lamda_O_ * self.O2)[x[0]])) + torch.sum(torch.square((lamda_OC_ * self.OC)[x[0] * self.N + x[1]])) + torch.sum(torch.square((lamda_OG_ * self.OG)[x[0] * self.M + x[2]]))

        return loss_


class matrix_model_complete(nn.Module):
    def __init__(self, K1, K2, K3, L, N, M, random_seed=111):
        super().__init__()
        setup_seed(random_seed)
        self.L = L
        self.N = N
        self.M = M
        
        self.C1 = nn.Parameter(torch.normal(0, 0.1, size=(self.N, K1), requires_grad=True, dtype=torch.float)) 
        self.C2 = nn.Parameter(torch.normal(0, 0.1, size=(self.N, K2), requires_grad=True, dtype=torch.float)) 

        self.G1 = nn.Parameter(torch.normal(0, 0.1, size=(self.M, K1), requires_grad=True, dtype=torch.float)) 
        self.G2 = nn.Parameter(torch.normal(0, 0.1, size=(self.M, K3), requires_grad=True, dtype=torch.float)) 

        self.O1 = nn.Parameter(torch.normal(0, 0.1, size=(self.L, K2), requires_grad=True, dtype=torch.float)) 
        self.O2 = nn.Parameter(torch.normal(0, 0.1, size=(self.L, K3), requires_grad=True, dtype=torch.float)) 
       
        self.OC = nn.Parameter(torch.normal(0, 0.1, size=(self.L * self.N, K1+K3), requires_grad=True, dtype=torch.float))
        self.OG = nn.Parameter(torch.normal(0, 0.1, size=(self.L * self.M, K1+K2), requires_grad=True, dtype=torch.float))

        self.beta_global = nn.Parameter(torch.normal(0, 0.1, size=(1, self.M), requires_grad=True, dtype=torch.float))
        self.beta_local = nn.Parameter(torch.normal(0, 0.1, size=(self.L, self.M), requires_grad=True, dtype=torch.float))
    
    def forward(self):

        return torch.matmul(self.C1, self.G1.T).reshape(1, self.N, self.M).expand(self.L, self.N, self.M) + torch.matmul(self.O1, self.C2.T).reshape(self.L, self.N, 1).expand(self.L, self.N, self.M) + torch.matmul(self.O2, self.G2.T).reshape(self.L, 1, self.M).expand(self.L, self.N, self.M) + torch.matmul(torch.cat([self.C1, self.C2], dim=1), self.OG.T).T.reshape(self.L, self.M, self.N).permute(0, 2, 1) + torch.matmul(torch.cat([self.G1, self.G2], dim=1), self.OC.T).T.reshape(self.L, self.N, self.M)
       
    def poisson_loss(self, matrix, matrix_hat, lamda_C, lamda_G, lamda_O, lamda_OC, lamda_OG):

        lamda_O_ = torch.sqrt(lamda_O)
        lamda_OC_ = torch.sqrt(lamda_OC.repeat(1, self.N).reshape((-1,1)))
        lamda_OG_ = torch.sqrt(lamda_OG.repeat(1, self.M).reshape((-1,1)))
        loss_ = torch.sum(torch.exp(matrix_hat) - torch.mul(matrix, matrix_hat)) + lamda_C * (torch.sum(torch.square(self.C1)) + torch.sum(torch.square(self.C2))) + lamda_G * (torch.sum(torch.square(self.G1)) + torch.sum(torch.square(self.G2))) + torch.sum(torch.square(lamda_O_ * self.O1)) + torch.sum(torch.square(lamda_O_ * self.O2)) + torch.sum(torch.square(lamda_OC_ * self.OC)) + torch.sum(torch.square(lamda_OG_ * self.OG))
         
        return loss_
    
    def gaussian_loss(self, matrix, matrix_hat, lamda_C, lamda_G, lamda_O, lamda_OC, lamda_OG, sigma=0.5):

        lamda_O_ = torch.sqrt(lamda_O)
        lamda_OC_ = torch.sqrt(lamda_OC.repeat(1, self.N).reshape((-1,1)))
        lamda_OG_ = torch.sqrt(lamda_OG.repeat(1, self.M).reshape((-1,1)))
        loss_ = torch.sum(torch.square(matrix_hat - matrix))/(2*sigma) + lamda_C * (torch.sum(torch.square(self.C1)) + torch.sum(torch.square(self.C2))) + lamda_G * (torch.sum(torch.square(self.G1)) + torch.sum(torch.square(self.G2))) + torch.sum(torch.square(lamda_O_ * self.O1)) + torch.sum(torch.square(lamda_O_ * self.O2)) + torch.sum(torch.square(lamda_OC_ * self.OC)) + torch.sum(torch.square(lamda_OG_ * self.OG))

        return loss_
    
    
    def negative_bionomial_loss(self, matrix, matrix_hat, lamda_C, lamda_G, lamda_O, lamda_OC, lamda_OG, alpha=0):
        
        lamda_O_ = torch.sqrt(lamda_O)
        lamda_OC_ = torch.sqrt(lamda_OC.repeat(1, self.N).reshape((-1,1)))
        lamda_OG_ = torch.sqrt(lamda_OG.repeat(1, self.M).reshape((-1,1)))
        beta_global = (self.beta_global ** 2).reshape(1, 1, self.M).expand(self.L, self.N, self.M)
        beta_local = (self.beta_local ** 2).reshape(self.L, 1, self.M).expand(self.L, self.N, self.M)
        beta = alpha * beta_global + (1 - alpha) * beta_local
        
        mu_beta = torch.mul(torch.exp(matrix_hat), beta)
        
        loss_ = torch.sum(-torch.mul(mu_beta, torch.log(beta)) + torch.lgamma(mu_beta) - torch.lgamma(matrix + mu_beta) + torch.mul((matrix + mu_beta), torch.log(1 + beta))) + lamda_C * (torch.sum(torch.square(self.C1)) + torch.sum(torch.square(self.C2))) + lamda_G * (torch.sum(torch.square(self.G1)) + torch.sum(torch.square(self.G2))) + torch.sum(torch.square(lamda_O_ * self.O1)) + torch.sum(torch.square(lamda_O_ * self.O2)) + torch.sum(torch.square(lamda_OC_ * self.OC)) + torch.sum(torch.square(lamda_OG_ * self.OG))
        
        return loss_


class matrix_list_model(nn.Module):
    def __init__(self, K1, K2, K3, L, N, M_list, random_seed=111):
        super().__init__()
        setup_seed(random_seed)
        self.L = L
        self.N = N
        self.M_list = M_list

        self.C1 = nn.Parameter(torch.normal(0, 0.1, size=(self.N, K1), requires_grad=True, dtype=torch.float)) 
        self.C2 = nn.Parameter(torch.normal(0, 0.1, size=(self.N, K2), requires_grad=True, dtype=torch.float)) 

        self.G1 = nn.ParameterList([nn.Parameter(torch.normal(0, 0.1, size=(M, K1), requires_grad=True, dtype=torch.float)) for M in self.M_list])
        self.G2 = nn.ParameterList([nn.Parameter(torch.normal(0, 0.1, size=(M, K3), requires_grad=True, dtype=torch.float)) for M in self.M_list])

        self.O1 = nn.Parameter(torch.normal(0, 0.1, size=(self.L, K2), requires_grad=True, dtype=torch.float)) 
        self.O2 = nn.Parameter(torch.normal(0, 0.1, size=(self.L, K3), requires_grad=True, dtype=torch.float)) 

        self.beta = nn.ParameterList([nn.Parameter(torch.normal(0, 0.1, size=(1, M), requires_grad=True, dtype=torch.float)) for M in self.M_list])

    def forward(self):
        
        return [torch.matmul(self.C1, self.G1[i].T) + torch.matmul(self.C2, self.O1[i:i+1].T).expand(self.N, self.M_list[i]) + torch.matmul(self.O2[i:i+1], self.G2[i].T).expand(self.N, self.M_list[i]) for i in range(self.L)]


    def poisson_loss(self, matrix_list, matrix_list_hat, x, lamda_C, lamda_G, lamda_O):
        loss_ = 0
        for l in range(self.L):
            x_ = np.array([x[1:, i] for i in range(x.shape[1]) if x[0, i]==l]).T
            if len(x_) != 0:
                loss_ += torch.sum(torch.exp(matrix_list_hat[l][x_]) - torch.mul(matrix_list[l][x_], matrix_list_hat[l][x_])) + lamda_C * (torch.sum(torch.square(self.C1[x_[0]])) + torch.sum(torch.square(self.C2[x_[0]]))) + lamda_G * (torch.sum(torch.square(self.G1[l][x_[1]])) + torch.sum(torch.square(self.G2[l][x_[1]]))) + lamda_O[l] * (torch.sum(torch.square(self.O1[l])) + torch.sum(torch.square(self.O2[l])))
        
        return loss_

    def gaussian_loss(self, matrix_list, matrix_list_hat, x, lamda_C, lamda_G, lamda_O, sigma=0.5):
        loss_ = 0
        for l in range(self.L):
            x_ = np.array([x[1:, i] for i in range(x.shape[1]) if x[0, i]==l]).T
            if len(x_) != 0:
                loss_ += torch.sum(torch.square(matrix_list_hat[l][x_]-matrix_list[l][x_]))/(2*sigma) + lamda_C * (torch.sum(torch.square(self.C1[x_[0]])) + torch.sum(torch.square(self.C2[x_[0]]))) + lamda_G * (torch.sum(torch.square(self.G1[l][x_[1]])) + torch.sum(torch.square(self.G2[l][x_[1]]))) + lamda_O[l] * (torch.sum(torch.square(self.O1[l])) + torch.sum(torch.square(self.O2[l])))

        return loss_

    def negative_bionomial_loss(self, matrix_list, matrix_list_hat, x, lamda_C, lamda_G, lamda_O):
        loss_ = 0
        for l in range(self.L):
            beta = (self.beta[l] ** 2).expand(self.N, self.M_list[l])
            x_ = np.array([x[1:, i] for i in range(x.shape[1]) if x[0, i]==l]).T
            mu_beta = torch.mul(torch.exp(matrix_list_hat[l][x_]), beta[x_])
            if len(x_) != 0:
                loss_ += torch.sum(-torch.mul(mu_beta, torch.log(beta[x_])) + torch.lgamma(mu_beta) - torch.lgamma(matrix_list[l][x_] + mu_beta) + torch.mul((matrix_list[l][x_] + mu_beta), torch.log(1 + beta[x_]))) + lamda_C * (torch.sum(torch.square(self.C1[x_[0]])) + torch.sum(torch.square(self.C2[x_[0]]))) + lamda_G * (torch.sum(torch.square(self.G1[l][x_[1]])) + torch.sum(torch.square(self.G2[l][x_[1]]))) + lamda_O[l] * (torch.sum(torch.square(self.O1[l])) + torch.sum(torch.square(self.O2[l])))
        
        return loss_



class matrix_list_model_complete(nn.Module):
    def __init__(self, K1, K2, K3, L, N, M_list, random_seed=111):
        super().__init__()
        setup_seed(random_seed)
        self.L = L
        self.N = N
        self.M_list = M_list
        
        self.C1 = nn.Parameter(torch.normal(0, 0.1, size=(self.N, K1), requires_grad=True, dtype=torch.float)) 
        self.C2 = nn.Parameter(torch.normal(0, 0.1, size=(self.N, K2), requires_grad=True, dtype=torch.float)) 

        self.G1 = nn.ParameterList([nn.Parameter(torch.normal(0, 0.1, size=(M, K1), requires_grad=True, dtype=torch.float)) for M in self.M_list])
        self.G2 = nn.ParameterList([nn.Parameter(torch.normal(0, 0.1, size=(M, K3), requires_grad=True, dtype=torch.float)) for M in self.M_list])

        self.O1 = nn.Parameter(torch.normal(0, 0.1, size=(self.L, K2), requires_grad=True, dtype=torch.float)) 
        self.O2 = nn.Parameter(torch.normal(0, 0.1, size=(self.L, K3), requires_grad=True, dtype=torch.float)) 
        
        self.beta = nn.ParameterList([nn.Parameter(torch.normal(0, 0.1, size=(1, M), requires_grad=True, dtype=torch.float)) for M in self.M_list])
    
    def forward(self):

        return [torch.matmul(self.C1, self.G1[i].T) + torch.matmul(self.C2, self.O1[i:i+1].T).expand(self.N, self.M_list[i]) + torch.matmul(self.O2[i:i+1], self.G2[i].T).expand(self.N, self.M_list[i]) for i in range(self.L)]

    def poisson_loss(self, matrix_list, matrix_list_hat, lamda_C, lamda_G, lamda_O):
        loss_ = 0
        for l in range(self.L):
            loss_ += torch.sum(torch.exp(matrix_list_hat[l]) - torch.mul(matrix_list[l], matrix_list_hat[l]))
            loss_ += lamda_G * (torch.sum(torch.square(self.G1[l])) + torch.sum(torch.square(self.G2[l]))) + lamda_O[l] * (torch.sum(torch.square(self.O1[l])) + torch.sum(torch.square(self.O2[l])))
            
        loss_ += lamda_C * (torch.sum(torch.square(self.C1)) + torch.sum(torch.square(self.C2)))
         
        return loss_
    
    def gaussian_loss(self, matrix_list, matrix_list_hat, lamda_C, lamda_G, lamda_O, sigma=0.5):
        loss_ = 0
        for l in range(self.L):
            loss_ += torch.sum(torch.square(matrix_list_hat[l] - matrix_list[l])) / (2*sigma)
            loss_ += lamda_G * (torch.sum(torch.square(self.G1[l])) + torch.sum(torch.square(self.G2[l]))) + lamda_O[l] * (torch.sum(torch.square(self.O1[l])) + torch.sum(torch.square(self.O2[l])))

        loss_ += lamda_C * (torch.sum(torch.square(self.C1)) + torch.sum(torch.square(self.C2)))

        return loss_
    
    def negative_bionomial_loss(self, matrix_list, matrix_list_hat, lamda_C, lamda_G, lamda_O):
        loss_ = 0
        for l in range(self.L):
            beta = (self.beta[l] ** 2).expand(self.N, self.M_list[l])
            mu_beta = torch.mul(torch.exp(matrix_list_hat[l]), beta)
            loss_ += torch.sum(-torch.mul(mu_beta, torch.log(beta)) + torch.lgamma(mu_beta) - torch.lgamma(matrix_list[l] + mu_beta) + torch.mul((matrix_list[l] + mu_beta), torch.log(1 + beta)))
            loss_ += lamda_G * (torch.sum(torch.square(self.G1[l])) + torch.sum(torch.square(self.G2[l]))) + lamda_O[l] * (torch.sum(torch.square(self.O1[l])) + torch.sum(torch.square(self.O2[l])))

        loss_ += lamda_C * (torch.sum(torch.square(self.C1)) + torch.sum(torch.square(self.C2)))

        return loss_
