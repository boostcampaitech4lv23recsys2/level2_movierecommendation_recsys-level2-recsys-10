import numpy as np
from time import time
import torch
import torch.nn as nn
import tqdm
from torch.optim import Adam
from utils import (
    NDCG_binary_at_k_batch,
    Recall_at_k_batch
)

class Trainer:
    def __init__(
        self,
        model,
        # criterion,
        optimizer,
        train_loader,
        valid_loader,
        # test_dataloader,
        # train_data, 
        # vad_data_tr, 
        # vad_data_te, 
        # test_data_tr, 
        # test_data_te,
        # submission_data,
        args,
    ):
        self.args = args
        self.model = model
        if self.args.device=="cuda":
            self.model.cuda()

        # self.criterion = model.loss_function
        self.optimizer = optimizer
        # self.cuda_condition = torch.cuda.is_available() and not self.args.cuda
        # self.device = torch.device("cuda" if self.cuda_condition else "cpu")
        self.update_count = 0

        # Setting the train and test data loader
        # self.train_data = train_data
        # self.vad_data_tr = vad_data_tr
        # self.vad_data_te = vad_data_te
        # self.test_data_tr = test_data_tr
        # self.test_data_te = test_data_te
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        # self.submission_data = submission_data

        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))
        # self.criterion = nn.BCELoss()


    def train(self, epoch, model_name):
        # Turn on training model
        self.model.train()
        train_loss = 0.0
        start_time = time()
        
        for batch_idx, batch_data in enumerate(self.train_loader):
            input_data = batch_data.to(self.args.device)

            if model_name == "multiVAE":
                self.optimizer.zero_grad()
                if self.args.total_anneal_steps > 0:
                    anneal = min(self.args.anneal_cap, 
                                1. * self.update_count / self.args.total_anneal_steps)
                else:
                    anneal = self.args.anneal_cap
                recon_batch, mu, logvar = self.model(input_data)
                loss = self.model.loss_function(recon_batch, input_data, mu, logvar, anneal)
                
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
            elif model_name == "multiDAE":
                self.optimizer.zero_grad()
                recon_batch = self.model(input_data)
                loss = self.model.loss_function(recon_batch, input_data)
                
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
            elif model_name == "recVAE":
                for opt in self.optimizer:
                    opt.zero_grad()
                recon_batch, mu, logvar = self.model(input_data)
                _, loss = self.model.loss_function(recon_batch, input_data, mu, logvar, 
                                    self.args.beta, self.args.gamma)
                
                loss.backward()
                for opt in self.optimizer:
                    opt.step()
                train_loss += loss.item()

            
            
            
            self.update_count += 1

            if batch_idx % self.args.log_freq == 0 and batch_idx > 0:
                elapsed = time() - start_time
                print('| epoch {:3d} | {:4d}/{:4d} batches | ms/batch {:4.2f} | '
                        'loss {:4.2f}'.format(
                            epoch, batch_idx, len(range(0, 6807, self.args.batch_size)),
                            elapsed * 1000 / self.args.log_freq,
                            train_loss / self.args.log_freq))

                start_time = time()
                train_loss = 0.0


    def evaluate(self, model_name):
        # Turn on evaluation mode
        self.model.eval()
        total_loss = 0.0

        r10_list = []
        r20_list = []
        
        with torch.no_grad():
            for batch_data in self.valid_loader:
                input_data, label_data = batch_data # label_data: 오로지 정답지로만 사용
                input_data = input_data.to(self.args.device)
                label_data = label_data.to(self.args.device)
                label_data = label_data.cpu().numpy()
                if model_name == "multiVAE" :
                    if self.args.total_anneal_steps > 0:
                        anneal = min(self.args.anneal_cap, 
                                        1. * self.update_count / self.args.total_anneal_steps)
                    else:
                        anneal = self.args.anneal_cap
                
                    recon_batch, mu, logvar = self.model(input_data)
                    loss = self.model.loss_function(recon_batch, input_data, mu, logvar, anneal)
                elif model_name == "multiDAE" :
                    recon_batch = self.model(input_data)
                    loss = self.model.loss_function(recon_batch, input_data)
                elif model_name == "recVAE":
                    recon_batch, mu, logvar = self.model(input_data)
                    _, loss = self.model.loss_function(recon_batch, input_data, mu, logvar, gamma=1, beta=None)

                total_loss += loss.item()
                recon_batch = recon_batch.cpu().numpy()
                recon_batch[input_data.cpu().numpy().nonzero()] = -np.inf

                # n100 = NDCG_binary_at_k_batch(recon_batch, heldout_data, 100)
                r10 = Recall_at_k_batch(recon_batch, label_data, 10)
                r20 = Recall_at_k_batch(recon_batch, label_data, 20)

                r10_list.append(r10)
                r20_list.append(r20)
    
        total_loss /= len(range(0, 6807, self.args.batch_size)) #6807????
        # n100_list = np.concatenate(n100_list)
        r10_list = np.concatenate(r10_list)
        r20_list = np.concatenate(r20_list)

        return total_loss, np.mean(r10_list), np.mean(r20_list)


    # def sparse2torch_sparse(self, data):
    #     """
    #     Convert scipy sparse matrix to torch sparse tensor with L2 Normalization
    #     This is much faster than naive use of torch.FloatTensor(data.toarray())
    #     https://discuss.pytorch.org/t/sparse-tensor-use-cases/22047/2
    #     """
    #     samples = data.shape[0]
    #     features = data.shape[1]
    #     coo_data = data.tocoo()
    #     indices = torch.LongTensor([coo_data.row, coo_data.col])
    #     row_norms_inv = 1 / np.sqrt(data.sum(1))
    #     row2val = {i : row_norms_inv[i].item() for i in range(samples)}
    #     values = np.array([row2val[r] for r in coo_data.row])
    #     t = torch.sparse.FloatTensor(indices, torch.from_numpy(values).float(), [samples, features])
    #     return t


    # def naive_sparse2tensor(self, data):
    #     return torch.FloatTensor(data.toarray())