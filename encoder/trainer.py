import numpy as np
import time
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
        criterion,
        optimizer,
        # train_dataloader,
        # eval_dataloader,
        # test_dataloader,
        train_data, 
        vad_data_tr, 
        vad_data_te, 
        test_data_tr, 
        test_data_te,
        submission_data,
        args,
    ):

        self.args = args
        self.criterion = criterion
        self.cuda_condition = torch.cuda.is_available() and not self.args.cuda
        # self.device = torch.device("cuda" if self.cuda_condition else "cpu")
        self.update_count = 0

        self.model = model
        if self.cuda_condition:
            self.model.cuda()

        # Setting the train and test data loader
        self.train_data = train_data
        self.vad_data_tr = vad_data_tr
        self.vad_data_te = vad_data_te
        self.test_data_tr = test_data_tr
        self.test_data_te = test_data_te
        self.submission_data = submission_data

        self.N = train_data.shape[0]
        self.idxlist = list(range(self.N))

        self.data_name = self.args.data_name
        # betas = (self.args.adam_beta1, self.args.adam_beta2)
        # self.optim = Adam(
        #     self.model.parameters(),
        #     lr=self.args.lr,
        #     betas=betas,
        #     weight_decay=self.args.weight_decay,
        # )
        self.optimizer = optimizer

        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))
        # self.criterion = nn.BCELoss()


    def train(self, epoch, is_VAE = False):
        # Turn on training model
        self.model.train()
        train_loss = 0.0
        start_time = time.time()

        np.random.shuffle(self.idxlist)
        
        for batch_idx, start_idx in enumerate(range(0, self.N, self.args.batch_size)):
            # device = torch.device("cuda" if self.args.cuda else "cpu")
            end_idx = min(start_idx + self.args.batch_size, self.N)
            data = self.train_data[self.idxlist[start_idx:end_idx]]
            data = self.naive_sparse2tensor(data).to(self.args.device)
            self.optimizer.zero_grad()

            if is_VAE:
                if self.args.total_anneal_steps > 0:
                    anneal = min(self.args.anneal_cap, 
                                1. * self.update_count / self.args.total_anneal_steps)
                else:
                    anneal = self.args.anneal_cap

                #TODO
                #model에 입력 출력 코드를 작성해주세요
                #loss 함수를 설정해주세요
                self.optimizer.zero_grad()
                recon_batch, mu, logvar = self.model(data)
                
                loss = self.criterion(recon_batch, data, mu, logvar, anneal)
            else:
                recon_batch = self.model(data)
                loss = self.criterion(recon_batch, data)

            loss.backward()
            train_loss += loss.item()
            self.optimizer.step()

            self.update_count += 1

            if batch_idx % self.args.log_freq == 0 and batch_idx > 0:
                elapsed = time.time() - start_time
                print('| epoch {:3d} | {:4d}/{:4d} batches | ms/batch {:4.2f} | '
                        'loss {:4.2f}'.format(
                            epoch, batch_idx, len(range(0, self.N, self.args.batch_size)),
                            elapsed * 1000 / self.args.log_freq,
                            train_loss / self.args.log_freq))

                start_time = time.time()
                train_loss = 0.0


    def evaluate(self, is_VAE=False):
        # Turn on evaluation mode
        self.model.eval()
        total_loss = 0.0
        # global update_count
        e_idxlist = list(range(self.vad_data_tr.shape[0]))
        e_N = self.vad_data_tr.shape[0]
        n100_list = []
        r20_list = []
        r50_list = []
        
        with torch.no_grad():
            for start_idx in range(0, e_N, self.args.batch_size):
                end_idx = min(start_idx + self.args.batch_size, self.N)
                data = self.vad_data_tr[e_idxlist[start_idx:end_idx]]
                heldout_data = self.vad_data_te[e_idxlist[start_idx:end_idx]]

                data_tensor = self.naive_sparse2tensor(data).to(self.args.device)
                if is_VAE :
                    if self.args.total_anneal_steps > 0:
                        anneal = min(self.args.anneal_cap, 
                                        1. * self.update_count / self.args.total_anneal_steps)
                    else:
                        anneal = self.args.anneal_cap
                
                    #TODO
                    #model에 입력 출력 코드를 작성해주세요
                    #loss 함수를 설정해주세요
                    recon_batch, mu, logvar = self.model(data_tensor)

                    loss = self.criterion(recon_batch, data_tensor, mu, logvar, anneal)

                else :
                    recon_batch = self.model(data_tensor)
                    loss = self.criterion(recon_batch, data_tensor)

                total_loss += loss.item()

                # Exclude examples from training set
                recon_batch = recon_batch.cpu().numpy()
                recon_batch[data.nonzero()] = -np.inf

                n100 = NDCG_binary_at_k_batch(recon_batch, heldout_data, 100)
                r20 = Recall_at_k_batch(recon_batch, heldout_data, 20)
                r50 = Recall_at_k_batch(recon_batch, heldout_data, 50)

                n100_list.append(n100)
                r20_list.append(r20)
                r50_list.append(r50)
    
        total_loss /= len(range(0, e_N, self.args.batch_size))
        n100_list = np.concatenate(n100_list)
        r20_list = np.concatenate(r20_list)
        r50_list = np.concatenate(r50_list)

        return total_loss, np.mean(n100_list), np.mean(r20_list), np.mean(r50_list)


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


    def naive_sparse2tensor(self, data):
        return torch.FloatTensor(data.toarray())