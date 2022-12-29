import numpy as np
import torch
import torch.nn as nn
import tqdm
from torch.optim import Adam
import wandb

from utils import ndcg_k, recall_at_k


class Trainer:
    def __init__(
        self,
        model,
        train_dataloader,
        eval_dataloader,
        test_dataloader,
        submission_dataloader,
        args,
    ):

        self.args = args
        self.cuda_condition = torch.cuda.is_available() and not self.args.no_cuda
        self.device = torch.device("cuda" if self.cuda_condition else "cpu")

        self.model = model
        if self.cuda_condition:
            self.model.cuda()

        # Setting the train and test data loader
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.test_dataloader = test_dataloader
        self.submission_dataloader = submission_dataloader

        # self.data_name = self.args.data_name
        betas = (self.args.adam_beta1, self.args.adam_beta2)
        self.optim = Adam(
            self.model.parameters(),
            lr=self.args.lr,
            betas=betas,
            weight_decay=self.args.weight_decay,
        )

        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))
        self.criterion = nn.BCELoss()

    def train(self, epoch):
        self.iteration(epoch, self.train_dataloader)

    def valid(self, epoch):
        return self.iteration(epoch, self.eval_dataloader, mode="valid")

    def test(self, epoch):
        return self.iteration(epoch, self.test_dataloader, mode="test")

    def submission(self, epoch):
        return self.iteration(epoch, self.submission_dataloader, mode="submission")

    def iteration(self, epoch, dataloader, mode="train"):
        raise NotImplementedError

    def get_full_sort_score(self, epoch, answers, pred_list):
        recall, ndcg = [], []
        for k in [5, 10]:
            recall.append(recall_at_k(answers, pred_list, k))
            ndcg.append(ndcg_k(answers, pred_list, k))
        post_fix = {
            "epoch": epoch,
            "RECALL@5": round(recall[0],4),
            "NDCG@5": round(ndcg[0],4),
            "RECALL@10": round(recall[1],4),
            "NDCG@10": round(ndcg[1],4),
        }
        print(post_fix)

        return [recall[0], ndcg[0], recall[1], ndcg[1]], post_fix

    def save(self, file_name):
        torch.save(self.model.cpu().state_dict(), file_name)
        self.model.to(self.device)

    def load(self, file_name):
        self.model.load_state_dict(torch.load(file_name))

    def cross_entropy(self, seq_out, pos_ids, neg_ids):
        # [batch seq_len hidden_size]
        pos_emb = self.model.item_embeddings(pos_ids)
        neg_emb = self.model.item_embeddings(neg_ids)
        # [batch*seq_len hidden_size]
        pos = pos_emb.view(-1, pos_emb.size(2))
        neg = neg_emb.view(-1, neg_emb.size(2))
        seq_emb = seq_out.view(-1, self.args.hidden_size)  # [batch*seq_len hidden_size]
        pos_logits = torch.sum(pos * seq_emb, -1)  # [batch*seq_len]
        neg_logits = torch.sum(neg * seq_emb, -1)
        istarget = (
            (pos_ids > 0).view(pos_ids.size(0) * self.model.args.max_seq_length).float()
        )  # [batch*seq_len]
        loss = torch.sum(
            -torch.log(torch.sigmoid(pos_logits) + 1e-24) * istarget
            - torch.log(1 - torch.sigmoid(neg_logits) + 1e-24) * istarget
        ) / torch.sum(istarget)

        return loss

    def predict_full(self, seq_out):
        # [item_num hidden_size]
        test_item_emb = self.model.item_embeddings.weight
        # [batch hidden_size ]
        rating_pred = torch.matmul(seq_out, test_item_emb.transpose(0, 1))
        return rating_pred


class PretrainTrainer(Trainer):
    def __init__(
        self,
        model,
        train_dataloader,
        eval_dataloader,
        test_dataloader,
        submission_dataloader,
        args,
    ):
        super(PretrainTrainer, self).__init__(
            model,
            train_dataloader,
            eval_dataloader,
            test_dataloader,
            submission_dataloader,
            args,
        )

    def pretrain(self, epoch, pretrain_dataloader):
        
        if self.args.sweep==False:
            wandb.watch(self.model, self.model.criterion, log="parameters", log_freq=self.args.log_freq)
            
        desc = (
            f"AAP-{self.args.aap_weight}-"
            f"MIP-{self.args.mip_weight}-"
            f"MAP-{self.args.map_weight}-"
            f"SP-{self.args.sp_weight}"
        )

        pretrain_data_iter = tqdm.tqdm(
            enumerate(pretrain_dataloader),
            desc=f"{self.args.model_name}-{self.args.data_name} Epoch:{epoch}",
            total=len(pretrain_dataloader),
            bar_format="{l_bar}{r_bar}",
        )

        self.model.train()
        aap_loss_avg = 0.0
        mip_loss_avg = 0.0
        map_loss_avg = 0.0
        sp_loss_avg = 0.0

        for i, batch in pretrain_data_iter:
            # 0. batch_data will be sent into the device(GPU or CPU)
            batch = tuple(t.to(self.device) for t in batch)
            (
                attributes,
                masked_item_sequence,
                pos_items,
                neg_items,
                masked_segment_sequence,
                pos_segment,
                neg_segment,
            ) = batch

            aap_loss, mip_loss, map_loss, sp_loss = self.model.pretrain(
                attributes,
                masked_item_sequence,
                pos_items,
                neg_items,
                masked_segment_sequence,
                pos_segment,
                neg_segment,
            )

            joint_loss = (
                self.args.aap_weight * aap_loss
                + self.args.mip_weight * mip_loss
                + self.args.map_weight * map_loss
                + self.args.sp_weight * sp_loss
            )

            self.optim.zero_grad()
            joint_loss.backward()
            self.optim.step()

            aap_loss_avg += aap_loss.item()
            mip_loss_avg += mip_loss.item()
            map_loss_avg += map_loss.item()
            sp_loss_avg += sp_loss.item()

        num = len(pretrain_data_iter) * self.args.pre_batch_size
        losses = {
            "epoch": epoch,
            "aap_loss_avg": aap_loss_avg / num,
            "mip_loss_avg": mip_loss_avg / num,
            "map_loss_avg": map_loss_avg / num,
            "sp_loss_avg": sp_loss_avg / num,
        }
        print(desc)
        print(str(losses))
        
        wandb.log(losses, step=epoch)
        
        return losses


class FinetuneTrainer(Trainer):
    def __init__(
        self,
        model,
        train_dataloader,
        eval_dataloader,
        test_dataloader,
        submission_dataloader,
        args,
    ):
        super(FinetuneTrainer, self).__init__(
            model,
            train_dataloader,
            eval_dataloader,
            test_dataloader,
            submission_dataloader,
            args,
        )

    def iteration(self, epoch, dataloader, mode="train"):
        
        if mode != "submission" and self.args.sweep==False:
            wandb.watch(self.model, self.cross_entropy, log="parameters", log_freq=self.args.log_freq)
            
        # Setting the tqdm progress bar
        
        rec_data_iter = tqdm.tqdm(
            enumerate(dataloader),
            desc="Recommendation EP_%s:%d" % (mode, epoch),
            total=len(dataloader),
            bar_format="{l_bar}{r_bar}",
        )
        if mode == "train":
            self.model.train()
            rec_avg_loss = 0.0
            rec_cur_loss = 0.0

            for i, batch in rec_data_iter:
                # 0. batch_data will be sent into the device(GPU or CPU)
                batch = tuple(t.to(self.device) for t in batch)
                _, input_ids, target_pos, target_neg, _ = batch
                # Binary cross_entropy
                sequence_output = self.model.finetune(input_ids)
                loss = self.cross_entropy(sequence_output, target_pos, target_neg)
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                rec_avg_loss += loss.item()
                rec_cur_loss = loss.item()
            
            rec_avg_loss_per_len = rec_avg_loss / len(rec_data_iter)
            post_fix = {
                "epoch": epoch,
                "rec_avg_loss": round(rec_avg_loss_per_len, 4),
                "rec_cur_loss": round(rec_cur_loss, 4),
            }
            wandb.log(post_fix, step=epoch)

            if (epoch + 1) % self.args.log_freq == 0:
                print(str(post_fix))

        else:
            self.model.eval()

            pred_list = None
            answer_list = None
            for i, batch in rec_data_iter:

                batch = tuple(t.to(self.device) for t in batch)
                user_ids, input_ids, _, target_neg, answers = batch
                recommend_output = self.model.finetune(input_ids)

                recommend_output = recommend_output[:, -1, :]

                rating_pred = self.predict_full(recommend_output)

                rating_pred = rating_pred.cpu().data.numpy().copy()
                batch_user_index = user_ids.cpu().numpy()
                rating_pred[self.args.train_matrix[batch_user_index].toarray() > 0] = 0

                ind = np.argpartition(rating_pred, -10)[:, -10:]

                arr_ind = rating_pred[np.arange(len(rating_pred))[:, None], ind]

                arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(rating_pred)), ::-1]

                batch_pred_list = ind[
                    np.arange(len(rating_pred))[:, None], arr_ind_argsort
                ]

                if i == 0:
                    pred_list = batch_pred_list
                    answer_list = answers.cpu().data.numpy()
                else:
                    pred_list = np.append(pred_list, batch_pred_list, axis=0)
                    answer_list = np.append(
                        answer_list, answers.cpu().data.numpy(), axis=0
                    )

            if mode == "submission":
                return pred_list
            else:
                score, metrics = self.get_full_sort_score(epoch, answer_list, pred_list)
                wandb.log(metrics, step=epoch)
                return score, metrics
            

class Bert4RecTrainer(Trainer):
    def __init__(
        self,
        model,
        train_dataloader,
        eval_dataloader,
        test_dataloader,
        submission_dataloader,
        args,
        elem,

    ):
        super(Bert4RecTrainer, self).__init__(
            model,
            train_dataloader,
            eval_dataloader,
            test_dataloader,
            submission_dataloader,
            args,
        )
        
        self.criterion = args.criterion
        self.model = model
        self.train_dataloader = train_dataloader
        self.eval_dataloader  = eval_dataloader
        self.test_dataloader  = test_dataloader
        self.submission_dataloader  = submission_dataloader
        self.args  = args
        self.elem = elem
        

    def iteration(self, epoch, dataloader, mode="train"):
        
        if mode != "submission" and self.args.sweep==False:
            wandb.watch(self.model, self.cross_entropy, log="parameters", log_freq=self.args.log_freq)
            
        # Setting the tqdm progress bar
        
        rec_data_iter = tqdm.tqdm(
            enumerate(dataloader),
            desc="Recommendation EP_%s:%d" % (mode, epoch),
            total=len(dataloader),
            bar_format="{l_bar}{r_bar}",
        )
        if mode == "train":
            self.model.train()
            rec_avg_loss = 0.0
            rec_cur_loss = 0.0
            valid_loss = 0
            masked_cnt = 0
            correct_cnt = 0

            # batch : log sequence and labels 
            for i, batch in rec_data_iter:
                # 0. batch_data will be sent into the device(GPU or CPU)
                user_ids,log_seqs, labels = batch
                 # size matching
                logits = self.model.get_result(log_seqs,self.device)

                logits = logits.view(-1, logits.size(-1))   # [51200, 6808]
                labels = labels.view(-1).to(self.device)    # 51200
                self.optim.zero_grad()
                loss = self.criterion(logits, labels)
                loss.backward()
                self.optim.step()

                rec_avg_loss += loss.item()
                rec_cur_loss = loss.item()
                
            rec_avg_loss_per_len = rec_avg_loss / len(rec_data_iter)
            post_fix = {
                "epoch": epoch,
                "rec_avg_loss": round(rec_avg_loss_per_len, 4),
                "rec_cur_loss": round(rec_cur_loss, 4),
            }
            wandb.log(post_fix, step=epoch)

            if (epoch + 1) % self.args.log_freq == 0:
                print(str(post_fix))

        else:
            self.model.eval()
            valid_loss = 0
            masked_cnt = 0
            correct_cnt = 0

            pred_list = None
            answer_list = None
            for i, batch in rec_data_iter:
                user_ids,log_seqs, labels = batch
                logits = self.model.get_result(log_seqs,self.device)
                
                # 값이 낮은 순으로 index 를 뽑는다.
                """
                예를 들어 값이 [[8,1,7],[1,2,3]] 이면 
                array.argsort() = [[1,2,0],[0,1,2]] 이고 
                array.argsort().argsort() = [[2,0,1],[0,1,2]]
                """
                # recommend_output = logits[:,:].argsort()[:,:,-1].view(-1)
                rating_pred = logits[:, -1, :] # ( 256, 6808 )매 batch 마지막 sequence 에 대한 user 의 predict 값 
                rating_pred = rating_pred.cpu().data.numpy().copy()
                
                # 이미 본 영화를 제거하려는 의도
                """ 
                tmp = np.array([0])
                tmp.resize(rating_pred.shape[0],1)
                rating_pred = np.append(rating_pred,tmp, axis=1)
                batch_user_index = user_ids.cpu().numpy()
                rating_pred[self.elem.train_matrix[batch_user_index].toarray() > 0] = 0
                """ 

                seq_np = log_seqs.cpu().data.numpy()[:,-1].copy()
                filter_arr = (seq_np != 6808)&(seq_np != 0)
                seq_np = seq_np[filter_arr]
                rating_pred[:,seq_np] = -np.inf # 이미 시청한 영화 제거

                # -10을 기준으로 정렬해서 큰 것 부터 10 개의 index  
                ind = np.argpartition(rating_pred, -10)[:, -10:]
                # -10을 기준으로 정렬해서 큰 것 부터 10 개의 값 
                arr_ind = rating_pred[np.arange(len(rating_pred))[:, None], ind]

                # 각 index 의 순위 ( 값이 작은 것부터 0 )! 
                arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(rating_pred)), ::-1]

                batch_pred_list = ind[
                    np.arange(len(rating_pred))[:, None], arr_ind_argsort
                ]

                if i == 0:
                    pred_list = batch_pred_list
                    answer_list = labels.cpu().data.numpy()
                else:
                    pred_list = np.append(pred_list, batch_pred_list, axis=0)
                    answer_list = np.append(
                        answer_list, labels.cpu().data.numpy(), axis=0
                    )
                
            # new = []
            # for a in pred_list : 
            #     new.append(self.args.item2idx[self.args.item2idx.isin(a.tolist())].index.values)

            if mode == "submission":
                return pred_list
            else:
                score, metrics = self.get_full_sort_score(epoch, answer_list, pred_list)
                wandb.log(metrics, step=epoch)
                return score, metrics

        # 맨 마지막 시점에서 item 10개 추천
        def last_time(self,key, data, result) :
            # key = user_index, data = input data(sequence), result = model output(probability)
            t = result[-1] # dim = [1, 6808]
            t[data[0]] = -np.inf # 이미 시청한 영화 제거
            top_k_idx = np.argpartition(t, -10)[-10:] # top 10 proability 계산
            rec_item_id = item_id[top_k_idx] # 영화 추출
            user = user_id[key]
            for item in rec_item_id :
                final.append((user, item))
                