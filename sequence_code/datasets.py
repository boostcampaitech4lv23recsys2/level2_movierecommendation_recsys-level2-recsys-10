import random
import numpy as np

import torch
from torch.utils.data import Dataset

from utils import (
    EarlyStopping,
    check_path,
    get_item2attribute_json,
    get_user_seqs,
    set_seed,
    indexinfo,
    neg_sample,
    get_popular_items,
    eg_sample_from_popular_items,
    generate_item2idx
)

class PretrainDataset(Dataset):
    def __init__(self, args:dict, user_seq:list, long_sequence:list):
        self.args = args
        self.user_seq = user_seq
        self.long_sequence = long_sequence
        self.max_len = args.max_seq_length
        self.part_sequence = []
        self.split_sequence()
        item2idx_, idx2item_ = indexinfo.get_index_info()
        
        if args.neg_from_pop:
            self.popular_items = get_popular_items(args.data_file, item2idx_, args.neg_from_pop)

    def split_sequence(self):
        """ user 가 본 movie id list 단위로 train set 을 분리한다.  
        """
        for seq in self.user_seq:
            # arg 에서 지정한 max_len 만큼의 movie id 를 input 으로 사용한다. 
            input_ids = seq[-(self.max_len + 2) : -2]  # keeping same as train set
            # 예를 들어 [5225, 1046, 64034] 라면, [5225], [5225, 1046], [5225, 1046, 6404] 를 append
            for i in range(len(input_ids)):
                self.part_sequence.append(input_ids[: i + 1])

    def __len__(self):
        return len(self.part_sequence)

    def __getitem__(self, index):

        sequence = self.part_sequence[index]  # pos_items
        # sample neg item for every masked item
        masked_item_sequence = []
        neg_items = []
        # Masked Item Prediction
        item_set = set(sequence)
        # masked_item_sequence 와 neg_items 를 추출한다.  
        # 각 sequence 에서 마지막 item 을 제외하고 순회한다. 
        for item in sequence[:-1]:
            prob = random.random()
            if prob < self.args.mask_p:
                prob /= self.args.mask_p
                if prob < 0.8:
                    # arg 의 max_len + 1 이 추가된다. 
                    masked_item_sequence.append(self.args.mask_id)
                elif prob < 0.9:
                    masked_item_sequence.append(random.randint(1, self.args.item_size-1))
                else:
                    masked_item_sequence.append(item)
                
                if self.args.neg_from_pop:
                    # 현재 item_set 에 없는 item 을 추가
                    neg_items.append(neg_sample_from_popular_items(item_set, self.popular_items, self.max_len))
                else:
                    neg_items.append(neg_sample(item_set, self.args.item_size))

            else:
                masked_item_sequence.append(item)
                neg_items.append(item)

        # add mask at the last position
        masked_item_sequence.append(self.args.mask_id)
        if self.args.neg_from_pop:
            neg_items.append(neg_sample_from_popular_items(item_set, self.popular_items, self.max_len))
        else:
            neg_items.append(neg_sample(item_set, self.args.item_size))

        # Segment Prediction
        if len(sequence) < 2:
            masked_segment_sequence = sequence
            pos_segment = sequence
            neg_segment = sequence
        else:
            sample_length = random.randint(1, len(sequence) // 2)
            start_id = random.randint(0, len(sequence) - sample_length)
            neg_start_id = random.randint(0, len(self.long_sequence) - sample_length)
            pos_segment = sequence[start_id : start_id + sample_length]
            neg_segment = self.long_sequence[
                neg_start_id : neg_start_id + sample_length
            ]
            masked_segment_sequence = (
                sequence[:start_id]
                + [self.args.mask_id] * sample_length
                + sequence[start_id + sample_length :]
            )
            pos_segment = (
                [self.args.mask_id] * start_id
                + pos_segment
                + [self.args.mask_id] * (len(sequence) - (start_id + sample_length))
            )
            neg_segment = (
                [self.args.mask_id] * start_id
                + neg_segment
                + [self.args.mask_id] * (len(sequence) - (start_id + sample_length))
            )

        assert len(masked_segment_sequence) == len(sequence)
        assert len(pos_segment) == len(sequence)
        assert len(neg_segment) == len(sequence)

        # padding sequence
        pad_len = self.max_len - len(sequence)
        masked_item_sequence = [0] * pad_len + masked_item_sequence
        pos_items = [0] * pad_len + sequence
        neg_items = [0] * pad_len + neg_items
        masked_segment_sequence = [0] * pad_len + masked_segment_sequence
        pos_segment = [0] * pad_len + pos_segment
        neg_segment = [0] * pad_len + neg_segment

        masked_item_sequence = masked_item_sequence[-self.max_len :]
        pos_items = pos_items[-self.max_len :]
        neg_items = neg_items[-self.max_len :]

        masked_segment_sequence = masked_segment_sequence[-self.max_len :]
        pos_segment = pos_segment[-self.max_len :]
        neg_segment = neg_segment[-self.max_len :]

        # Associated Attribute Prediction
        # Masked Attribute Prediction
        attributes = []
        for item in pos_items:
            attribute = [0] * self.args.attribute_size
            try:
                now_attribute = self.item2attribute[str(item)]
                for a in now_attribute:
                    attribute[a] = 1
            except:
                pass
            attributes.append(attribute)

        assert len(attributes) == self.max_len
        assert len(masked_item_sequence) == self.max_len
        assert len(pos_items) == self.max_len
        assert len(neg_items) == self.max_len
        assert len(masked_segment_sequence) == self.max_len
        assert len(pos_segment) == self.max_len
        assert len(neg_segment) == self.max_len

        cur_tensors = (
            torch.tensor(attributes, dtype=torch.long),
            torch.tensor(masked_item_sequence, dtype=torch.long),
            torch.tensor(pos_items, dtype=torch.long),
            torch.tensor(neg_items, dtype=torch.long),
            torch.tensor(masked_segment_sequence, dtype=torch.long),
            torch.tensor(pos_segment, dtype=torch.long),
            torch.tensor(neg_segment, dtype=torch.long),
        )
        return cur_tensors


class SASRecDataset(Dataset):
    def __init__(self, args, elem, user_seq, test_neg_items=None, data_type="train"):
        self.args = args
        self.elem = elem

        self.user_seq = user_seq
        self.test_neg_items = test_neg_items
        self.data_type = data_type
        self.max_len = args.max_seq_length
        item2idx_, idx2item_ = indexinfo.get_index_info()
        
        if self.args.neg_from_pop:
            self.popular_items = get_popular_items(self.args.data_file, item2idx_, args.neg_from_pop)

    def __getitem__(self, index):

        user_id = index
        items = self.user_seq[index]

        assert self.data_type in {"train", "valid", "test", "submission"}

        # [0, 1, 2, 3, 4, 5, 6]
        # train [0, 1, 2, 3]
        # target [1, 2, 3, 4]

        # valid [0, 1, 2, 3, 4]
        # answer [5]

        # test [0, 1, 2, 3, 4, 5]
        # answer [6]

        # submission [0, 1, 2, 3, 4, 5, 6]
        # answer None

        if self.data_type == "train":
            input_ids = items[:-3]
            target_pos = items[1:-2]
            answer = [0]  # no use

        elif self.data_type == "valid":
            input_ids = items[:-2]
            target_pos = items[1:-1]
            answer = [items[-2]]

        elif self.data_type == "test":
            input_ids = items[:-1]
            target_pos = items[1:]
            answer = [items[-1]]
        else:
            input_ids = items[:]
            target_pos = items[:]  # will not be used
            answer = []

        target_neg = []
        seq_set = set(items)
        for _ in input_ids:
            if self.args.neg_from_pop:
                target_neg.append(neg_sample_from_popular_items(seq_set, self.popular_items, self.max_len))
            else:
                target_neg.append(neg_sample(seq_set, self.elem.item_size))

        pad_len = self.max_len - len(input_ids)
        input_ids = [0] * pad_len + input_ids
        target_pos = [0] * pad_len + target_pos
        target_neg = [0] * pad_len + target_neg

        input_ids = input_ids[-self.max_len :]
        target_pos = target_pos[-self.max_len :]
        target_neg = target_neg[-self.max_len :]

        assert len(input_ids) == self.max_len
        assert len(target_pos) == self.max_len
        assert len(target_neg) == self.max_len

        if self.test_neg_items is not None:
            test_samples = self.test_neg_items[index]

            cur_tensors = (
                torch.tensor(user_id, dtype=torch.long),  # user_id for testing
                torch.tensor(input_ids, dtype=torch.long),
                torch.tensor(target_pos, dtype=torch.long),
                torch.tensor(target_neg, dtype=torch.long),
                torch.tensor(answer, dtype=torch.long),
                torch.tensor(test_samples, dtype=torch.long),
            )
        else:
            cur_tensors = (
                torch.tensor(user_id, dtype=torch.long),  # user_id for testing
                torch.tensor(input_ids, dtype=torch.long),
                torch.tensor(target_pos, dtype=torch.long),
                torch.tensor(target_neg, dtype=torch.long),
                torch.tensor(answer, dtype=torch.long),
            )

        return cur_tensors

    def __len__(self):
        return len(self.user_seq)


class ClozeDataSet(Dataset):
    def __init__(self, user_seq,attr_seq, args, elem, is_submission:bool=False):
        self.user_seq  = user_seq
        self.num_user  = len(user_seq)
        self.num_item  = elem.num_item
        self.max_len   = args.max_len
        self.mask_prob = args.mask_prob
        self.is_submission = is_submission

        self.attr_seq = attr_seq


    def __getitem__(self, user): 
        # iterator를 구동할 때 사용됩니다.
        user_id = user

        seq = self.user_seq[user]
        attr_seq = self.attr_seq[user]
        tokens = []
        attrs = [] 
        labels = []

        if( "submission"== self.is_submission ):
            tokens = seq[:].copy()
            attrs = attr_seq[:].copy()
            # labbels not use

        else :
            seq = seq[:-1]
            attrs = attr_seq[:-1]
            for s in seq:
                prob = np.random.rand() 
                if prob < self.mask_prob:
                    prob /= self.mask_prob

                    # BERT 학습
                    # random 하게 80% 를 mask token 으로 변환 
                    if prob < 0.8:
                        # masking
                        tokens.append(self.num_item + 1)  # mask_index: num_item + 1, 0: pad, 1~num_item: item index
                    # random 하게 10% 를 random token 으로 변환 
                    elif prob < 0.9:
                        tokens.append(np.random.randint(1, self.num_item+1))  # item random sampling
                    else:
                    # 나머지 10% 를 original token 으로 사용
                        tokens.append(s)
                    labels.append(s)  # 학습에 사용
                else:
                    tokens.append(s)
                    labels.append(0)  # 학습에 사용 X, trivial

        tokens = tokens[-self.max_len:]
        attrs  = attrs[-self.max_len:]
        labels = labels[-self.max_len:]
        mask_len = self.max_len - len(tokens)

        # zero padding
        tokens = [0] * mask_len + tokens
        attrs  = [0] * mask_len + attrs
        labels = [0] * mask_len + labels
        return torch.tensor(user_id, dtype=torch.long),torch.LongTensor(tokens), torch.LongTensor(labels),torch.LongTensor(attrs)
    
    def __len__(self):
        # 총 user의 수 = 학습에 사용할 sequence의 수
        return self.num_user