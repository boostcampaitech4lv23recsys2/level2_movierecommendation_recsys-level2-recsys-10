import torch
import torch.nn as nn

from modules import Encoder, LayerNorm

from modules import ScaledDotProductAttention, MultiHeadAttention, PositionwiseFeedForward, BERT4RecBlock

import numpy as np


class S3RecModel(nn.Module):
    # https://github.com/RUCAIBox/CIKM2020-S3Rec/blob/master/model.PNG
    def __init__(self, args,elem):
        super(S3RecModel, self).__init__()
        self.item_embeddings = nn.Embedding(
            elem.item_size, args.hidden_size, padding_idx=0
        )
        self.attribute_embeddings = nn.Embedding(
            elem.attribute_size, args.hidden_size, padding_idx=0
        )
        self.position_embeddings = nn.Embedding(args.max_seq_length, args.hidden_size)
        self.item_encoder = Encoder(args)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        self.args = args

        # add unique dense layer for 4 losses respectively
        self.aap_norm = nn.Linear(args.hidden_size, args.hidden_size)
        self.mip_norm = nn.Linear(args.hidden_size, args.hidden_size)
        self.map_norm = nn.Linear(args.hidden_size, args.hidden_size)
        self.sp_norm = nn.Linear(args.hidden_size, args.hidden_size)
        self.criterion = nn.BCELoss(reduction="none")
        self.apply(self.init_weights)

    # AAP
    def associated_attribute_prediction(self, sequence_output, attribute_embedding):
        """
        :param sequence_output: [B L H]
        :param attribute_embedding: [arribute_num H]
        :return: scores [B*L tag_num]
        """
        sequence_output = self.aap_norm(sequence_output)  # [B L H]
        sequence_output = sequence_output.view(
            [-1, self.args.hidden_size, 1]
        )  # [B*L H 1]
        # [tag_num H] [B*L H 1] -> [B*L tag_num 1]
        score = torch.matmul(attribute_embedding, sequence_output)
        return torch.sigmoid(score.squeeze(-1))  # [B*L tag_num]

    # MIP sample neg items
    def masked_item_prediction(self, sequence_output, target_item):
        """
        :param sequence_output: [B L H]
        :param target_item: [B L H]
        :return: scores [B*L]
        """
        sequence_output = self.mip_norm(
            sequence_output.view([-1, self.args.hidden_size])
        )  # [B*L H]
        target_item = target_item.view([-1, self.args.hidden_size])  # [B*L H]
        score = torch.mul(sequence_output, target_item)  # [B*L H]
        return torch.sigmoid(torch.sum(score, -1))  # [B*L]

    # MAP
    def masked_attribute_prediction(self, sequence_output, attribute_embedding):
        sequence_output = self.map_norm(sequence_output)  # [B L H]
        sequence_output = sequence_output.view(
            [-1, self.args.hidden_size, 1]
        )  # [B*L H 1]
        # [tag_num H] [B*L H 1] -> [B*L tag_num 1]
        score = torch.matmul(attribute_embedding, sequence_output)
        return torch.sigmoid(score.squeeze(-1))  # [B*L tag_num]

    # SP sample neg segment
    def segment_prediction(self, context, segment):
        """
        :param context: [B H]
        :param segment: [B H]
        :return:
        """
        context = self.sp_norm(context)
        score = torch.mul(context, segment)  # [B H]
        return torch.sigmoid(torch.sum(score, dim=-1))  # [B]

    #
    def add_position_embedding(self, sequence):

        seq_length = sequence.size(1)
        position_ids = torch.arange(
            seq_length, dtype=torch.long, device=sequence.device
        )
        position_ids = position_ids.unsqueeze(0).expand_as(sequence)
        item_embeddings = self.item_embeddings(sequence)
        position_embeddings = self.position_embeddings(position_ids)
        sequence_emb = item_embeddings + position_embeddings
        sequence_emb = self.LayerNorm(sequence_emb)
        sequence_emb = self.dropout(sequence_emb)

        return sequence_emb

    def pretrain(
        self,
        attributes,
        masked_item_sequence,
        pos_items,
        neg_items,
        masked_segment_sequence,
        pos_segment,
        neg_segment,
    ):

        # Encode masked sequence

        sequence_emb = self.add_position_embedding(masked_item_sequence)
        sequence_mask = (masked_item_sequence == 0).float() * -1e8
        sequence_mask = torch.unsqueeze(torch.unsqueeze(sequence_mask, 1), 1)

        encoded_layers = self.item_encoder(
            sequence_emb, sequence_mask, output_all_encoded_layers=True
        )
        # [B L H]
        sequence_output = encoded_layers[-1]

        attribute_embeddings = self.attribute_embeddings.weight
        # AAP
        aap_score = self.associated_attribute_prediction(
            sequence_output, attribute_embeddings
        )
        aap_loss = self.criterion(
            aap_score, attributes.view(-1, self.elem.attribute_size).float()
        )
        # only compute loss at non-masked position
        aap_mask = (masked_item_sequence != self.elem.mask_id).float() * (
            masked_item_sequence != 0
        ).float()
        aap_loss = torch.sum(aap_loss * aap_mask.flatten().unsqueeze(-1))

        # MIP
        pos_item_embs = self.item_embeddings(pos_items)
        neg_item_embs = self.item_embeddings(neg_items)
        pos_score = self.masked_item_prediction(sequence_output, pos_item_embs)
        neg_score = self.masked_item_prediction(sequence_output, neg_item_embs)
        mip_distance = torch.sigmoid(pos_score - neg_score)
        mip_loss = self.criterion(
            mip_distance, torch.ones_like(mip_distance, dtype=torch.float32)
        )
        mip_mask = (masked_item_sequence == self.elem.mask_id).float()
        mip_loss = torch.sum(mip_loss * mip_mask.flatten())

        # MAP
        map_score = self.masked_attribute_prediction(
            sequence_output, attribute_embeddings
        )
        map_loss = self.criterion(
            map_score, attributes.view(-1, self.elem.attribute_size).float()
        )
        map_mask = (masked_item_sequence == self.elem.mask_id).float()
        map_loss = torch.sum(map_loss * map_mask.flatten().unsqueeze(-1))

        # SP
        # segment context
        segment_context = self.add_position_embedding(masked_segment_sequence)
        segment_mask = (masked_segment_sequence == 0).float() * -1e8
        segment_mask = torch.unsqueeze(torch.unsqueeze(segment_mask, 1), 1)
        segment_encoded_layers = self.item_encoder(
            segment_context, segment_mask, output_all_encoded_layers=True
        )

        # take the last position hidden as the context
        segment_context = segment_encoded_layers[-1][:, -1, :]  # [B H]
        # pos_segment
        pos_segment_emb = self.add_position_embedding(pos_segment)
        pos_segment_mask = (pos_segment == 0).float() * -1e8
        pos_segment_mask = torch.unsqueeze(torch.unsqueeze(pos_segment_mask, 1), 1)
        pos_segment_encoded_layers = self.item_encoder(
            pos_segment_emb, pos_segment_mask, output_all_encoded_layers=True
        )
        pos_segment_emb = pos_segment_encoded_layers[-1][:, -1, :]

        # neg_segment
        neg_segment_emb = self.add_position_embedding(neg_segment)
        neg_segment_mask = (neg_segment == 0).float() * -1e8
        neg_segment_mask = torch.unsqueeze(torch.unsqueeze(neg_segment_mask, 1), 1)
        neg_segment_encoded_layers = self.item_encoder(
            neg_segment_emb, neg_segment_mask, output_all_encoded_layers=True
        )
        neg_segment_emb = neg_segment_encoded_layers[-1][:, -1, :]  # [B H]

        pos_segment_score = self.segment_prediction(segment_context, pos_segment_emb)
        neg_segment_score = self.segment_prediction(segment_context, neg_segment_emb)

        sp_distance = torch.sigmoid(pos_segment_score - neg_segment_score)

        sp_loss = torch.sum(
            self.criterion(
                sp_distance, torch.ones_like(sp_distance, dtype=torch.float32)
            )
        )

        return aap_loss, mip_loss, map_loss, sp_loss

    # Fine tune
    # same as SASRec
    def finetune(self, input_ids):

        attention_mask = (input_ids > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(
            2
        )  # torch.int64
        max_len = attention_mask.size(-1)
        attn_shape = (1, max_len, max_len)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)  # torch.uint8
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
        subsequent_mask = subsequent_mask.long()

        if self.args.cuda_condition:
            subsequent_mask = subsequent_mask.cuda()

        extended_attention_mask = extended_attention_mask * subsequent_mask
        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype
        )  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        sequence_emb = self.add_position_embedding(input_ids)

        item_encoded_layers = self.item_encoder(
            sequence_emb, extended_attention_mask, output_all_encoded_layers=True
        )

        sequence_output = item_encoded_layers[-1]
        return sequence_output

    def init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.args.initializer_range)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
            

class BERT4RecModel(nn.Module):
    def __init__(self, args, elem):
        super(BERT4RecModel, self).__init__()

        self.args = args
        self.num_user = elem.num_user
        self.num_item = elem.num_item
        self.hidden_units = args.hidden_units
        self.num_heads = args.num_heads
        self.num_layers = args.num_layers 
        
        self.item_emb = nn.Embedding( self.num_item + 2, self.args.max_len, padding_idx = 0  )  
        self.attr_emb = nn.Embedding(2+1, self.args.max_len, padding_idx = 0  )  
        self.pos_emb = nn.Embedding( self.args.max_len, self.hidden_units) # learnable positional encoding
        self.dropout = nn.Dropout(self.args.dropout_rate)
        self.emb_layernorm = nn.LayerNorm(self.args.hidden_units, eps=1e-6)
        
        self.blocks = nn.ModuleList([BERT4RecBlock(self.args.num_heads, self.args.hidden_units, self.args.dropout_rate) for _ in range(self.num_layers)])
        self.out = nn.Linear(self.hidden_units, self.num_item + 1 )  # TODO3: 예측을 위한 output layer를 구현해보세요. (num_item 주의)
        
        torch.nn.init.xavier_uniform_(self.item_emb.weight)
        torch.nn.init.xavier_uniform_(self.attr_emb.weight)
        torch.nn.init.xavier_uniform_(self.pos_emb.weight)

    def get_result(self, log_seqs,attr_seqs, device):
        seqs = self.item_emb(torch.LongTensor(log_seqs).to(device))
        seqs += self.attr_emb(torch.LongTensor(attr_seqs).to(device))
        # seqs += self.attr_emb(torch.LongTensor(attr_seqs).to(device))
        positions = np.tile(np.array(range(log_seqs.shape[1])), [log_seqs.shape[0], 1])
        """ position
        [[ 0  1  2 ... 47 48 49]
         [ 0  1  2 ... 47 48 49]
         [ 0  1  2 ... 47 48 49]
         ...
         [ 0  1  2 ... 47 48 49]
         [ 0  1  2 ... 47 48 49]
         [ 0  1  2 ... 47 48 49]]
        """
        seqs += self.pos_emb(torch.LongTensor(positions).to(device))
        seqs = self.emb_layernorm(self.dropout(seqs))
        
        """
        ( 1 ) `torch.BoolTensor(log_seqs > 0)` 
         - log_seqs 값이 0 보다 크면 True , 작으면 False 인 Tensor 를 생성
         - Torch.Size([128, 50])
        ( 2 ) `unsqueeze(1)` 
         - (batch_size, 1, sequence_length)
         - Torch.Size([128, 1, 50])

        ( 3 ) `repeat(1, log_seqs.shape[1], 1)`
         - (batch_size, sequence_length, sequence_length)
         - Torch.Size([128, 50, 50])

        ( 4 ) unsqueeze(1)
         - (batch_size, 1, sequence_length, sequence_length)
         - Torch.Size([128, 1, 50, 50])

        """
        
        mask = torch.BoolTensor(log_seqs > 0).unsqueeze(1).repeat(1, log_seqs.shape[1], 1).unsqueeze(1).to(device) # mask for zero pad
        for block in self.blocks:
            seqs, attn_dist = block(seqs, mask)
        out = self.out(seqs)
        return out