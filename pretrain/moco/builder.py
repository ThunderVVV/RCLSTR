# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import re
import torch
import torch.nn as nn
from einops import rearrange
import numpy as np
import math
import random
# import ot


class MoCo(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, base_encoder, args, dim=128, K=65536, m=0.999, T=0.07, mlp=False):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(MoCo, self).__init__()

        self.K = K
        self.m = m
        self.T = T
        self.loss_setting = args.loss_setting
        self.multi_level_consistent = args.multi_level_consistent
        self.permutation = args.permutation
        self.permute_probability = args.permute_probability
        self.fw_consist = args.fw_consist

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = base_encoder(args, SelfSL_layer=args.SelfSL_layer)
        self.encoder_k = base_encoder(args, SelfSL_layer=args.SelfSL_layer)

        if mlp:  # hack: brute-force replacement
            dim_mlp = self.encoder_q.fc.weight.shape[1]
            self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.fc)
            self.encoder_k.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.fc)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient
        
        for name_q, param_q in self.encoder_q.named_parameters():
            if "Transformation" in name_q:
                param_q.requires_grad = False  # not update by gradient
        
        if self.permutation:
            self.permutation_block_count = self.encoder_q.permutation_block_count
            self.permutation_img_count = self.encoder_q.permutation_img_count

        self.frame_count = self.encoder_q.frame_count
        self.subword_count = self.encoder_q.subword_count
        self.word_count = self.encoder_q.word_count

        # create the queue
        select_subword_count = self.subword_count
        select_word_count = self.word_count
        select_frame_count = 26
        memory_size = args.memory_size
        self.frame_K = (memory_size // (select_frame_count * args.batch_size)) * (select_frame_count * args.batch_size)
        self.subword_K = (memory_size // (select_subword_count * args.batch_size)) * (select_subword_count * args.batch_size)
        self.word_K = (memory_size // (select_word_count * args.batch_size)) * (select_word_count * args.batch_size)        

        self.register_buffer("frame_queue", torch.randn(dim, self.frame_K))
        self.frame_queue = nn.functional.normalize(self.frame_queue, dim=0)
        self.register_buffer("frame_queue_ptr", torch.zeros(1, dtype=torch.long))

        self.register_buffer("subword_queue", torch.randn(dim, self.subword_K))
        self.subword_queue = nn.functional.normalize(self.subword_queue, dim=0)
        self.register_buffer("subword_queue_ptr", torch.zeros(1, dtype=torch.long))

        self.register_buffer("word_queue", torch.randn(dim, self.word_K))
        self.word_queue = nn.functional.normalize(self.word_queue, dim=0)
        self.register_buffer("word_queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for (name_q, param_q), (name_k, param_k) in zip(self.encoder_q.named_parameters(), self.encoder_k.named_parameters()):
            if not "Transformation" in name_q:
                param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, queue_name, keys):
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        if queue_name == "frame":
            ptr = int(self.frame_queue_ptr)
            assert self.frame_K % batch_size == 0  # for simplicity
            # replace the keys at ptr (dequeue and enqueue)
            self.frame_queue[:, ptr:ptr + batch_size] = keys.T
            ptr = (ptr + batch_size) % self.frame_K  # move pointer
            self.frame_queue_ptr[0] = ptr
        elif queue_name == "subword":
            ptr = int(self.subword_queue_ptr)
            assert self.subword_K % batch_size == 0  # for simplicity
            # replace the keys at ptr (dequeue and enqueue)
            self.subword_queue[:, ptr:ptr + batch_size] = keys.T
            ptr = (ptr + batch_size) % self.subword_K  # move pointer
            self.subword_queue_ptr[0] = ptr
        elif queue_name == "word":
            ptr = int(self.word_queue_ptr)
            assert self.word_K % batch_size == 0  # for simplicity
            # replace the keys at ptr (dequeue and enqueue)
            self.word_queue[:, ptr:ptr + batch_size] = keys.T
            ptr = (ptr + batch_size) % self.word_K  # move pointer
            self.word_queue_ptr[0] = ptr
        else:
            raise NotImplementedError

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]
    
    def do_permutation(self, img, permutation_index):
        permutation_count = self.permutation_block_count * self.permutation_img_count
        img = img.reshape(img.size(0) // self.permutation_img_count, self.permutation_img_count, img.size(1), img.size(2), img.size(3)) # (b/2, 2, c, h, w)
        img = img.permute(0, 2, 3, 1, 4)  # (b/2, c, h, 2, w)
        img = img.reshape(img.size(0), img.size(1), img.size(2), self.permutation_img_count * img.size(4)) # (b/2, c, h, 2*w)
        img = img.reshape(img.size(0), img.size(1), img.size(2), permutation_count, img.size(3) // permutation_count) # (b/2, c, h, 8, 2*w/8)
        img = img[:, :, :, permutation_index, :]

        img = img.reshape(img.size(0), img.size(1), img.size(2), permutation_count * img.size(4)) # (b/2, c, h, 2*w)
        img = img.reshape(img.size(0), img.size(1), img.size(2), self.permutation_img_count, img.size(3) // self.permutation_img_count)  # (b/2, c, h, 2, w)
        img = img.permute(0, 3, 1, 2, 4)  # (b/2, 2, c, h, w)
        img = img.reshape(img.size(0) * self.permutation_img_count, img.size(2), img.size(3), img.size(4)) # (b, c, h, w)

        return img

    def forward(self, im_q, im_k):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """

        frame_select_index = None
        
        if self.permutation:
            permutation_count = self.permutation_block_count * self.permutation_img_count
            permutation_index = np.random.permutation(permutation_count)
            permutated_im_q = self.do_permutation(im_q, permutation_index) # (b, c, h, w)

            permutated_restored_frame_q, permutated_restored_subword_q, permutated_restored_word_q = self.encoder_q(permutated_im_q, None, permutation_index)  # queries: NxTxC
            permutated_restored_frame_q = nn.functional.normalize(permutated_restored_frame_q, dim=-1)
            permutated_restored_subword_q = nn.functional.normalize(permutated_restored_subword_q, dim=-1)
            permutated_restored_word_q = nn.functional.normalize(permutated_restored_word_q, dim=-1)
        else:
            permutated_restored_frame_q = None
            permutated_restored_subword_q = None
            permutated_restored_word_q = None
            
        # compute query features
        frame_q, subword_q, word_q = self.encoder_q(im_q, frame_select_index, None)  # queries: NxTxC
        frame_q = nn.functional.normalize(frame_q, dim=-1)
        subword_q = nn.functional.normalize(subword_q, dim=-1)
        word_q = nn.functional.normalize(word_q, dim=-1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)

            frame_k, subword_k, word_k = self.encoder_k(im_k, frame_select_index, None)  # keys: NxTxC
            frame_k = nn.functional.normalize(frame_k, dim=-1)
            subword_k = nn.functional.normalize(subword_k, dim=-1)
            word_k = nn.functional.normalize(word_k, dim=-1)

            # undo shuffle
            frame_k = self._batch_unshuffle_ddp(frame_k, idx_unshuffle)
            subword_k = self._batch_unshuffle_ddp(subword_k, idx_unshuffle)
            word_k = self._batch_unshuffle_ddp(word_k, idx_unshuffle)
        if self.multi_level_consistent == "global2local":
            # frame to subword
            stride = math.floor(self.frame_count / self.subword_count)
            kernel = self.frame_count - (self.subword_count - 1) * stride
            frame_q_split = []
            for i in range(self.subword_count):
                frame_q_split.append(frame_q[:, stride * i: stride * i + kernel, :])
            frame_q_split = torch.stack(frame_q_split, dim=1)  # (b, subword_count, kernel, c)
            frame_q_split = frame_q_split.reshape(frame_q_split.size(0), -1, frame_q_split.size(-1))  # (b, t, c)
            subword_k_expand = subword_k.unsqueeze(2)
            subword_k_expand = subword_k_expand.repeat(1, 1, kernel, 1)
            subword_k_expand = subword_k_expand.reshape(subword_k_expand.size(0), -1, subword_k_expand.size(-1))  # (b, t, c)

            # subword to word
            subword_q_split = subword_q # (b, 4, c)
            word_k_expand = word_k.repeat(1, self.subword_count, 1) # (b, 4, c)

            if self.fw_consist:
                frame_q_fw = frame_q
                word_k_fw = word_k.repeat(1, frame_q_fw.size(1), 1) # (b, 26, c)
                return (*self.get_results(frame_q, frame_k, "frame", permutated_restored_frame_q, False),
                        *self.get_results(subword_q, subword_k, "subword", permutated_restored_subword_q, False),
                        *self.get_results(word_q, word_k, "word", permutated_restored_word_q, False),
                        *self.get_results(frame_q_split, subword_k_expand, "subword", None, True),
                        *self.get_results(subword_q_split, word_k_expand, "word", None, True),
                        *self.get_results(frame_q_fw, word_k_fw, "word", None, True)
                        )


            return (*self.get_results(frame_q, frame_k, "frame", permutated_restored_frame_q, False),
                    *self.get_results(subword_q, subword_k, "subword", permutated_restored_subword_q, False),
                    *self.get_results(word_q, word_k, "word", permutated_restored_word_q, False),
                    *self.get_results(frame_q_split, subword_k_expand, "subword", None, True),
                    *self.get_results(subword_q_split, word_k_expand, "word", None, True)
                   )

        return (*self.get_results(frame_q, frame_k, "frame", permutated_restored_frame_q, False),
                *self.get_results(subword_q, subword_k, "subword", permutated_restored_subword_q, False),
                *self.get_results(word_q, word_k, "word", permutated_restored_word_q, False),
                )
    
    def get_results(self, q, k, queue_name, permutated_restored_q, between_level):
        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1

        k_for_queue = k
        k_for_queue = k_for_queue.reshape(-1, k_for_queue.size(-1))
        k_for_queue = k_for_queue.contiguous()

        if permutated_restored_q != None:
            if self.permute_probability:
                if np.random.rand() < 0.5:
                    q = permutated_restored_q
            else:
                # cat in batch dim
                q = torch.cat((q, permutated_restored_q), dim=0)
                k = torch.cat((k, k), dim=0)

        q = q.reshape(-1, q.size(-1))  # (b, t, c) -> (bt, c)
        k = k.reshape(-1, k.size(-1))  # (b, t, c) -> (bt, c)

        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        t_sim = 0.04
        if queue_name == "frame":
            l_neg = torch.einsum('nc,ck->nk', [q, self.frame_queue.clone().detach()])

        elif queue_name == "subword":
            l_neg = torch.einsum('nc,ck->nk', [q, self.subword_queue.clone().detach()])

        elif queue_name == "word":
            l_neg = torch.einsum('nc,ck->nk', [q, self.word_queue.clone().detach()])

        else:
            raise NotImplementedError

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        # for consistent contrast
        if self.loss_setting == "consistent":
            t_con = 0.04
            similarity_q = l_neg.div(t_con).softmax(dim=-1)  # (n, k) , query similarity distribution
            if queue_name == "frame":
                similarity_p = torch.einsum('nc,ck->nk', [k, self.frame_queue.clone().detach()]).div(t_con).softmax(dim=-1) # (n, k) , positive similarity distribution
            elif queue_name == "subword":
                similarity_p = torch.einsum('nc,ck->nk', [k, self.subword_queue.clone().detach()]).div(t_con).softmax(dim=-1) # (n, k) , positive similarity distribution
            elif queue_name == "word":
                similarity_p = torch.einsum('nc,ck->nk', [k, self.word_queue.clone().detach()]).div(t_con).softmax(dim=-1) # (n, k) , positive similarity distribution
            else:
                raise NotImplementedError

        # dequeue and enqueue
        if between_level:
            pass
        else:
            self._dequeue_and_enqueue(queue_name, k_for_queue)

        if self.loss_setting == "consistent":

            return logits, labels, similarity_q, similarity_p

        return logits, labels



# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
