# -*- coding: utf-8 -*-
"""
UNEM: Unrolled Network for Expectation-Maximization
A pure visual metric learning model for few-shot classification.

Realistic transductive setting only.
Optimized for batch processing with episode_size > 1.

This implementation aligns with U_PADDLE_EMBT_L from UNEM-Transductive repository.
"""
import torch
import torch.nn.functional as F
from torch import nn

from core.utils import accuracy
from .metric_model import MetricModel


class UNEM(MetricModel):
    def __init__(self, L=10, diff_gamma_layers=True, **kwargs):
        super(UNEM, self).__init__(**kwargs)
        self.L = L
        self.diff_gamma_layers = diff_gamma_layers

        self.t1 = nn.Parameter(torch.ones(1))

        if not diff_gamma_layers:
            self.gamma = nn.Parameter(torch.ones(1))
            self.t2 = nn.Parameter(torch.ones(1))
        else:
            self.gamma = nn.ParameterList([nn.Parameter(torch.ones(1)) for _ in range(self.L)])
            self.t2 = nn.ParameterList([nn.Parameter(torch.ones(1)) for _ in range(self.L)])

        for param in self.emb_func.parameters():
            param.requires_grad = False

        self.loss_func = nn.CrossEntropyLoss()

        config = kwargs.get("config", {})
        rt_config = config.get("realistic_transductive", {})
        self.k_way = rt_config.get("k_way", 20)
        self.k_eff = rt_config.get("k_eff", 5)
        self.n_shot = rt_config.get("n_shot", 5)
        self.q_total = rt_config.get("q_total", 75)

    def get_one_hot(self, y_s, n_ways):
        eye = torch.eye(n_ways).to(y_s.device)
        one_hot = []
        for y_task in y_s:
            one_hot.append(eye[y_task].unsqueeze(0))
        one_hot = torch.cat(one_hot, 0)
        return one_hot

    def init_w(self, support, y_s, n_ways):
        n_tasks = support.size(0)
        one_hot = self.get_one_hot(y_s, n_ways)
        counts = one_hot.sum(1).view(n_tasks, -1, 1)
        weights = one_hot.transpose(1, 2).bmm(support)
        return weights / counts

    def split_realistic_batch(self, feat, global_target, k_way, n_shot, q_total):
        episode_size = feat.size(0)
        support_size = k_way * n_shot
        
        support_feat = feat[:, :support_size, :]
        query_feat = feat[:, support_size:, :]
        support_target = global_target[:, :support_size]
        query_target = global_target[:, support_size:]
        
        return support_feat, query_feat, support_target, query_target

    def set_forward(self, batch):
        image_or_feat, global_target = batch
        image_or_feat = image_or_feat.to(self.device)

        if image_or_feat.dim() == 4:
            with torch.no_grad():
                feat = self.emb_func(image_or_feat)
                feat = feat.view(feat.size(0), -1)
        else:
            feat = image_or_feat

        return self._set_forward_realistic(feat, global_target)

    def _set_forward_realistic(self, feat, global_target):
        if feat.dim() == 2:
            feat = feat.unsqueeze(0)
            global_target = global_target.unsqueeze(0) if global_target.dim() == 2 else global_target.unsqueeze(0)
        
        episode_size = feat.size(0)
        total_samples = feat.size(1)
        q_total = self.q_total if hasattr(self, 'q_total') else 75
        support_size_estimate = total_samples - q_total
        n_shot = self.n_shot if hasattr(self, 'n_shot') else 5
        k_way = support_size_estimate // n_shot
        
        support_feat, query_feat, support_target, query_target = self.split_realistic_batch(
            feat, global_target, k_way, n_shot, q_total
        )

        support_feat = support_feat * nn.Softplus()(self.t1)
        query_feat = query_feat * nn.Softplus()(self.t1)
        
        support_classes = support_target.view(episode_size, k_way, n_shot)[:, :, 0]
        
        support_local_target = (support_target.unsqueeze(2) == support_classes.unsqueeze(1)).long().argmax(dim=2)
        
        U_support = self.get_one_hot(support_local_target, k_way)

        n_query = query_feat.size(1)
        n_tasks = query_feat.size(0)

        W = self.init_w(support_feat, support_local_target, k_way)

        V = torch.zeros(n_tasks, k_way).to(self.device) + 1

        for i in range(self.L):
            if self.diff_gamma_layers:
                gamma = self.gamma[i]
                t2 = self.t2[i]
            else:
                gamma = self.gamma
                t2 = self.t2

            logits = (query_feat.bmm(W.transpose(1, 2)) 
                    - 1 / 2 * (W**2).sum(2).view(n_tasks, 1, -1) 
                    - 1 / 2 * (query_feat**2).sum(2).view(n_tasks, -1, 1))

            U_new = ((1/(nn.Softplus()(t2)+1)) * (logits + nn.Softplus()(gamma) * (V.unsqueeze(1).repeat(1, n_query, 1)))).softmax(2)
            
            V = torch.log(U_new.sum(1) / n_query + 1e-6) + 1

            num = torch.einsum('bkq,bqd->bkd', torch.transpose(U_new, 1, 2), query_feat) \
                    + torch.einsum('bkq,bqd->bkd', torch.transpose(U_support, 1, 2), support_feat)
            den = U_new.sum(1) + U_support.sum(1)
            
            W = torch.div(num, den.unsqueeze(2))
        
        output = U_new

        query_target = query_target.to(self.device)

        query_local_target = (query_target.unsqueeze(-1) == support_classes.unsqueeze(1)).long().argmax(dim=-1)
        
        output_flat = output.view(episode_size * n_query, k_way)
        query_local_target_flat = query_local_target.view(-1)
        
        acc = accuracy(output_flat, query_local_target_flat)

        return output_flat, acc

    def set_forward_loss(self, batch):
        image_or_feat, global_target = batch
        image_or_feat = image_or_feat.to(self.device)

        if image_or_feat.dim() == 4:
            with torch.no_grad():
                feat = self.emb_func(image_or_feat)
                feat = feat.view(feat.size(0), -1)
        else:
            feat = image_or_feat

        return self._set_forward_loss_realistic(feat, global_target)

    def _set_forward_loss_realistic(self, feat, global_target):
        if feat.dim() == 2:
            feat = feat.unsqueeze(0)
            global_target = global_target.unsqueeze(0) if global_target.dim() == 2 else global_target.unsqueeze(0)

        episode_size = feat.size(0)
        total_samples = feat.size(1)
        q_total = self.q_total if hasattr(self, 'q_total') else 75
        support_size_estimate = total_samples - q_total
        n_shot = self.n_shot if hasattr(self, 'n_shot') else 5
        k_way = support_size_estimate // n_shot

        support_feat, query_feat, support_target, query_target = self.split_realistic_batch(
            feat, global_target, k_way, n_shot, q_total
        )

        support_feat = support_feat * nn.Softplus()(self.t1)
        query_feat = query_feat * nn.Softplus()(self.t1)
        
        support_classes = support_target.view(episode_size, k_way, n_shot)[:, :, 0]
        
        support_local_target = (support_target.unsqueeze(2) == support_classes.unsqueeze(1)).long().argmax(dim=2)
        
        U_support = self.get_one_hot(support_local_target, k_way)

        n_query = query_feat.size(1)
        n_tasks = query_feat.size(0)

        W = self.init_w(support_feat, support_local_target, k_way)

        V = torch.zeros(n_tasks, k_way).to(self.device) + 1

        for i in range(self.L):
            if self.diff_gamma_layers:
                gamma = self.gamma[i]
                t2 = self.t2[i]
            else:
                gamma = self.gamma
                t2 = self.t2

            logits = (query_feat.bmm(W.transpose(1, 2)) 
                    - 1 / 2 * (W**2).sum(2).view(n_tasks, 1, -1) 
                    - 1 / 2 * (query_feat**2).sum(2).view(n_tasks, -1, 1))

            U_new = ((1/(nn.Softplus()(t2)+1)) * (logits + nn.Softplus()(gamma) * (V.unsqueeze(1).repeat(1, n_query, 1)))).softmax(2)
            
            V = torch.log(U_new.sum(1) / n_query + 1e-6) + 1

            num = torch.einsum('bkq,bqd->bkd', torch.transpose(U_new, 1, 2), query_feat) \
                    + torch.einsum('bkq,bqd->bkd', torch.transpose(U_support, 1, 2), support_feat)
            den = U_new.sum(1) + U_support.sum(1)
            
            W = torch.div(num, den.unsqueeze(2))

        output = U_new

        query_target = query_target.to(self.device)

        query_local_target = (query_target.unsqueeze(-1) == support_classes.unsqueeze(1)).long().argmax(dim=-1)
        
        output_flat = output.view(episode_size * n_query, k_way)
        query_local_target_flat = query_local_target.view(-1)
        
        loss = self.loss_func(output_flat, query_local_target_flat)
        acc = accuracy(output_flat, query_local_target_flat)

        return output_flat, acc, loss
