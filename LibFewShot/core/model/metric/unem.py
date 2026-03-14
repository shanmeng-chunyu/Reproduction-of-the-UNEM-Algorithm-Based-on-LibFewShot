# -*- coding: utf-8 -*-
"""
UNEM: Unrolled Network for Expectation-Maximization
A pure visual metric learning model for few-shot classification.

Supports both standard few-shot and realistic transductive settings.
"""
import torch
import torch.nn.functional as F
from torch import nn

from core.utils import accuracy
from .metric_model import MetricModel


class UNEM(MetricModel):
    def __init__(self, L=10, **kwargs):
        super(UNEM, self).__init__(**kwargs)
        self.L = L

        self.a = nn.Parameter(torch.zeros(L))
        self.b = nn.Parameter(torch.zeros(L))
        self.T_z = nn.Parameter(torch.ones(1))

        for param in self.emb_func.parameters():
            param.requires_grad = False

        self.loss_func = nn.CrossEntropyLoss()

        config = kwargs.get("config", {})
        self.realistic_mode = config.get("realistic_transductive", {}).get("enable", False)
        if self.realistic_mode:
            rt_config = config.get("realistic_transductive", {})
            self.k_way = rt_config.get("k_way", 20)
            self.k_eff = rt_config.get("k_eff", 5)
            self.n_shot = rt_config.get("n_shot", 5)
            self.q_total = rt_config.get("q_total", 75)

    def split_realistic(self, feat, global_target, k_way, n_shot):
        """
        Split features for realistic transductive setting.
        
        In realistic mode:
        - Support set: k_way classes, n_shot samples each
        - Query set: k_eff classes (subset), q_total samples total
        
        Args:
            feat: [total_samples, feat_dim]
            global_target: [1, total_samples]
            k_way: number of classes in support set
            n_shot: number of support samples per class
            
        Returns:
            support_feat: [k_way * n_shot, feat_dim]
            query_feat: [q_total, feat_dim]
            support_target: [k_way * n_shot]
            query_target: [q_total]
        """
        support_size = k_way * n_shot
        
        feat = feat.view(-1, feat.size(-1))
        global_target = global_target.view(-1)
        
        support_feat = feat[:support_size]
        query_feat = feat[support_size:]
        support_target = global_target[:support_size]
        query_target = global_target[support_size:]
        
        return support_feat, query_feat, support_target, query_target

    def set_forward(self, batch):
        image_or_feat, global_target = batch
        image_or_feat = image_or_feat.to(self.device)

        if image_or_feat.dim() == 4:
            with torch.no_grad():
                feat = self.emb_func(image_or_feat)
                feat = feat.view(feat.size(0), -1)
        else:
            feat = image_or_feat  # 直接使用传入的特征

        # feat = F.normalize(feat, p=2, dim=-1)
        # feat = feat * self.T_z

        if self.realistic_mode:
            total_samples = feat.size(0)
            q_total = self.q_total if hasattr(self, 'q_total') else 75
            support_size_estimate = total_samples - q_total
            n_shot = self.n_shot if hasattr(self, 'n_shot') else 5
            k_way = support_size_estimate // n_shot
            
            support_feat, query_feat, support_target, query_target = self.split_realistic(
                feat, global_target, k_way, n_shot
            )
            task_mean = torch.cat([support_feat, query_feat], dim=0).mean(dim=0, keepdim=True)
            support_feat = support_feat - task_mean
            query_feat = query_feat - task_mean
            support_feat = F.normalize(support_feat, p=2, dim=-1) * self.T_z
            query_feat = F.normalize(query_feat, p=2, dim=-1) * self.T_z
            
            Q_num = query_feat.size(0)
            way_num = k_way
            
            support_feat_reshaped = support_feat.view(k_way, n_shot, -1)
            theta = support_feat_reshaped.mean(dim=1)
            theta = theta.view(k_way, -1)

            support_sum = support_feat_reshaped.sum(dim=1)

            pi = torch.ones(k_way).to(self.device)

            logits = None
            T = 1.0

            for l in range(self.L):
                lam = F.softplus(self.a[l])
                T = 1.0 + F.softplus(self.b[l])

                dist_sq = torch.cdist(query_feat, theta) ** 2

                logits = -0.5 * dist_sq + lam * torch.log(pi + 1e-8)

                u = F.softmax(logits / T, dim=1)

                theta = ((u.T @ query_feat) + support_sum) / (u.sum(dim=0, keepdim=True).T + n_shot + 1e-8)
                theta = theta.view(k_way, -1)

                pi = u.mean(dim=0)

            output = logits / T
            
            support_classes = support_target.view(k_way, n_shot)[:, 0]
            query_local_target = torch.zeros(query_target.size(0), dtype=torch.long, device=self.device)
            for i in range(k_way):
                query_local_target[query_target.to(self.device) == support_classes[i].to(self.device)] = i
            
            acc = accuracy(output, query_local_target)

            return output, acc
        else:
            support_feat, query_feat, support_target, query_target = self.split_by_episode(
                feat, mode=1
            )

            task_mean = torch.cat([support_feat, query_feat], dim=1).mean(dim=1, keepdim=True)
            support_feat = support_feat - task_mean
            query_feat = query_feat - task_mean
            support_feat = F.normalize(support_feat, p=2, dim=-1) * self.T_z
            query_feat = F.normalize(query_feat, p=2, dim=-1) * self.T_z

            episode_size = support_feat.size(0)
            Q_num = query_feat.size(1)
            way_num = self.way_num

            support_feat_reshaped = support_feat.view(episode_size, way_num, self.shot_num, -1)
            theta = support_feat_reshaped.mean(dim=2)
            theta = theta.view(episode_size * way_num, -1)

            support_sum = support_feat_reshaped.sum(dim=2)

            pi = torch.ones(way_num).to(self.device)

            logits = None
            T = 1.0

            for l in range(self.L):
                lam = F.softplus(self.a[l])
                T = 1.0 + F.softplus(self.b[l])

                query_feat_flat = query_feat.view(episode_size * Q_num, -1)
                dist_sq = torch.cdist(query_feat_flat, theta) ** 2

                logits = -0.5 * dist_sq + lam * torch.log(pi + 1e-8)

                u = F.softmax(logits / T, dim=1)

                u_reshaped = u.view(episode_size, Q_num, way_num)
                query_feat_reshaped = query_feat.view(episode_size, Q_num, -1)

                theta = (torch.bmm(
                    u_reshaped.transpose(1, 2),
                    query_feat_reshaped
                ) + support_sum) / (u_reshaped.sum(dim=1, keepdim=True).transpose(1, 2) + self.shot_num + 1e-8)
                theta = theta.view(episode_size * way_num, -1)

                pi = u_reshaped.mean(dim=(0, 1))

            output = logits / T
            output = output.view(episode_size * Q_num, way_num)
            acc = accuracy(output, query_target.view(-1))

            return output, acc

    def set_forward_loss(self, batch):
        image_or_feat, global_target = batch
        image_or_feat = image_or_feat.to(self.device)

        if image_or_feat.dim() == 4:
            with torch.no_grad():
                feat = self.emb_func(image_or_feat)
                feat = feat.view(feat.size(0), -1)
        else:
            feat = image_or_feat  # 直接使用传入的特征

        # feat = F.normalize(feat, p=2, dim=-1)
        # feat = feat * self.T_z

        if self.realistic_mode:
            total_samples = feat.size(0)
            q_total = self.q_total if hasattr(self, 'q_total') else 75
            support_size_estimate = total_samples - q_total
            n_shot = self.n_shot if hasattr(self, 'n_shot') else 5
            k_way = support_size_estimate // n_shot
            
            support_feat, query_feat, support_target, query_target = self.split_realistic(
                feat, global_target, k_way, n_shot
            )

            task_mean = torch.cat([support_feat, query_feat], dim=0).mean(dim=0, keepdim=True)
            support_feat = support_feat - task_mean
            query_feat = query_feat - task_mean
            support_feat = F.normalize(support_feat, p=2, dim=-1) * self.T_z
            query_feat = F.normalize(query_feat, p=2, dim=-1) * self.T_z

            Q_num = query_feat.size(0)
            way_num = k_way
            
            support_feat_reshaped = support_feat.view(k_way, n_shot, -1)
            theta = support_feat_reshaped.mean(dim=1)
            theta = theta.view(k_way, -1)

            support_sum = support_feat_reshaped.sum(dim=1)

            pi = (torch.ones(k_way) / k_way).to(self.device)

            logits = None
            T = 1.0

            for l in range(self.L):
                lam = F.softplus(self.a[l])
                T = 1.0 + F.softplus(self.b[l])

                dist_sq = torch.cdist(query_feat, theta) ** 2

                logits = -0.5 * dist_sq + lam * torch.log(pi + 1e-8)

                u = F.softmax(logits / T, dim=1)

                theta = ((u.T @ query_feat) + support_sum) / (u.sum(dim=0, keepdim=True).T + n_shot + 1e-8)
                theta = theta.view(k_way, -1)

                pi = u.mean(dim=0)

            output = logits / T
            
            support_classes = support_target.view(k_way, n_shot)[:, 0]
            query_local_target = torch.zeros(query_target.size(0), dtype=torch.long, device=self.device)
            for i in range(k_way):
                query_local_target[query_target.to(self.device) == support_classes[i].to(self.device)] = i
            
            loss = self.loss_func(output, query_local_target)
            acc = accuracy(output, query_local_target)

            return output, acc, loss
        else:
            support_feat, query_feat, support_target, query_target = self.split_by_episode(
                feat, mode=1
            )

            task_mean = torch.cat([support_feat, query_feat], dim=1).mean(dim=1, keepdim=True)
            support_feat = support_feat - task_mean
            query_feat = query_feat - task_mean
            support_feat = F.normalize(support_feat, p=2, dim=-1) * self.T_z
            query_feat = F.normalize(query_feat, p=2, dim=-1) * self.T_z

            episode_size = support_feat.size(0)
            Q_num = query_feat.size(1)
            way_num = self.way_num

            support_feat_reshaped = support_feat.view(episode_size, way_num, self.shot_num, -1)
            theta = support_feat_reshaped.mean(dim=2)
            theta = theta.view(episode_size * way_num, -1)

            support_sum = support_feat_reshaped.sum(dim=2)

            pi = torch.ones(way_num).to(self.device)

            logits = None
            T = 1.0

            for l in range(self.L):
                lam = F.softplus(self.a[l])
                T = 1.0 + F.softplus(self.b[l])

                query_feat_flat = query_feat.view(episode_size * Q_num, -1)
                dist_sq = torch.cdist(query_feat_flat, theta) ** 2

                logits = -0.5 * dist_sq + lam * torch.log(pi + 1e-8)

                u = F.softmax(logits / T, dim=1)

                u_reshaped = u.view(episode_size, Q_num, way_num)
                query_feat_reshaped = query_feat.view(episode_size, Q_num, -1)

                theta = (torch.bmm(
                    u_reshaped.transpose(1, 2),
                    query_feat_reshaped
                ) + support_sum) / (u_reshaped.sum(dim=1, keepdim=True).transpose(1, 2) + self.shot_num + 1e-8)
                theta = theta.view(episode_size * way_num, -1)

                pi = u_reshaped.mean(dim=(0, 1))

            output = logits / T
            output = output.view(episode_size * Q_num, way_num)
            acc = accuracy(output, query_target.view(-1))
            loss = self.loss_func(output, query_target.view(-1))

            return output, acc, loss
