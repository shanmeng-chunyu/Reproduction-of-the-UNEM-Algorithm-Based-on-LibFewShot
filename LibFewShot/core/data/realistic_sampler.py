# -*- coding: utf-8 -*-
"""
Realistic Transductive Few-Shot Sampler

This module implements a custom sampler for the "Realistic Transductive Few-Shot" 
evaluation protocol, where the query set contains only a subset of the support set's 
classes (K_eff < K). The extra classes in the support set serve as distractors.

Author: LibFewShot Contributor
"""

import numpy as np
import torch
from torch.utils.data import Sampler


class RealisticTransductiveSampler(Sampler):
    """
    A Sampler for Realistic Transductive Few-Shot Learning.
    
    In this setting:
    - Support set contains K classes (some are distractors)
    - Query set contains only K_eff classes (K_eff < K), which are a subset of support classes
    
    Args:
        label (list or np.ndarray): Label list of all samples in the dataset.
        n_batch (int): Number of episodes/batches to generate.
        k_way (int): Total number of classes in the support set (K).
        n_shot (int): Number of support samples per class (s).
        k_eff (int): Number of effective classes in the query set (K_eff).
        q_total (int): Total number of query samples (|Q|).
    """

    def __init__(
        self,
        label,
        n_batch,
        k_way,
        n_shot,
        k_eff,
        q_total,
    ):
        super(RealisticTransductiveSampler, self).__init__()
        
        self.n_batch = n_batch
        self.k_way = k_way
        self.n_shot = n_shot
        self.k_eff = k_eff
        self.q_total = q_total

        label = np.array(label)
        self.unique_labels = np.unique(label)
        self.num_classes = len(self.unique_labels)

        self.idx_list = []
        for label_idx in self.unique_labels:
            ind = np.argwhere(label == label_idx).reshape(-1)
            ind = torch.from_numpy(ind)
            self.idx_list.append(ind)

        if self.k_way > self.num_classes:
            raise ValueError(
                f"k_way ({k_way}) cannot exceed total number of classes ({self.num_classes})"
            )
        if self.k_eff > self.k_way:
            raise ValueError(
                f"k_eff ({k_eff}) cannot exceed k_way ({k_way})"
            )

    def __len__(self):
        return self.n_batch

    def __iter__(self):
        """
        Generate episodes for realistic transductive few-shot learning.
        
        Each episode contains:
        - Support set: k_way * n_shot samples from k_way classes
        - Query set: q_total samples from k_eff classes (subset of support classes)
        
        Yields:
            torch.Tensor: Indices of samples for one episode, with shape 
                         ((k_way * n_shot) + q_total,)
        """
        for _ in range(self.n_batch):
            episode_indices = []

            all_class_indices = torch.randperm(self.num_classes)
            selected_k_way_indices = all_class_indices[:self.k_way]

            support_indices = []
            for class_idx in selected_k_way_indices:
                class_sample_indices = self.idx_list[class_idx.item()]
                num_available = class_sample_indices.size(0)
                
                if num_available < self.n_shot:
                    raise RuntimeError(
                        f"Class {class_idx.item()} has only {num_available} samples, "
                        f"but n_shot={self.n_shot} is required."
                    )
                
                perm = torch.randperm(num_available)
                selected_samples = class_sample_indices[perm[:self.n_shot]]
                support_indices.append(selected_samples)

            support_indices = torch.stack(support_indices)
            support_flat = support_indices.view(-1)

            query_class_perm = torch.randperm(self.k_way)
            query_class_positions = query_class_perm[:self.k_eff]
            query_class_indices = selected_k_way_indices[query_class_positions]

            query_indices = []
            samples_per_query_class = self.q_total // self.k_eff
            remainder = self.q_total % self.k_eff

            for i, class_idx in enumerate(query_class_indices):
                class_sample_indices = self.idx_list[class_idx.item()]
                num_available = class_sample_indices.size(0)

                support_mask = torch.isin(class_sample_indices, support_flat)
                available_for_query = class_sample_indices[~support_mask]
                num_available_for_query = available_for_query.size(0)

                n_query_for_this_class = samples_per_query_class
                if i < remainder:
                    n_query_for_this_class += 1

                if num_available_for_query < n_query_for_this_class:
                    raise RuntimeError(
                        f"Class {class_idx.item()} has only {num_available_for_query} "
                        f"samples available for query (excluding support), "
                        f"but {n_query_for_this_class} query samples are needed."
                    )

                perm = torch.randperm(num_available_for_query)
                selected_query_samples = available_for_query[perm[:n_query_for_this_class]]
                query_indices.append(selected_query_samples)

            query_flat = torch.cat(query_indices)

            episode_indices = torch.cat([support_flat, query_flat])

            expected_length = (self.k_way * self.n_shot) + self.q_total
            assert episode_indices.size(0) == expected_length, (
                f"Episode length mismatch: got {episode_indices.size(0)}, "
                f"expected {expected_length}"
            )

            yield episode_indices


class DistributedRealisticTransductiveSampler(Sampler):
    """
    A Distributed Sampler for Realistic Transductive Few-Shot Learning.
    
    This sampler is designed for DDP (Distributed Data Parallel) training,
    ensuring reproducibility across multiple GPUs.
    
    Args:
        label (list or np.ndarray): Label list of all samples in the dataset.
        n_batch (int): Number of episodes/batches to generate.
        k_way (int): Total number of classes in the support set (K).
        n_shot (int): Number of support samples per class (s).
        k_eff (int): Number of effective classes in the query set (K_eff).
        q_total (int): Total number of query samples (|Q|).
        rank (int): Rank of the current process.
        seed (int): Random seed for reproducibility.
        world_size (int): Number of processes participating in the job.
    """

    def __init__(
        self,
        label,
        n_batch,
        k_way,
        n_shot,
        k_eff,
        q_total,
        rank,
        seed=0,
        world_size=1,
    ):
        super(DistributedRealisticTransductiveSampler, self).__init__()
        
        self.n_batch = n_batch
        self.k_way = k_way
        self.n_shot = n_shot
        self.k_eff = k_eff
        self.q_total = q_total
        self.rank = rank
        self.seed = seed
        self.world_size = world_size
        self.epoch = 0

        label = np.array(label)
        self.unique_labels = np.unique(label)
        self.num_classes = len(self.unique_labels)

        self.idx_list = []
        for label_idx in self.unique_labels:
            ind = np.argwhere(label == label_idx).reshape(-1)
            ind = torch.from_numpy(ind)
            self.idx_list.append(ind)

        if self.k_way > self.num_classes:
            raise ValueError(
                f"k_way ({k_way}) cannot exceed total number of classes ({self.num_classes})"
            )
        if self.k_eff > self.k_way:
            raise ValueError(
                f"k_eff ({k_eff}) cannot exceed k_way ({k_way})"
            )

        self.cls_g = torch.Generator()
        self.sample_g = torch.Generator()
        self._reset_generators()

    def _reset_generators(self):
        self.cls_g.manual_seed(self.seed + self.epoch)
        self.sample_g.manual_seed(self.seed + self.epoch + 10000)

    def set_epoch(self, epoch: int) -> None:
        """
        Set the epoch for this sampler.
        
        This ensures all replicas use a different random ordering for each epoch.
        
        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch
        self._reset_generators()

    def __len__(self):
        return self.n_batch

    def __iter__(self):
        """
        Generate episodes for realistic transductive few-shot learning with
        deterministic random number generation for distributed training.
        
        Yields:
            torch.Tensor: Indices of samples for one episode.
        """
        for _ in range(self.n_batch):
            episode_indices = []

            all_class_indices = torch.randperm(
                self.num_classes, generator=self.cls_g
            )
            selected_k_way_indices = all_class_indices[:self.k_way]

            support_indices = []
            for class_idx in selected_k_way_indices:
                class_sample_indices = self.idx_list[class_idx.item()]
                num_available = class_sample_indices.size(0)
                
                if num_available < self.n_shot:
                    raise RuntimeError(
                        f"Class {class_idx.item()} has only {num_available} samples, "
                        f"but n_shot={self.n_shot} is required."
                    )
                
                perm = torch.randperm(num_available, generator=self.sample_g)
                selected_samples = class_sample_indices[perm[:self.n_shot]]
                support_indices.append(selected_samples)

            support_indices = torch.stack(support_indices)
            support_flat = support_indices.view(-1)

            query_class_perm = torch.randperm(self.k_way, generator=self.cls_g)
            query_class_positions = query_class_perm[:self.k_eff]
            query_class_indices = selected_k_way_indices[query_class_positions]

            query_indices = []
            samples_per_query_class = self.q_total // self.k_eff
            remainder = self.q_total % self.k_eff

            for i, class_idx in enumerate(query_class_indices):
                class_sample_indices = self.idx_list[class_idx.item()]

                support_mask = torch.isin(class_sample_indices, support_flat)
                available_for_query = class_sample_indices[~support_mask]
                num_available_for_query = available_for_query.size(0)

                n_query_for_this_class = samples_per_query_class
                if i < remainder:
                    n_query_for_this_class += 1

                if num_available_for_query < n_query_for_this_class:
                    raise RuntimeError(
                        f"Class {class_idx.item()} has only {num_available_for_query} "
                        f"samples available for query (excluding support), "
                        f"but {n_query_for_this_class} query samples are needed."
                    )

                perm = torch.randperm(num_available_for_query, generator=self.sample_g)
                selected_query_samples = available_for_query[perm[:n_query_for_this_class]]
                query_indices.append(selected_query_samples)

            query_flat = torch.cat(query_indices)

            episode_indices = torch.cat([support_flat, query_flat])

            expected_length = (self.k_way * self.n_shot) + self.q_total
            assert episode_indices.size(0) == expected_length, (
                f"Episode length mismatch: got {episode_indices.size(0)}, "
                f"expected {expected_length}"
            )

            yield episode_indices


def get_realistic_transductive_sampler(
    dataset,
    distribute,
    mode,
    config,
):
    """
    Factory function to create a RealisticTransductiveSampler.
    
    This function reads the configuration and creates the appropriate sampler
    (distributed or single-GPU) for the realistic transductive few-shot setting.
    
    Args:
        dataset: The dataset object, must have `label_list` attribute.
        distribute (bool): Whether to use distributed sampling.
        mode (str): 'train', 'val', or 'test'.
        config (dict): Configuration dictionary containing sampler parameters.
        
    Returns:
        Sampler: The appropriate sampler instance.
    """
    sampler_config = config.get("realistic_transductive", {})
    
    if mode == "train":
        k_way = sampler_config.get("k_way", config.get("way_num", 20))
        k_eff = sampler_config.get("k_eff", config.get("way_num", 5))
        n_shot = sampler_config.get("n_shot", config.get("shot_num", 5))
        q_total = sampler_config.get("q_total", 75)
        n_batch = config.get("train_episode", 1000)
    elif mode == "val":
        # 新增针对验证集的逻辑
        k_way = sampler_config.get("val_k_way", config.get("way_num", 5))
        k_eff = sampler_config.get("val_k_eff", config.get("way_num", 5))
        n_shot = sampler_config.get("val_n_shot", config.get("shot_num", 5))
        q_total = sampler_config.get("val_q_total", 75)
        n_batch = config.get("val_episode", config.get("test_episode", 1000))
    else:
        k_way = sampler_config.get("test_k_way", config.get("test_way", config.get("way_num", 20)))
        k_eff = sampler_config.get("test_k_eff", config.get("test_way", config.get("way_num", 5)))
        n_shot = sampler_config.get("test_n_shot", config.get("test_shot", config.get("shot_num", 5)))
        q_total = sampler_config.get("test_q_total", 75)
        n_batch = config.get("test_episode", 1000)

    if distribute:
        sampler = DistributedRealisticTransductiveSampler(
            label=dataset.label_list,
            n_batch=n_batch // config["n_gpu"],
            k_way=k_way,
            n_shot=n_shot,
            k_eff=k_eff,
            q_total=q_total,
            rank=config["rank"],
            seed=0,
            world_size=config["n_gpu"],
        )
    else:
        sampler = RealisticTransductiveSampler(
            label=dataset.label_list,
            n_batch=n_batch,
            k_way=k_way,
            n_shot=n_shot,
            k_eff=k_eff,
            q_total=q_total,
        )
    
    return sampler
