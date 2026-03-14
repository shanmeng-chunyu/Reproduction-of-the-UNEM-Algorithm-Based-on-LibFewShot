# -*- coding: utf-8 -*-
from .collate_functions import (
    GeneralCollateFunction,
    FewShotAugCollateFunction,
    RealisticTransductiveCollateFunction,
)
from .contrib import get_augment_method, get_mean_std
from ...utils import ModelType


def get_collate_function(config, trfms, mode, model_type):
    """Set the corresponding `collate_fn` by dict.

    + For finetuning-train, return `GeneralCollateFunction`
    + For finetuning-val, finetuning-test and meta/metric-train/val/test, return `FewShotAugCollateFunction`
    + For realistic transductive mode, return `RealisticTransductiveCollateFunction`

    Args:
        config (dict): A LFS setting dict.
        trfms (list): A torchvision transform list.
        mode (str): Model mode in ['train', 'test', 'val']
        model_type (ModelType): An ModelType enum value of model.

    Returns:
        [type]: [description]
    """
    assert (
        model_type != ModelType.ABSTRACT
    ), "model_type should not be ModelType.ABSTRACT"

    use_realistic = config.get("realistic_transductive", {}).get("enable", False)

    if mode == "train" and model_type == ModelType.FINETUNING:
        collate_function = GeneralCollateFunction(trfms, config["augment_times"])
    elif use_realistic:
        rt_config = config.get("realistic_transductive", {})
        if mode == "train":
            k_way = rt_config.get("k_way", config.get("way_num", 20))
            n_shot = rt_config.get("n_shot", config.get("shot_num", 5))
            k_eff = rt_config.get("k_eff", config.get("way_num", 5))
            q_total = rt_config.get("q_total", 75)
        else:
            k_way = rt_config.get("test_k_way", config.get("test_way", config.get("way_num", 20)))
            n_shot = rt_config.get("test_n_shot", config.get("test_shot", config.get("shot_num", 5)))
            k_eff = rt_config.get("test_k_eff", config.get("test_way", config.get("way_num", 5)))
            q_total = rt_config.get("test_q_total", 75)
        
        collate_function = RealisticTransductiveCollateFunction(
            trfms,
            config["augment_times"],
            config["augment_times_query"],
            k_way,
            n_shot,
            k_eff,
            q_total,
        )
    else:
        collate_function = FewShotAugCollateFunction(
            trfms,
            config["augment_times"],
            config["augment_times_query"],
            config["way_num"] if mode == "train" else config["test_way"],
            config["shot_num"] if mode == "train" else config["test_shot"],
            config["query_num"] if mode == "train" else config["test_query"],
        )

    return collate_function
