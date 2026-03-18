# 基于LibfewShot的UNEM算法复现

## 项目简介
本项目基于LibfewShot框架，实现了**UNEM: UNrolled Generalized EM for Transductive Few-Shot Learning**文章中纯视觉模型部分的复现，实验结果与原文基本一致
**原文链接 [here](https://cvpr.thecvf.com/virtual/2025/poster/34715)**
**原文代码仓库 [here](https://github.com/ZhouLong0/UNEM-Transductive)**

## 1.如何开始
### 1.1环境配置
1. 创建对应conda环境。
    ```
    cd <path-to-LibFewShot> # 进入clone好的项目目录
    conda create -n libfewshot python=3.10
    conda activate libfewshot
    ```
2. 跟随PyTorch和torchvision的[官方引导](https://pytorch.org/get-started/locally/)进行安装。
3. 安装对应依赖
   - pip
    ```
    cd <path-to-LibFewShot> # cd 进入`LibFewShot` 目录
    pip install -r requirements.txt
    ```
### 1.2下载数据集与模型
本项目使用与原文一致的数据集与预训练模型
请确保你所下载的数据集格式与LibfewShot所要求的格式一致 [数据格式](https://libfewshot-en.readthedocs.io/zh-cn/latest/tutorials/t2-add_a_new_dataset.html)
数据集与预训练模型下载请参考这个[仓库](https://github.com/imtiazziko/LaplacianShot)

## 2.复现结果
### 2.1修改配置文件
若想要复现原文的实验结果，请修改`config/unem.yaml`中的配置，将`realistic_tranductive`其中的`k_way,val_k_way,test_k_way`修改为数据集中训练集，验证集，测试集的类别数,并在`config/headers/data.yaml`中修改数据集目录。
按上述修改后，运行以下指令开始训练:
```
python run_trainer.py
```
### 2.2预期结果
预期结果如下：
![]()
![]()
