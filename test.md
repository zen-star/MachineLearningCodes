# 2018云从人头计数说明

*Author: 郑仕达*  
*Team: GGSDT*  
*Email: luckyzsd@163.com*  
*Tel: 15700080609*  
*Last Modified: 2019.02.14 22:50*

---

## 概述

本竞赛的目标是实现给定图像中人头位置的框选和人头计数，为目标检测任务。
考察数据集，人头尺度多样、场景多样、有很多模糊和遮挡的人头，比较有难度。
经典的通用目标检测模型（如：Fast/Faster-RCNN、SSD、FPN、Yolo系列等），
虽然一定程度上解决了目标检测中的一些通用问题，但是在针对本课题中多尺度、模糊、遮挡条件下的人头检测，仍然存在很大的提升空间。
考虑到人头检测和人脸检测具有一定的相似性，而且人脸检测在WIDERFACE数据竞赛上和论文有了非常多的成熟经验和模型积累，如SSH、Face RCNN、S3FD、FAN、PyramidBox等。
本次任务采用百度2018年下半年提出的**PyramidBox**人脸检测模型，根据人头检测的实际情况进行了适当的调整。
训练以**ImageNet**官方的预训练模型为骨架，在两块**Titan XP**（实际训练和测试只需要1块即可）上训练yuncong的数据集，得到了最终的人头检测模型和结果。
---

## 数据集特点

yuncong_data数据集中包含了—— 
- Mall: 固定视角的商场单一摄像头的拍摄图片
- Part_A/B: 以室外为主的多尺度密集人头场景
- our: 地铁站场景人头数多且模糊、商场场景背景复杂
- UCSD: 校园室外场景，框尺寸相同，且比例偏大，人头普遍偏小

另外，存在几个问题——
- 标注框相对于人头的大小在不同文件夹下不同，有些只框了头的一部分，有些框了整个上半身。
- 存在无人、标注框质量很差、标注框超出图像等问题的图片，需要进行数据清洗。
针对以上对数据的了解，在data/widerface.py中，进行了相关清洗和处理工作。具体见“代码说明”和程序文件中的注释。
---

## 模型介绍

PyramidBox是百度的一篇人脸检测论文提出的模型，截止2018.08在WIDERFACE数据集上排名第一，
fddb上continuous score排名第一，discrete score排名第二。
目前看来是一个非常出色的人脸检测模型。这里会简要介绍一下PyramidBox的思想，不展开分析。
如果有兴趣的话，可以参考arxiv上的论文原文https://arxiv.org/pdf/1803.07737.pdf。
1. PyramidBox是1-stage single shot的;
2. 提出**PyramidAnchors**，设计contextual anchor通过半监督的方法，来有监督地学习高层feature map的上下文信息；
具体地，结合了人头、人体等上下文信息，以检出小尺度、模糊、遮挡人脸；
3. 提出**LFPN**(Low-level Feature Pyramid Network)，联合合适的高层feature map语义信息
（不使用高层，而是中层feature map特征）+低层feature map特征，让PyramidBox可以single shot检测所有尺度人脸；
4. 提出**CPM**(Context-sensitive Predict Module)，借鉴SSH+DSSD，使用wider+deeper的网络结构，融合人脸附近的上下文信息，
并提出max-in-out层，提升检测分支中人脸检测+分类的准确率；
5. 提出**DAS**(Data Augmentation Sampling),增强不同尺度上的训练样本，特别是提升训练集中小尺度人脸的多样性

> 注1：以上2-4均为PyramidBox融合上下文信息检测人脸的关键部分，除了2中将人脸-人头-人体三级PyramidAnchors修改成了人头-人体两级PyramidAnchors
，其余均直接使用到人头检测中。  
> 注2：原文的骨架为VGG16，我们实现了VGG16和RESNET50骨架的PyramidBox，以对比性能。
---

## 代码说明

程序文件目录：`./GGSDT_PyramidBox/`  
环境：Python 3.5、Pytorch 0.3.1、OpenCV 3、visdom（可选）

- `./weights/`  
保存训练的中间结果。其中Vgg16_pyramid_70000x.pth和Res50_pyramid_75000x.pth是已训练好的最终提交的两个模型，vgg16bn.pth和resnet50.pth是ImageNet的预训练模型。

- `./data/`  
`config.py`是PyramidBox模型骨架的配置文件;  
`widerface.py`是dataloader代码，用于将ground truth转化为训练可用的数据结构。其中在AnnotationTransform()函数中调整了异常标注、标注格式，在Detection()函数中增加了Part_A/B的样本数量进行**数据平衡**、去掉了标注极差的样本。

- `./layers/`  
`box_utils.py`包含了可以调整先验框的工具;  
`functions/prior_box.py`网络构建的一部分，具体功能见注释;  
`modules/l2norm.py`用于论文中LFPN层l2正则化的实现;  
`modules/multibox_loss.py`为多损失函数的实现。

- `./utils/`  
`augmentations.py`是**数据增广**的代码，采用了PyramidBox的DAS、随机对比度、随机亮度、随机翻转。

- `./results/`  
保存测试的结果文件。其中test_final_pbr_0.2.txt是最终提交的成绩最好的文件。

- `./extra/`  
`create_nice_zsd.sh`用于将多个ground truth文件合并成一个，命名为nice_zsd.txt;  
`means_std.py`用于计算样本图片中BGR的均值和方差，用于训练和测试中;  
`resize.py`用于对测试结果进行缩放，**数据后处理**;  
`trim.py`用于对测试结果进行置信度筛选，仅留下>threshold的测试框。  

- `./`  
`pyramid.py`
网络模型构建的主程序，包含了模型构建的相关类/函数和模型导入的函数，模型骨架为resnet50。  
`pyramid_vgg.py`
网络模型构建的主程序，包含了模型构建的相关类/函数和模型导入的函数，模型骨架为vgg16bn，由于骨架不同，代码有较大调整。  
`train.py`
resnet50骨架模型的训练代码。  
`train_vgg.py`
vgg16骨架模型的训练代码，和train.py主要区别体现在**warmup**学习率变化曲线不同。  
`test.py`
resnet50骨架模型的测试代码，使用了**test with augmentation**的trick。  
`test_vgg.py`
vgg16骨架模型的测试代码，和test.py程序主体相同，仅模型和路径稍有不同  
*`test2m.py`
res50和vgg16集成测试代码，可适用于更多模型的联合测试，提交结果没有使用这个程序，代码主体大致相同。
---

## 训练预测流程

0. **文件目录结构**  
训练数据： `./yuncong_data/`  
测试数据： `./yuncong_test_set/`  
测试文件列表： `./merged_list_test.txt`  
模型代码文件： `./GGSDT_PyramidBox/`  
终端执行代码路径： `./`

1. **数据前处理**  
Part_B_train.txt中，Part_B/train_data/IMG_188.jpg的末尾补上一个' 8'，或去掉最后这个标注框。
将每个GT的.txt合成一个.txt，这里命名为nice_zsd.txt。  
执行: `sh ./GGSDT_PyramidBox/extra/create_nice_zsd.sh`

2. **环境准备**  
创建一个conda虚拟环境，其余环境如opencv3、visdom、torchvision、pickle等可以创建完conda环境后再安装，如果python和pytorch不同时安装的话，pytorch0.3.1的安装会自动安装一个python2.7，覆盖原来的python3.5。  
执行：`conda create -n GGSDT python=3.5 pytorch=0.3.1`  
进入(GGSDT)环境：  
执行：`source activate GGSDT`

3. **训练**（可跳过这一步，直接测试现有模型）  
训练resnet50骨架的模型：  
执行：`python ./GGSDT_PyramidBox/train.py`  
训练vgg16骨架的模型：  
执行：`python ./GGSDT_PyramidBox/train_vgg.py`  
训练过程可以使用visdom可视化管理。训练完成后在weights/下可以得到模型。

4. **测试**  
测试resnet50骨架模型的结果：  
执行：`python ./GGSDT_PyramidBox/test.py`  
测试vggnet16骨架模型的结果：  
执行：`python ./GGSDT_PyramidBox/test_vgg.py`  

5. **数据后处理**（根据标注风格调整）
将标注框放大至**1.2倍**，得最终提交的结果文件（需进入resize.py设置文件名）  
执行：`python ./GGSDT_PyramidBox/extra/resize.py`

6. **结束**  
提交结果，退出当前虚拟环境，完成任务  
执行：`source deactivate` 


> 以上准备的命令/代码均在本地通过测试，但由于代码在比赛期间有过多次调整，可能存在相关小问题，可发邮件或电话交流沟通，便于及时解决问题。
