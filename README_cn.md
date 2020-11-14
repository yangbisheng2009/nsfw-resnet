<div align=center>
  <img width=400 src="https://github.com/yangbisheng2009/nsfw-resnet/blob/master/image/nsfw_logo.jpg" >
</div>

[English](https://github.com/yangbisheng2009/nsfw-resnet/blob/master/README.md) | 中文版  

## 概述
基于60万图片数据训练性感&色情模型，标签如下：  

- `porn` - 色情
- `hentai` - 动漫色情、图画
- `sexy` - 性感
- `neutral` - 普通
- `drawings` - 普通动漫、图画

## 依赖
pytorch 1.0+
如果你需要pytorch0.4版本，请在release里面找到V1

## 如何使用
```
python train.py --model resnet101 --epochs 90 --batch-size 512 --checkpoint ./checkpoint --data-dir ./data

#test
python test_confusion_matrix.py

#predict
python predict --model resnet101 --checkpoint ./checkpoint/x

#if your machine has connected to the internet and you dosen't want to download the image to your disk
cat urls.txt | python predict_url.py
```

## 训练数据来源
好心人提供的开源数据： [nsfw_data_scraper](https://github.com/alexkimxyz/nsfw_data_scrapper).  
如果你想训练鲁棒性、效果更好的模型，可使用这[这份数据](https://github.com/EBazarov/nsfw_data_source_urls).  
如果你想训练适合工业生产环境的高准召模型，可以联系[我](https://twitter.com/yangbisheng2009).

## 当前效果
<div align=left>
  <img width=700 src="https://github.com/yangbisheng2009/nsfw-resnet/blob/master/image/test_confusion_matrix.jpg" >
</div>
由于色情、低俗在一定程度相似，也可以认为是标准定义的问题。对于我们使用来讲，影响不大。

## 心得
这个工程，我尝试了各种各样的方法。试验了通过直方图特征/傅里叶变换特征/小波变换特征 + 传统机器学习方法，以及inceptionv3，resnetX等各式各样的迁移学习方法，总结如下：  

- 尽可能调大batch_size，我在我的P40机器上，设置了512  
- 我最终选用了resnet101 pretrained model，它在诸多的方案中，表现最好
- 在整体finetune后，可以lock模型前面的N层，重新finetune一次。为什么这么做能带来更好的效果，我想你们都懂的
- 根据你的数据集、模型选择等因素，动态调整你的learning rate
- 我采用了很多数据增强的方法，如颜色变换、高斯噪声点、旋转、平移、剪切、色调对比度变换。但是结果发现这些方法，并没有太大的卵用（针对这个工程而言），最终只保留一小部分

## 感谢
这个工程耗时很长，在模型选型、模型调参、数据筛选过程中，都遇到了各种各样的困难（当然我认为，如果你想取得一个很不错的效果，你的数据肯定是最重要的）。感谢我老婆李菲菲女士，帮我制作了NSFW的logo，就是文章开始红色的那个，漂亮吧。同时也感谢我的伙伴李快同学，他给了我很多比较好的建议。
