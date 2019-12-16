Mask Scoring R-CNN (MS R-CNN)
-----------------
By [Zhaojin Huang](https://github.com/zjhuang22), [Lichao Huang](https://scholar.google.com/citations?user=F2e_jZMAAAAJ&hl=en), [Yongchao Gong](https://dblp.org/pers/hd/g/Gong:Yongchao), [Chang Huang](https://scholar.google.com/citations?user=IyyEKyIAAAAJ&hl=zh-CN), [Xinggang Wang](http://www.xinggangw.info/index.htm).

CVPR 2019 Oral Paper, [pdf](http://openaccess.thecvf.com/content_CVPR_2019/papers/Huang_Mask_Scoring_R-CNN_CVPR_2019_paper.pdf)

This project is based on [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark).


这篇CVPR 2019论文就提出了一种新的打分方法：给蒙版打分，他们称之为蒙版得分（mask score）。
△ MS R-CNN架构
Mask Scoring R-CNN中提出的计分方式很简单：不仅仅直接依靠检测得到的分类算分，而且还让模型单独学一个针对蒙版的得分规则：MaskIoU head。
MaskIoU head是在经典评估指标AP（平均正确率）启发下得到的，会拿预测蒙版与物体特征进行对比。MaskIoU head同时接收蒙版head的输出与ROI的特征（Region of Interest）作为输入，用一种简单的回归损失进行训练。
最后，同时考虑分类得分与蒙版的质量得分，就可以去评估算法质量了。
评测方法公平公正，实例分割模型性能自然也上去了。
实验证明，在挑战COCO benchmark时，在用MS R-CNN的蒙版得分评估时，在不同基干网路上，AP始终提升近1.5%。
优于Mask R-CNN
下面的表格，是COCO 2017测试集（Test-Dev set）上MS R-CNN和其他实例分割方法的成绩对比。
无论基干网络是纯粹的ResNet-101，还是用了DCN、FPN，MS R-CNN的AP成绩都比Mask R-CNN高出一点几个百分点。
在COCO 2017验证集上，MS R-CNN的得分也优于Mask R-CNN：
作者是谁？
第一作者，名为黄钊金，华中科技大学的硕士生，师从华中科技大学电信学院副教授王兴刚，王兴刚也是这篇论文的作者之一。


Introduction
-----------------
[Mask Scoring R-CNN](https://arxiv.org/pdf/1903.00241.pdf) contains a network block to learn the quality of the predicted instance masks. The proposed network block takes the instance feature and the corresponding predicted mask together to regress the mask IoU. The mask scoring strategy calibrates the misalignment between mask quality and mask score, and improves instance segmentation performance by prioritizing more accurate mask predictions during COCO AP evaluation. By extensive evaluations on the COCO dataset, Mask Scoring R-CNN brings consistent and noticeable gain with different models and different frameworks. The network of MS R-CNN is as follows:

![alt text](demo/network.png)


Install
-----------------
  Check [INSTALL.md](INSTALL.md) for installation instructions.


Prepare Data
----------------
```
  mkdir -p datasets/coco
  ln -s /path_to_coco_dataset/annotations datasets/coco/annotations
  ln -s /path_to_coco_dataset/train2014 datasets/coco/train2014
  ln -s /path_to_coco_dataset/test2014 datasets/coco/test2014
  ln -s /path_to_coco_dataset/val2014 datasets/coco/val2014
```


Pretrained Models
---------------
```
  mkdir pretrained_models
  #The pretrained models will be downloaded when running the program.
```
My training log and pre-trained models can be found here [link](https://1drv.ms/f/s!AntfaTaAXHobhkCKfcPPQQfOfFAB) or [link](https://pan.baidu.com/s/192lRQozksu5XwpU9EO5neg)(pw:xm3f).




Running
----------------
Single GPU Training
```
  python tools/train_net.py --config-file "configs/e2e_ms_rcnn_R_50_FPN_1x.yaml" SOLVER.IMS_PER_BATCH 2 SOLVER.BASE_LR 0.0025 SOLVER.MAX_ITER 720000 SOLVER.STEPS "(480000, 640000)" TEST.IMS_PER_BATCH 1
```
Multi-GPU Training
```
  export NGPUS=8
  python -m torch.distributed.launch --nproc_per_node=$NGPUS tools/train_net.py --config-file "configs/e2e_ms_rcnn_R_50_FPN_1x.yaml" 
```


Results
------------
| NetWork  | Method | mAP(mask) | mAP(det)  |
|----------|--------|-----------|-----------|
| ResNet-50 FPN | Mask R-CNN | 34.2 | 37.8 |
| ResNet-50 FPN | MS R-CNN | 35.6 | 37.9 |
| ResNet-101 FPN | Mask R-CNN | 36.1 | 40.1 |
| ResNet-101 FPN | MS R-CNN | 37.4 | 40.1 |



Visualization
-------------
![alt text](demo/demo.png)
The left four images show good detection results with high classification scores but low mask quality. Our method aims at solving this problem. The rightmost image shows the case of a good mask with a high classification score. Our method will retrain the high score. As can be seen, scores predicted by our model can better interpret the actual mask quality.

Acknowledgment
-------------
The work was done during an internship at [Horizon Robotics](http://en.horizon.ai/).

Citations
---------------
If you find MS R-CNN useful in your research, please consider citing:
```
@inproceedings{huang2019msrcnn,
    author = {Zhaojin Huang and Lichao Huang and Yongchao Gong and Chang Huang and Xinggang Wang},
    title = {{Mask Scoring R-CNN}},
    booktitle = {CVPR},
    year = {2019},
}   
```

License
---------------
maskscoring_rcnn is released under the MIT license. See [LICENSE](LICENSE) for additional details.

Thanks to the Third Party Libs
---------------  
[maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark)   
[Pytorch](https://github.com/pytorch/pytorch)   
