# MobileNet V4
Unofficial PyTorch implementation of _MobileNetV4_ based on https://arxiv.org/abs/2404.10518 as a plug-and-play component.

This version covers the smaller, convolutional-only models presented, made in a straightforward way to be included in other projects.

#### Some customizations are offered
- Input / output channels of the model (to adapt to any dataset or use case)
- _Stem_ / _classifier_ layers can be omitted (to adapt to customized architectures)

#### Considerations for future updates
- Dropout / Stochastic depth
- Additional depth-wise convolution (as shown in the [official repo](https://github.com/tensorflow/models/blob/master/official/vision/modeling/layers/nn_blocks.py#L777))

#### Additional
The _Universal Inverted Bottleneck_ block is shown below (reproduced from the paper):

[![uib.png](https://i.postimg.cc/Fz4VT9Kn/uib.png)](https://postimg.cc/BLmF6fHx)

The specifications for each layer use the following list format (consult the image above)
```
Extra DW kernel size............(int)...enables the first depth-wise convolution in the block with the given kernel size (None = disables this convolution)
Intermediate DW kernel size.....(int)...enables the second depth-wise convolution in the block with the given kernel size (None = disables this convolution)
Expanded dim....................(int)...sets the output channel dimension of the first convolution in the block
Layer output size...............(int)...sets the output channel dimension of the block
Layer stride....................(int)...sets the stride of the convolutions in the block
Fused...........................(bool)..enables the first convolution in the block as regular (outputs expanded dim channels, instead of using DW and PW)
``` 

#### Citation
Credit to the authors:
```
@inproceedings{10.1007/978-3-031-73661-2_5,
author = {Qin, Danfeng and Leichner, Chas and Delakis, Manolis and Fornoni, Marco and Luo, Shixin and Yang, Fan and Wang, Weijun and Banbury, Colby and Ye, Chengxi and Akin, Berkin and Aggarwal, Vaibhav and Zhu, Tenghui and Moro, Daniele and Howard, Andrew},
title = {MobileNetV4: Universal Models for the Mobile Ecosystem},
year = {2024},
isbn = {978-3-031-73660-5},
doi = {10.1007/978-3-031-73661-2_5},
booktitle = {18th European Conference of Computer Vision (ECCV)},
pages = {78–96},
numpages = {19},
location = {Milan, Italy}
}
```
