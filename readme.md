# MadyGraph for Video Compressive Sensing

This repository contains the code for the paper [**Motion-aware Dynamic Graph Neural Network for Video Compressive
Sensing**](https://arxiv.org/pdf/2203.00387) (***IEEE TPAMI 2024***) by [Ruiying Lu*](https://faculty.xidian.edu.cn/LRY/zh_CN/index/444613/list/index.htm)
, [Ziheng Cheng*](https://github.com/zihengcheng), [Bo Chen](https://web.xidian.edu.cn/bchen/),
and [Xin Yuan](https://en.westlake.edu.cn/faculty/xin-yuan.html).

## Requirements

```
PyTorch
numpy
scipy
Scikit-Image
cupy
OpenCV-Python
```

## Test

The proposed **MadyGraph** can enhance the performance of any existing video SCI reconstructions. We have released our pretrained model
in ```model/```. This repository uses the results of [EfficientSCI](https://github.com/ucaswangls/EfficientSCI) as an
example. Run

```
python test.py
```

where will evaluate the performance of simulation data using the pre-trained model in ```model/```.

If you want to experiment with improving results using other methods, please modify the path (line 26) and adjust the corresponding codes in lines 60–64 of the ```test.py```.

## Citation

```
@ARTICLE{lu2024motion,
  author={Lu, Ruiying and Cheng, Ziheng and Chen, Bo and Yuan, Xin},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title={Motion-Aware Dynamic Graph Neural Network for Video Compressive Sensing}, 
  year={2024},
  volume={46},
  number={12},
  pages={7850-7866},
  doi={10.1109/TPAMI.2024.3395804}}
```

## Acknowledgements 

This work references or builds upon the following projects: [STFormer](https://github.com/ucaswangls/STFormer)
, [pytorch-liteflownet](https://github.com/sniklaus/pytorch-liteflownet/tree/master),
and [pytorch-deform-conv-v2](https://github.com/4uiiurz1/pytorch-deform-conv-v2/tree/master). We extend our gratitude to the authors for their contributions.

## Contact

[Ziheng Cheng, Xidian University](mailto:zhcheng@stu.xidian.edu.cn "Ziheng Cheng, Xidian University")

[Ruiying Lu, Xidian University](mailto:luruiying@xidian.edu.cn "Ruiying Lu, Xidian University")

[Bo Chen, Xidian University](mailto:bchen@mail.xidian.edu.cn "Bo Chen, Xidian University")

[Xin Yuan, Westlake University](mailto:xylab@westlake.edu.cn "Xin Yuan, Westlake University")  




































