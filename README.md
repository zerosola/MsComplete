<div align="center">

<h3>Geometry-Aware Multiscale Feedback Network for Point Cloud Completion with Skip-Connected Structured Compensation</h3>

[![Paper](https://img.shields.io/badge/Journal-Submitted-blue.svg)]()


*Official PyTorch Implementation for Point Cloud Completion*

</div>


---




## 📦 Pretrained model
We provide pretrained MsComplete models as follows:
- 📥 **[Download](https://pan.baidu.com/s/1ws-dInH1IsCn_tkNGFOspA)**
[Extract Code: 1234]





## 🚀 Running the Code

### PCN Dataset
```bash
【Train】 python main.py
--config ./cfgs/PCN_models/MsComplete.yaml
--exp_name example
```

```bash
【Test】 python main.py
--config ./cfgs/PCN_models/MsComplete.yaml
--exp_name example
--ckpts ./pretrained/Pcn_final/ckpt-best.pth
--test
```

### ShapeNet55 Dataset
```bash
【Train】 python main.py
--config ./cfgs/ShapeNet55_models/MsComplete.yaml
--exp_name example
```

```bash
【Test】 python main.py
--config ./cfgs/ShapeNet55_models/MsComplete.yaml
--mode easy  # median  hard
--exp_name example
--ckpts ./pretrained/Shapenet55_final/ckpt-best.pth
--test
```


### ShapeNet34 Dataset
```bash
【Train】 python main.py
--config ./cfgs/ShapeNet34_models/MsComplete.yaml
--exp_name example

【Train-Unseen21】 python main.py
--config ./cfgs/ShapeNetUnseen21_models/MsComplete.yaml
--exp_name example
```

```bash
【Test】 python main.py
--config ./cfgs/ShapeNet34_models/MsComplete.yaml
--mode easy  # median  hard
--exp_name example
--ckpts ./pretrained/Shapenet34_final/ckpt-best.pth
--test

【Test-Unseen21】 python main.py
--config ./cfgs/ShapeNetUnseen21_models/MsComplete.yaml
--mode easy  # median  hard
--exp_name example
--ckpts ./pretrained/Shapenet34_final/ckpt-best.pth
--test
```

### KITTI Dataset
```bash
【Train】 python main.py
--config ./cfgs/KITTI_models/MsComplete.yaml
--exp_name example
--start_ckpts ./pretrained/Pcn_final/ckpt-best.pth
```

```bash
【Test】 python main.py
--config ./cfgs/KITTI_models/MsComplete.yaml
--exp_name example
--ckpts ./pretrained/Kitti_final/ckpt-best.pth
--test

【Metric】 KITTI_metric.py
--vis_path=./experiments/MsComplete/KITTI_models/test_example/vis_result
```


### MVP Dataset
```bash
【Train】 python main_mvp.py
--config ./cfgs/MVP_models/MsComplete.yaml
--exp_name example
```

```bash
【Test】 python main_mvp.py
--config ./cfgs/MVP_models/MsComplete.yaml
--exp_name example
--ckpts ./pretrained/Mvp_final/ckpt-best.pth
--test
```

## Acknowledgements

Our code is inspired by [PoinTr](https://github.com/yuxumin/PoinTr).
