<div align="center">

<h3>Geometry-Aware Multiscale Feedback Network for Point Cloud Completion with Skip-Connected Structured Compensation</h3>

\[!\[Paper](https://img.shields.io/badge/Journal-Submitted-blue.svg)]()



*Official PyTorch Implementation for Point Cloud Completion*

</div>
---
## 📦 Pretrained model
We provide pretrained MsComplete models as follows:


- 📥 \[Download](https://pan.baidu.com/s/1ws-dInH1IsCn\_tkNGFOspA)
\[Extract Code: 1234]





## 🚀 Running the Code

### PCN Dataset

```bash
【Train】 python main.py
--config ./cfgs/PCN\_models/MsComplete.yaml
--exp\_name example
```

```bash
【Test】 python main.py
--config ./cfgs/PCN\_models/MsComplete.yaml
--exp\_name example
--ckpts ./pretrained/Pcn\_final/ckpt-best.pth
--test
```

### ShapeNet55 Dataset

```bash
【Train】 python main.py
--config ./cfgs/ShapeNet55\_models/MsComplete.yaml
--exp\_name example
```

```bash
【Test】 python main.py
--config ./cfgs/ShapeNet55\_models/MsComplete.yaml
--mode easy  # median  hard
--exp\_name example
--ckpts ./pretrained/Shapenet55\_final/ckpt-best.pth
--test
```



### ShapeNet34 Dataset

```bash
【Train】 python main.py
--config ./cfgs/ShapeNet34\_models/MsComplete.yaml
--exp\_name example

【Train-Unseen21】 python main.py
--config ./cfgs/ShapeNetUnseen21\_models/MsComplete.yaml
--exp\_name example
```

```bash
【Test】 python main.py
--config ./cfgs/ShapeNet34\_models/MsComplete.yaml
--mode easy  # median  hard
--exp\_name example
--ckpts ./pretrained/Shapenet34\_final/ckpt-best.pth
--test

【Test-Unseen21】 python main.py
--config ./cfgs/ShapeNetUnseen21\_models/MsComplete.yaml
--mode easy  # median  hard
--exp\_name example
--ckpts ./pretrained/Shapenet34\_final/ckpt-best.pth
--test
```

### KITTI Dataset

```bash
【Train】 python main.py
--config ./cfgs/KITTI\_models/MsComplete.yaml
--exp\_name example
--start\_ckpts ./pretrained/Pcn\_final/ckpt-best.pth
```

```bash
【Test】 python main.py
--config ./cfgs/KITTI\_models/MsComplete.yaml
--exp\_name example
--ckpts ./pretrained/Kitti\_final/ckpt-best.pth
--test

【Metric】 KITTI\_metric.py
--vis\_path=./experiments/MsComplete/KITTI\_models/test\_example/vis\_result
```



### MVP Dataset

```bash
【Train】 python main\_mvp.py
--config ./cfgs/MVP\_models/MsComplete.yaml
--exp\_name example
```

```bash
【Test】 python main\_mvp.py
--config ./cfgs/MVP\_models/MsComplete.yaml
--exp\_name example
--ckpts ./pretrained/Mvp\_final/ckpt-best.pth
--test
```

## Acknowledgements

Our code is inspired by [PoinTr](https://github.com/yuxumin/PoinTr).

