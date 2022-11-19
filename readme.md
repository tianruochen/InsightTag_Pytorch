# InsightTag_Pytorch
[![python version](https://img.shields.io/badge/python-3.6%2B-brightgreen)]()
[![coverage](https://img.shields.io/badge/coverage-56%25-orange)]()

多模态模型训练与推理

多模态的两种方案：
> 方案一：
- vit 提取视频 (m * 1024)和图片特征 (n * 1024)
- vigish 提取音频特征 (j * 128)
- bret 提取文本特征 
- 特征融合

> 方案二
- tsn + lstm 提取视频特征  (1 * 300 * 2048) + (1 * 4096)
- eff_103类 提取图片特征  (n * 2048)
- vigish 提取音频特征  (j * 128)
- bert 提取文本特征
- 特征融合


视频特征与图片特征的两种融合策略
- 将图片视作视频帧处理
- 将图片作为独立模态处理 

文本内容与ocr的处理策略
- 将ocr作为文本内容的补充
- 将ocr与文本内容分开，独立处理

## Table of Contents

- [Structure](#structure)
- [Data](#Data)
- [Train](#Train)
- [Inference](#Inference)
- [Contributing](#contributing)
- [License](#license)



## structure
```
├── config                     
│     ├── infer_config.yaml                    测试时配置文件 
│     ├── train_config.yaml                    训练时配置文件
│     ├── infer_config_e2e.yaml                端到端的前向推理文件  
│     ├── idx2name_clsxxxx.json                id与label映射文件
├── data     
│     ├── test_cls500_samples.json                 测试样例数据                         
│     ├── valid_cls500_samples.json                验证样例数据
│     ├── train_cls500_samples.json                训练样例数据
├── modules 
│     ├── __init__.py                    
│     ├── dataset                              数据集相关
│     ├── metric                               metric相关
│     ├── loss                                 loss相关
│     ├── models                               模型相关
│     ├── optimier                             优化器相关
│     ├── prepare                              前处理任务相关
│     ├── slover 
│     │       ├── base.py                      
│     │       ├── trainer.py                   deal with the whole training process
│     │       ├── inferer.py                   deal with the whole inference process
├── scripts                                    
│     ├── data_scripts                           数据处理相关脚本
│     ├── mongo_supervisor                       数据库监控相关脚本
│     ├── extract_audio_features.py                     提取音频特征的脚本 
│     ├── extract_image_features.py                     提取图像特征的脚本
│     ├── extract_text_features.py                      提取文本特征的脚本
│     ├── extract_video_features.py                     提取视频特征的脚本
├── utils                              utils工具包
├── workshop                           
│     ├── ckpt                         a directory to save checkpoint
│     ├── log                          a directory to save log 
│     ├── weights                      最终上线时权重存放路径       
├── train_net.py                       training script      
├── infer_net.py                       inference script     
├── flask_server.py                    web服务方式处理入口   
├── king.py                            消息队列方式核心处理类 
├── online_process.py.py               消息队列方式的服务入口   

```     

## Data
> 第一步 拉取数据
```yaml
cd scripts/data_script 
python pull_xxx_data.py       # 拉取帖子数据（在140机器上，15机器上拉不到数据）
python analyze_xxx_data.py    # 分析获取的数据
python  download_mp.py           # 多进程下载数据
```

> 第二步 提取特征, 构建数据集

```yaml
cd scripts/
python extract_text_features.py     # 提取文本特征
python extract_audio_features.py    # 提取音频特征
python extract_video_features.py    # 提取视频特征
python build_trainval_data.py       # 构建训练集验证集

最终的训练集验证集格式参考 data/valid_cls24.json
```


## Train
To train a new model, use the main.py script.

use default params (all parameters setted in model_config file)
```
python train_net.py --model_config XXX 
```
use custom params
```
python train_net.py --model_config XXX --batch_size XXX --learning_rate XXX ...
# custom params
# --batch_size: int, [training batch size, None to use config setting]
# --learning_rate: float, [training learning rate, None to use config setting]
# --resume: str, [path to pretrain weights]
# --n_gpu: int, [the number of gpus]
# --epoch: int, [epoch number, 0 for read from config file]
# --save_dir: str, [directory name to save train snapshoot, None to use config setting]
# --valid_interval: str, [validation epoch interval, 0 for no validation]
# --log_interval: str, [mini batch interval to log]
# --fix_random_seed: bool, [If set True, set rand seed]
```

## Inference
Use the following command to inference.
> predict
```
python infer_net.py --infer_config XXX --input XXX.json
```
> other params
```
python infer_net.py --infer_config XXX --input XXX.txt --n_gpus XXX --best_model XXX
# --n_gpus: int, [the numbers of gpu needed (default 1)]
# --best_model: str, [the best model for inference]
```
> flask接口
```
python flask_server.py
```
> 消息队列服务接口
```
python online_process.py
```
> 预训练好的权重位置(15机器)
```yaml
1.直接torch.load()加载
/data/changqing/multi_mode/Insight_Multimodal_Pytorch/weights/checkpoints/vit_best_epoch15_gap76.40_acc98.66_iter700m.pth
2.先troch.load() 再load_state_dict()
/data/changqing/multi_mode/Insight_Multimodal_Pytorch/workshop/multimodal_v01_cls24_0622_160733/ckpt
```

 
## Contributing

## License

