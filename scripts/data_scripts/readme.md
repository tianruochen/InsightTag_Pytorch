
在视频、音频、文本、图像特征均提取完成的情况下，制作数据集
1.提取类别信息: 
```
python extract_category.py 
```
2.划分数据集
```yaml
python build_trainval_data.py
```


注意：在线上机器上跑，15机器抓取不到数据
