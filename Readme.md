
<h2 align="center"> <strong>2022 인공지능 온라인 경진 대회</strong> </h2>

<p align="center">
  <img src="https://user-images.githubusercontent.com/48243487/177003289-038ff31e-3459-49c3-8c64-c3c51b47bcfd.jpg" width="350" height="450">
</p>

### [Task: 소고기 이미지를 통한 등급 분류 문제](https://aichallenge.or.kr/competition/detail/1/task/2/taskInfo)
#### Team Name: SKKUAI (Members: Chaeeun Ryu, Sangeun Lee, Sangmin Han, Jaemin Yoo)
#### Final Ranking: 16th (top 17%)

#### **과제 설명**

![Untitled](https://user-images.githubusercontent.com/48243487/176602246-030edf7b-369d-4be0-8b41-f56583c6ff64.png)

✔ 소고기 도축 이미지를 보고 등급을 분류하는 문제

#### **추진배경**

✔ 우리나라는 소고기 마블링 스코어를 품질 등급 분류에 주요 변수로 삼기에 외국의 등급 체계와 차이가 있어 수입 소고기에 대해 국내 소비자들의 혼란이 있을 수 있음

✔ 축산 공공 데이터의 활용 방안 확대

<br>
<br>

## Data Preprocessing

<p align="center">
  <img src="https://user-images.githubusercontent.com/48243487/176877307-2321c6e1-302b-4dff-9e1e-f8975c6d99b6.JPG" width="400" height="500">
</p>

- Cutmix
- Mixup
- Combinations of augmentation methods (kornia & pytorch)

<br>

**Note**:
Although Cutmix led to considerable enhancement in model's performace, combinations of augmentations didn't always lead to increase in test accuracy.
Also, as it was easier to insert code in training rather than making an independent dataset with cutmix or mixup method, we did not make independent dataset for those two methods.
<br>


| Augmentation method  | links |
| ------------- | ------------- |
| Horizontal Flip + Color Jitter  | link  |
| Grayscale + Vertical Flip | link|
| Center Crop + Color Jitter + Random Erase | link  |
| Horizontal Flip + Random Rotation  | link |
| Random Vertical Flip + Rotation  | link |
| Random Vertical Flip + Center Crop  | link |
| Random Vertical Flip + Color Jitter  | link |
| Cutmix  | link |
| Mixup  | link |



**Note**: Currently, the links to datasets are not allowed due to the policy of competition.

<br>
<br>

## Modeling

- AlexNet
- Sharpest Aware Minimization ([SAM](https://github.com/davda54/sam))
- EfficientNet
- ResNet Series
- WideResNet
- Contrastive Learning (simclr pretrained weights)
- Consideration of Color Histogram

<br>
<br>

# Results in numbers

Evaluation Metrics: Weighted Kappa Score

| Methodology  | Test Accuracy |
| ------------- | ------------- |
| WideResNet+SAM  | 0.218  |
| RseNet101 | 0.865  |
| EfficientNet  | 0.914  |
| SpiralNet-ResNet  | 0.905  |
| ResNet34+Endsemble  | 0.727  |
| Transformer (Backbone: ResNet50)  | 0.906  |
| WideResNet+Ensemble  | 0.942  |
| EfficientNet with augmentation | 0.930  |
| EfficientNet + Augmentation + Cutmix  | 0.94  |
| EfficientNet + cutmix  | 0.932 |
| EfficientNet + Augmentation + Cutmix + lr=1e-03, epoch = 150  | 0.957 |
| AlexNet+Semi-supervised  | 0.953  |
| AlexNet+Cutmix+Mixup  | 0.936  |
| EfficientNet+cutmix+mixup+color histogram (w/o regularization)  | 0.943  |
| EfficientNet+all augmented data  | 0.952  |
| Ensemble+augmentation  | 0.913  |
| Snapshot Ensemble  | 0.931  |

**Insight**: Not all models' performance was improved by large augmented data. And all models faced overfitting, with training accuracy reaching 100%, and yet validation accuracy converging below 100%.


<br>
<br>

# Result Analyis in Visualization

**Note**: Every accuracy or error rate shown below are based on standard accuracy, and we noted that when we submit our submission file, and assess our submission based on weighted Kappa Score, which takes the relative distance between labels into consideration, the accuracy is calculated higher than ours.


**EfficientNet+Augmentated Data + Cutmix + Mixup + Color Histogram**
<p align="center">
  <img src="https://user-images.githubusercontent.com/48243487/177065793-6d372420-0829-4371-bba8-b064f6c22de7.png" width="800" height="500">
</p>


**Color histogram info.**

<p align="center">
  <img src="https://user-images.githubusercontent.com/48243487/177067633-4b79a018-d074-4d2d-9a23-ee38b297dca8.png" width="500" height="500">
</p>

<p align="center">
  <img src="https://user-images.githubusercontent.com/48243487/177067597-b315b554-c840-4edf-aaf6-f64bdfe1431c.png" width="500" height="500">
</p>

<p align="center">
  <img src="https://user-images.githubusercontent.com/48243487/177067527-0ea8f8e7-7fbf-40aa-ab0a-d0c49a94efc0.png" width="500" height="500">
</p>

<p align="center">
  <img src="https://user-images.githubusercontent.com/48243487/177067682-91070b80-57a5-444e-a40a-81d00465f874.png" width="500" height="500">
</p>

<p align="center">
  <img src="https://user-images.githubusercontent.com/48243487/177067699-315972f7-f273-470b-bed3-5a9e3ad0612e.png" width="500" height="500">
</p>



**Efficient Net**
<p align="center">
  <img src="https://user-images.githubusercontent.com/48243487/177024171-dba390ec-b506-400a-99f6-918ba3d5a16a.JPG" width="500" height="500">
</p>
<p align="center">
  <img src="https://user-images.githubusercontent.com/48243487/177024194-79467760-2683-4212-a785-cfadbdbb6d83.JPG" width="500" height="500">
</p>



**WideResNet 101**
<p align="center">
  <img src="https://user-images.githubusercontent.com/48243487/176878012-6e71b61c-eb54-4ef9-80c4-8dc894626d30.png" width="500" height="500">
</p>
<p align="center">
  <img src="https://user-images.githubusercontent.com/48243487/176878156-d9ee730d-e18e-4b68-a351-f66fb2029b3d.png" width="500" height="500">
</p>


**EfficientNet b4**

<p align="center">
  <img src="https://user-images.githubusercontent.com/48243487/176878558-74b5dbc0-481a-4ba4-bea3-dd15dcdcf30f.png" width="500" height="500">
</p>


<p align="center">
  <img src="https://user-images.githubusercontent.com/48243487/176878587-d5a3227c-2c63-4a54-be73-492c447ee09b.png" width="500" height="500">
</p>



**Note**: Contact via github accounts in **contributors**, if any issue.


