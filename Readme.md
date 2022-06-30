
<h2 align="center"> <strong>2022 인공지능 온라인 경진 대회</strong> </h2>
<p align="center">
  <img src="http://cdn.aitimes.kr/news/photo/202205/24975_37447_521.jpg" width="350" height="450">
</p>

### [Task: 소고기 이미지를 통한 등급 분류 문제](https://aichallenge.or.kr/competition/detail/1/task/2/taskInfo)
#### Team Name: SKKUAI

#### **과제 설명**

![Untitled](https://user-images.githubusercontent.com/48243487/176602246-030edf7b-369d-4be0-8b41-f56583c6ff64.png)

✔ 소고기 도축 이미지를 보고 등급을 분류하는 문제

#### **추진배경**

✔ 우리나라는 소고기 마블링 스코어를 품질 등급 분류에 주요 변수로 삼기에 외국의 등급 체계와 차이가 있어 수입 소고기에 대해 국내 소비자들의 혼란이 있을 수 있음

✔ 축산 공공 데이터의 활용 방안 확대

<br>
<br>

## Data Preprocessing
- Cutmix
- Mixup
- Combinations of augmentation methods (kornia & pytorch)

<br>

**Note**:
Although Cutmix led to considerable enhancement in model's performace, combinations of augmentations didn't always lead to increase in test accuracy.
Also, as it was easier to insert code in training rather than making an independent dataset with cutmix or mixup method, we did not make independent dataset for those two methods.



| Augmentation method  | links |
| ------------- | ------------- |
| Horizontal Flip + Color Jitter  | [google drive link](https://drive.google.com/file/d/1gKT7zqmfNoD965EU1xwtg-q6rMMsH44K/view?usp=sharing)  |
| Grayscale + Vertical Flip | [google drive link](https://drive.google.com/file/d/1-NfobpSD6s9bSEUoddT2gKfuH_uTnwNP/view?usp=sharing)|
| Center Crop + Color Jitter + Random Erase | [google drive link](https://drive.google.com/file/d/1-MEVLznocCb8chCK8wDRGWbiJ1iWv6UH/view?usp=sharing)  |
| Content Cell  | [google drive link]() |

<br>
<br>

## Modeling

- AlexNet
- Sharpest Aware Minimization ([SAM](https://github.com/davda54/sam))

# Results

Evaluation Metrics: Weighted Kappa Score

| Methodology  | Test Accuracy |
| ------------- | ------------- |
| Content Cell  | Content Cell  |
| Content Cell  | Content Cell  |
| Content Cell  | Content Cell  |
| Content Cell  | Content Cell  |
