# 자율주행 센서의 안테나 성능 예측 AI 경진대회

## 개요
* 본 대회는 LG AI Research가 주최한 대회입니다.
* 공정 데이터를 활용하여 Rader 센서의 안테나 성능 예측을 위한 AI를 개발하는 것이 본 대회의 목표였습니다.
* 팀명은 1교시 통학러였으며, 저의 활동명은 통계가조아였습니다.

## 배경
* 본 대회는 리더보드를 통하여 정량평가로 진행되었습니다.
* Rader는 자율주행 차에 있어 차량과의 거리, 상대속도, 방향 등을 측정해주는 필수적인 센서 부품입니다.
* 전기자동차, 자율주행차, 로보택시, 자율주행 택배로봇 등 Radar가 활용되는 시장은 점차 커지고 있습니다. 
* 제품 종류도 기존 단거리, 중거리 및 장거리 Radar 뿐 아니라 차량 실내용 및 4D 이미징 Radar등 다변화 되는 추세입니다.
* LG에서는 제품의 성능 평가 공정에서 양품과 불량을 선별하고 있으며, AI 기술을 활용하여 공정 데이터와 제품 성능간 상관 분석을 통해 제품의 불량을 예측/분석하고, 수율을 극대화하여 불량으로 인한 제품 폐기 비용을 감축하려고 노력하고 있습니다.

## 평가 산식
* 해당 대회에서는 평가산식이 존재하였습니다.
* 평가 산식은 Normalized RMSE(NRMSE)를 사용하였습니다.
```
def lg_nrmse(gt, preds):
    # 각 Y Feature별 NRMSE 총합
    # Y_01 ~ Y_08 까지 20% 가중치 부여
    all_nrmse = []
    for idx in range(1,15): # ignore 'ID'
        rmse = metrics.mean_squared_error(gt[:,idx], preds[:,idx], squared=False)
        nrmse = rmse/np.mean(np.abs(gt[:,idx]))
        all_nrmse.append(nrmse)
    score = 1.2 * np.sum(all_nrmse[:8]) + 1.0 * np.sum(all_nrmse[8:14])
    return score
```

## 개발환경
<p align="center">
  <img src="https://img.shields.io/badge/TensorFlow-FF6F00?style=flat-square&logo=TensorFlow&logoColor=white"/></a>&nbsp
  <br>
  <img src="https://img.shields.io/badge/pandas-150458?style=flat-square&logo=pandas&logoColor=white"/></a>&nbsp
  <img src="https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=NumPy&logoColor=white"/></a>&nbsp
  <img src="https://img.shields.io/badge/SciPy-8CAAE6?style=flat-square&logo=SciPy&logoColor=white"/></a>&nbsp
  <img src="https://img.shields.io/badge/scikit-learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white"/></a>&nbsp
  <br>
    <img src="https://img.shields.io/badge/-LightGBM-yellow"/></a>&nbsp
    <img src="https://img.shields.io/badge/-catboost-yellow"/></a>&nbsp
    <img src="https://img.shields.io/badge/-byes_opt-blueviolet"/></a>&nbsp
    <img src="https://img.shields.io/badge/-matplotlib-blue"/></a>&nbsp
</p>

## 프로젝트 결과
* public : 30/988
![public](https://user-images.githubusercontent.com/89781598/189795628-7b00ffc2-b06e-44d6-ae6a-e23dd46e881a.png)

* private : 42/988
![private](https://user-images.githubusercontent.com/89781598/189795640-c554775c-0220-4ab7-800c-fa377c32ba21.png)

* 교육 수료증

![화면 캡처 2022-09-13 114246](https://user-images.githubusercontent.com/89781598/189798956-432201b0-bc56-4b11-9f2d-8c4857f0fbd7.png)

## LG Aimers / Data Intelligence
- LG AI 연구원에서 주관하는 LG Aimers / Data Intelligence 교육을 수료한 후, 해당 공모전을 진행하였습니다.
- 교육에서는 아래의 내용을 학습하였습니다.
    - SPC
    - 지도학습
    - 비지도학습
    - 설명 가능한 AI
    - 인과추론
    - 자율주행과 레이더센서의 이해

## 프로젝트 설명
<h3 align="center">🪄 프로젝트에 대한 설명은 차후에 적겠습니다! 🪄</h3>

## 파일 구조
```
📦LG.RADER
 ┣ 📂EDA
 ┃ ┣ 📜ANOVA.ipynb
 ┃ ┣ 📜Correlation and Distribution.ipynb
 ┃ ┗ 📜NULL.ipynb
 ┣ 📂MODEL
 ┃ ┣ 📜MultiOutput_Catboost_Bayesian.ipynb
 ┃ ┣ 📜MultiOutput_LGBM_Bayesian.ipynb
 ┃ ┗ 📜Submission.ipynb
 ┣ 📂Trial
   ┣ 📂AutoEncoder
   ┃ ┗ 📜AutoEncoder.ipynb
   ┣ 📂LSTM
   ┃ ┣ 📜LSTM_for_Submit.ipynb
   ┃ ┗ 📜LSTM_for_Validation.ipynb
   ┣ 📂SDCFEModel
   ┃ ┣ 📜SDCFEModel_for_Submit.ipynb
   ┃ ┗ 📜SDCFEModel_for_Validation.ipynb
   ┗ 📂UNET-Ensemble
   ┗ 📜UNET-Ensemble_for_validation.ipynb
```
## 폴더 설명
- EDA : 통계적인 기법을 사용하여 데이터를 분석한 내용에 대해서 다룹니다.
- MODEL : 최종적으로 가장 성능이 우수한 모델에 대한 코드입니다.
- Trial : 문제를 해결하기 위해 여러 모델링을 시도한 코드입니다.
    - AutoEncoder : AutoEncoder를 이용하여 모델을 구성해보았습니다.
    - LSTM : LSTM layer를 이용하여 장기기억을 끌고가면서 모델을 학습시키면 성능이 더욱 좋을 것이라고 판단하여 이를 이용해 모델을 구성해보았습니다.
    - SDCFEModel
        - Seperate Dense Concat Fold Ensenble Model
        - 연관이 있는 열을 하나로 묶고, 연관성이 있는 그룹들 각각을 Dense layer로 차원을 축소하여 잠재벡터를 만듦.
        - 이 후, 해당 잠재벡터들을 모두 병합한 후, KFOLD 5로 5개의 모델을 각각 생성한 후, 모델들을 저장하고 Ensemble Method를 사용하여 5개의 모델을 모두 앙상블학습 시킴.
    - UNET-Ensemble
        - UNET의 구조를 착안하여 만든 모델로, UNET의 구조처럼 잠재벡터를 단계적으로 만들어나가면서, 단계별로 잠재벡터를 형성한다.

## 파일 설명
- Code.ipynb
    - 기상청 API를 크롤링 하는 코드입니다.
- index.xlsx
    - 기상청 API를 크롤링하면, 열 부분이 해석할 수 없는 영어 코드명으로 나옵니다. 이를 알아볼 수 있는 한글로 변환해주기 위한 정보가 담긴 엑셀 파일입니다.
- Best Model.json
    - 해당 프로젝트에서 이용된 모든 모델이 담겨져 있습니다.
- Data Preprocessing for Solar.json
    - 일조량, 일사량, 전운량을 예측하기 위한 모델이 담겨져 있습니다.
- Solar Power Prediction.json
    - 태양광 발전량을 예측하기 위한 모델이 담겨져 있습니다.
- Wind Power Prediction.json
    - 풍력 발전량을 예측하기 위한 모델이 담겨져 있습니다.
- Code Explanation.docx
    - 코드를 설명하기 위한 워드 파일입니다.
- PT.pptx
    - PT를 위한 ppt 파일입니다. 해당 PT는 9월 15일 14시 슈피겐홀에서 진행하였습니다.

## 결과 및 결언

![슬라이드39](https://user-images.githubusercontent.com/89781598/189540503-e23b15ca-f8f6-4e32-921a-72b431cfd98a.JPG)

## 참고사항
- 데이터의 경우 대회 직후 파기해야 하므로, 깃허브에 올릴 수 없는 점 양해부탁드립니다.

## 문의사항
* email : ajc227ung@gmail.com

