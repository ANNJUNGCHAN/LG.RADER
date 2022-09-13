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


## 프로젝트 결과
![image](https://user-images.githubusercontent.com/89781598/189381041-bfd51ac9-5c10-49f9-9b48-a3dba3f9a2fc.png)
- 2등, 한국서부발전(주)시장상

## 프로젝트 설명
<h3 align="center">🪄 해당 프로젝트는 3개의모델로 구성되어 있습니다! 🪄</h3>

![슬라이드4](https://user-images.githubusercontent.com/89781598/189539548-59d43959-ce0f-4185-b209-25c37fa67c11.JPG)

1. 각각의 모델에 대한 설명은 각각의 모델에 해당하는 폴더의 readme를 참고해주세요!<br>
2. 각각의 모델에 해당하는 폴더는 Data Preprocessing for solar, Solar Power Prediction, Wind Power Prediction입니다!
3. 크롤링에 대한 내용 또한 Data Preprocessing에 담겨져 있으니 참고 부탁드리겠습니다!

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

