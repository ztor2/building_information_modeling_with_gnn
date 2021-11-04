# Bullding Information Modeling with Graph Convolution Networks
<br>그래프 합성곱 신경망을 이용한 건물 정보 모델링 정합성 검토.<br>

<br>

### 연구 배경 및 목적
- 이 저장소의 파일은 한국연구재단 주관의 <2020년 중견연구 신규과제>에서 서울과학기술대학교 건설시스템공학과와 데이터사이언스학과가 협력하여 수행한 **'3차원 딥러닝과 확률그래프모형을 이용한 건축 BIM 모델 부재·공간 정합성 검토 및 지능적 부재 추천 시스템 개발'** 프로젝트의 일부이다.
- 이 프로젝트는 머신러닝 및 딥러닝 방법론을 건물 정보 모델링(bullding information modeling, BIM)에 적용해 그 정합성 및 활용도를 증대시키는 것을 목표로 한다.
- 특히, BIM에서 추출한 건물 내 각 공간에 포함되는 부재 정보를 이용해 공간 간의 인접 관계를 추출한 뒤, 이를 공간의 특성치 데이터와 결합해 그래프 합성곱 신경망(graph convolutional networks, GCN)으로 분류를 수행하고 이 결과를 활용해 공간의 유형 정보 정합성을 증가시키고자 한다. 

<br>

### 연구 방법
1. 건물 정보 모델링 소프트웨어를 이용해, 건물 내 공간의 각종 측정치 정보를 수집한다.
2. 같은 방법으로, 각 공간에 포함된 부재들의 정보를 출한다.
3. 공간과 공간에 포함된 부재들의 관계를 행렬로 구축한다.
4. 공간-부재 행렬에서 공간 간 부재의 유사도를 계산해 공간-공간 행렬을 다시 구축한다. 이 때, Jaccard 유사도 또는 cosine 유사도를 이용할 수 있다.
5. 공간-공간 행렬의 밀도를 확인하고, 지나치게 밀도가 높을 시 적절한 cut-off를 설정해 일정 값 이하의 유사도는 0으로 치환한다(sparsification).
6. 데이터의 단순화를 위해, 공간-공간 행렬의 각 값을 0 또는 1로 이진화한다(binarization).
7. 추출된 공간-공간 인접행렬과 공간의 특성 행렬을 이용해 GCN으로 분류를 수행한다.
8. GCN 분류 결과를 활용해 건물 정보의 정합성을 확인하고, 오류가 발견되면 정보를 업데이트한다. 

<p align="center"> <img src="https://i.esdrop.com/d/fha5flk1blzo/8R2esflGXg.png" width="100%" align="center"> </p>
<p align="center">  <b> 그림 1. </b> 건물 정보 모델링을 이용해 공간 데이터를 추출 후 GCN 공간 분류 프로세스. </p>

<br>

### 코드 파일 설명
- **GCN 모델 파트**<br>

**GCN.ipynb:** 추출된 공간 간 인접 정보와 공간의 특성 정보를 이용해 GCN을 수행하고, 정확도와 precision/recall 결과, confusion matrix를 확인한다.<br>
**GCN_iter.ipynb:** GCN을 지정한 횟수만큼 반복해서 수행하고 모든 결과를 저장한다.<br>
**model.py:** GCN 수행에 필요한 GCN 모델 및 layer 함수를 포함한다.<br>
**utils.py:** GCN 수행에 필요한 각종 함수를 포함한다. 데이터 로드, 인코딩, 전처리 등의 함수가 정의되어 있다.<br>

<br>

- **데이터 및 기타 파트**<br>

**learning_curve.ipynb:** GCN 수행 후 정확도 및 loss 결과를 이용해 learning curve plot을 그려 학습 양상을 확인한다.<br>
**space_preprocessing.ipynb:** 공간 데이터 전처리 코드 파일로, 공간-부재 정보로부터 인접 행렬을 구축하고 공간 특성 정보를 전처리해 각각 sparse matrix로 변환 후 pickle 형식 파일로 저장한다.<br>
**space_adj.csv:** 공간-부재 인접 관계가 edgelist 형태로 저장된 raw 데이터.<br>
**space_feat.csv:** 공간의 특성 정보 raw 데이터. 10개의 측정치 등의 특성이 포함된다.<br>
**space.graph.jac / space.graph.cos:** 전처리를 거쳐 저장된 공간-공간 인접 행렬. space.graph.jac 파일은 Jaccard 유사도가 사용되었고, space.graph.cos 파일은 cosine 유사도가 사용되었다.<br>
**space.feature:** 전처리를 거쳐 저장된 공간 특성 정보.<br>
**space.labels:** 전처리를 거쳐 저장된 공간의 label 정보.<br>
**train_accs.pkl, train_losses.pkl, val_accs.pkl, val_losses.pkl:** learning curve를 그리기 위해 저장된 정확도 및 loss 정보가 pickle 형식 파일.

<br>

### 실험 결과



