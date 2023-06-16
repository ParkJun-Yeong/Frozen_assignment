
# Introduction 
- 현재 연구 중인 모델입니다.

Frozen은 audio와 text를 입력값으로 받는 멀티모달 모델이다.<br>
Large Language Model의 향상된 성능을 활용하여 여러가지 task(few-shot)를 처리할 수 있는 멀티모달 시스템을 고안한다.<br>
기존의 pretrain-finetune 스타일은 특정 task에 대해 높은 성능을 얻을 수 있었지만, task 마다 별도의 모델을 설계해야 했으며 완성된 finetuned model의 copy를 모두 저장해야하므로 resource에 대한 부담이 컸다.<br>
연구 중인 FrozenTune의 배경이 된 Frozen 모델(https://arxiv.org/abs/2106.13884)과 WavPrompt 모델(https://www.isca-speech.org/archive/interspeech_2022/gao22e_interspeech.html)은 Prefix-tuning(https://arxiv.org/abs/2101.00190)의 학습 방법을 멀티모달에 적용했다.<br>
먼저 오디오 인코더로부터 오디오에 대한 feature를 얻는다. 이 이미지를 Language model의 embedding vector와 동일한 차원의 벡터 n개로 압축해 audio prefix를 얻는다.<br>
정답이 될 text transcript는 tokenizing과 text embedding을 거친 후 audio prefix와 이어붙여진다.<br>
학습 이후 inference에서는 demonstrations(optional), audio prefix, prompt를 concat하여 input embedding으로 사용한다.<br>
prompt는 task에 대한 정보를 주는 지시문이다. <br>예를 들어 음성인식용 오디오가 주어진다면, audio encoder에서 오디오를 임베딩 벡터화 하고 prompt로는 "그가 무슨 말을 했니?"와 같은 prompt를 제시할 수 있다.
<br> demonstration은 few-shot learning을 위해 제공하는 예시의 수 이다. 만약 one-shot learning이라면 앞선 음성인식 예시를 수행하기 위해 오디오 음성인식의 예시와 정답을 동일한 input 형태로 앞에 붙여주는 것이다. 

## Algorithm - before revision
pseudo code
![img_1.png](img_1.png)

## Algorithm - after revision
pseudo code
![img_2.png](img_2.png)
![img_3.png](img_3.png)

## 수정사항
1. Gradient Clipping
   - loss가 특정값 이하로 내려가지 않는 현상을 보완하고자 gradient exploding을 막기 위해 해당 방법을 적용
   - 코드(train.py의 line 152)
   - ![img_4.png](img_4.png)

2. Top-k, Top-p Autoregressive generation
    코드(train.py의 generate 함수 - line 289)
    ![img_10.png](img_10.png)
    ![img_11.png](img_11.png)
   - Greedy Generation 방법은 빠르고 컴퓨팅 자원을 적게 소모하지만, 
     동일한 단어를 반복하거나, 매우 단순한('음..' 등의 다조롭지만 자주 사용되는 말들) 
     문장만 생성하는 한계점이 있다. 이를 해결하기 위해 Langauge Model의 특성에 맞도록 Auto-regressive generation 방법으로 문장을 생성한다.
   - Greedy는 다음 단어를 예측할 때 output logits(batch, seq, vocab_size) 텐서를 vocab_size 차원으로 argmax를 적용해 가장 높은 확률을 가진 단어들(batch, seq)만 추출한다.
     Autoregressive는 output logits에서 마지막 시퀀스의 확률분포(batch, 1, vocab_size)만 분리하여 이 다음에 올 '하나의 토큰(단어)'만 추측한다.
     이 next token은 임베딩 된 후 현 상태의 input embeds에 concat 된다. 이 방법을 max seq 길이만큼 반복하면서 next token을 누적으로 붙여가며 다음 단어를 예측한다.
   - Top-k Top-p Search는 가장 높은 확률의 단어만 선택하는 greedy search와 달리 확률이 높은 k개의 토큰들 중 누적확률이 p 이상인 최소한의 집합으로부터 다음 단어를 샘플링하는 것이다.
     단조로운 문장들만 생성할 위험이 있는 greedy search와 달리 해당 방법을 사용하면 다양한 단어의 long sequence를 생성할 수 있다.

## 효과
1. Gradient Clipping
   - Before Revision(train cross entropy loss)
    ![img_5.png](img_5.png)
    5 에서 더 내려가지 않는다.
   
   - After Revision
   ![img_9.png](img_9.png)
   loss가 1점대로 내려갔다.
   
2. Top-k, Top-p Autoregressive generation
   - Before Revision(generated sentence)
    ![img_6.png](img_6.png)
    ![img_7.png](img_7.png)
    의미없는 단어 혹은 모든 샘플에 대해 동일한 문장(좌: preds, 우: target)이 예측된다.
   
   - After Revision
   ![img_8.png](img_8.png)
   정상적으로 좌우 문장이 동일함. (좌: preds, 우: target)
   

# Installation
- Requirements: txt 참고

- Environment
  - os: ubuntu 18.04
  - gpu: A100 80GB
  - system ram: 64GB
  - conda 23.3.1


# Run
기본폴더로 이동

## Train
1. Download LibriSpeech Dataset 
   - dataset/librispeech.yaml 에서 dataset_path를 변경(train,test,dev는 자동으로 다운로드)
   - 데이터셋 디렉토리 구조
     - dataset_path
       - LibriSpeech
         - train-clean-360
         - dev-clean
         - test-clean

```
>> python train.py
```
