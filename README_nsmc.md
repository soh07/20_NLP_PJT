# KoELECTRA 기반 네이버 영화 리뷰 감성 분석 
 네이버 영화 리뷰([NSMC](https://github.com/e9t/nsmc))의 긍/부정(Polarity) Binary Classification 수행
 
## 모델 소개 ##
## [ELECTRA](https://openreview.net/pdf?id=r1xMH1BtvB) (Efficiently Learning an Encoder that Classifies Token Replacements Accurately) ##
<p float="left" align="center">
    <img width="500" src="https://weeklypythoncode.files.wordpress.com/2020/03/image4.gif?w=400&zoom=2" />  
</p>

 - ICLR 2020에서 구글 리서치 팀에서 발표한 새로운 pre-training 기법을 적용한 language model.
 - 학습 효율을 향상시키기 위해 'Replaced Token Detection,(RTD)'이라는 새로운 pre-training 테스크를 통해 학습하며, 
   동일 조건(모델 크기, 데이터, 컴퓨팅 리소스)에서 BERT의 성능을 능가했습니다.
 - RTD는 generator를 이용해 실제 입력의 일부 토큰을 fake 토큰으로 바꾸고, 각 토큰이 실제 입력에 있는 original 토큰인지 generator가 생성해낸 replaced 토큰인지 
   discriminator가 맞히는 이진 분류 문제입니다. ELECTRA는 RTD 태스크로 입력의 15%가 아닌 모든 토큰에 대해서 학습하기 때문에 상당히 효율적이면서도 효과적입니다.
   -> GAN(Generative Adversarial Network)의 Generator/Discriminator 개념이 적용된 구조입니다.
 - cf.[BERT] (Bidirectional Encoder Representations from Transformers) : MLM(Masked Language Model) 방식의 pre-trained model.
    - 한 문장에 있는 단어들을 차례대로 학습하는 것이 아니라 양방향으로 학습하면서 마스킹처리(Masked out, 모델이 문장을 이해해서 단어를 예측하도록 “가려놓은” 단어들)된 단어들을 예측하는 모델.
    - 입력 시퀀스의 토큰 중 약 15% 정도를 마스킹하고 이를 복원하는 테스크를 통해 학습합니다.

이 과제에서는,
ELECTRA모델을 한국어로 학습한 KoELECTRA를 적용하였습니다.


## About [KoELECTRA](https://github.com/monologg/KoELECTRA) ##
 - **34GB 한국어 text**로 학습
 - **Wordpiece 사용**
 - 사용 버전 : KoELECTRA-Base-v3
    
```python
from transformers import ElectraModel, ElectraTokenizer

model = ElectraModel.from_pretrained("monologg/koelectra-base-v3-discriminator")
tokenizer = ElectraTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")  #ElectraTokenizer는 Google의 wordpiece인 BertTokenizer와 동일합니다. 
```
|                   |               | Layers | Embedding Size | Hidden Size | # heads |
| ----------------- | ------------: | -----: | -------------: | ----------: | ------: |
| `KoELECTRA-Base`  | Discriminator |     12 |            768 |         768 |      12 |
|                   |     Generator |     12 |            768 |         256 |       4 |


**Pretraining Details**
 - 뉴스, 위키, 나무위키, 모두의 말뭉치 Corpus
 - Vocab Len 35000
 - [[Preprocessing]](./docs/preprocessing.md) 참고

| Model        | Batch Size | Train Steps |   LR | Max Seq Len | Generator Size | Train Time |
| :----------- | ---------: | ----------: | ---: | ----------: | -------------: | ---------: |
| `Base v3`    |        256 |        1.5M | 2e-4 |         512 |           0.33 |        14d |

*1) Tensorflow v2 Model*

```python
from transformers import TFElectraModel

model = TFElectraModel.from_pretrained("monologg/koelectra-base-v3-discriminator", from_pt=True)
```

*2) Tokenizer Example*

```python
>>> from transformers import ElectraTokenizer
>>> tokenizer = ElectraTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")
>>> tokenizer.tokenize("[CLS] 한국어 ELECTRA를 공유합니다. [SEP]")
['[CLS]', '한국어', 'EL', '##EC', '##TRA', '##를', '공유', '##합니다', '.', '[SEP]']
>>> tokenizer.convert_tokens_to_ids(['[CLS]', '한국어', 'EL', '##EC', '##TRA', '##를', '공유', '##합니다', '.', '[SEP]'])
[2, 11229, 29173, 13352, 25541, 4110, 7824, 17788, 18, 3]
```

## Data 분석 과정 ##

### Data Set ###
 - Naver sentiment movie corpus v1.0 => https://github.com/e9t/nsmc
 - Column 정보
    - id) The review id, provieded by Naver
    - document) The actual review
    - label) The sentiment class of the review. (0: negative, 1: positive) 
 - 'ratings_train.txt' 150,000건 리뷰 
 - 'ratings_test.txt' 50,000건 리뷰
  
### Data Split ###
 - 검증 데이터셋 확보를 위해, training.txt의 data를 (학습:검증)=(80:20)으로 분할합니다.
 - [Data 수] 학습:검증:평가 = 120,000 : 30,000 : 50,000
 
```python
from sklearn.model_selection import train_test_split
train_sentences, dev_sentences, train_labels, dev_labels = train_test_split(train_sentences, train_labels,
                                                                            test_size=0.2, random_state=42)
```
### Tokenization ###
```python
from transformers import BertTokenizer
text_tokenizer = ElectraTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")
```
### Building Model ###

| Layer (type)                     | Output Shape         | Param # | Connected to         |
| ---------------------------------| -------------------: | -------:| -------------------: |
| input_word_ids (InputLayer)      | [(None, None)]       |     0   |                      |
| input_mask (InputLayer)          | [(None, None)]       |     0   |                      |
| input_type_ids (InputLayer)      | [(None, None)]       |     0   |                      |
| input_word_ids (InputLayer)      | [(None, None)]       |     0   |                      |
| tf_electra_model (TFElectraMode  | TFBaseModelOutput(la |112330752|input_word_ids[0][0]  |
|                                  |                      |         |input_mask[0][0]      |            
|                                  |                      |         |input_type_ids[0][0]  |
| dense (Dense)                    | (None, None, 128)    |   98342 |tf_electra_model[0][0]|
| dense_1 (Dense)                  | (None, None, 256)    |   33024 |dense[0][0]           |
| classifier (Dense)               | (None, None, 1)      |     257 |dense_1[0][0]         |

### Training Model ###
 - Optimizer : AdamW  (BERT adopts the Adam optimizer with weight decay) 
    -> 매 weight 업데이트마다 learning rate를 일정 비율 감소시켜주는 learning rate schedule 적용
    -> weight decay: gradient descent에서 weight 업데이트 할 때, 이전 weight의 크기를 일정 비율 감소시켜줌으로써 over-fitting 방지
 - learning rate : 2e-5
 - Binary Classification이므로 손실함수는 binary_crossentropy,활성화 함수는 Sigmoid 적용.
 - 평가지표는 Accuracy 
 - Batch Size : 32   (몇 개의 데이터를 보고 한 번의 가중치 업데이트를 수행할 것인지 결정)
 - Epoch : 4         (전체 학습데이터를 한 번씩 학습에 활용한 것이 1 epoch)
 - Hyperparameter Tuning은 별도없이 config의 셋팅을 그대로 사용 
| train data size | 120000|
| steps per epochs| 3750  |
| num train steps | 15000 |
| warmup steps    | 1500  |

### Performance ###
 - 참고로 한 [KoELECTRA](https://github.com/monologg/KoELECTRA) 의 결과와 동일하게 Accuracy 90.6% 확인  
 - 모델 성능 비교를 위해 classic한 LSTM 적용 모델을 만들어 KoELECTRA 적용 모델과의 성능 비교를 하였습니다.
 - LSTM기반 모델 코드는 [LSTM Code](https://github.com/soh07/NLP_PJT/blob/main/99-2.notebook_nsmc_lstm.py) 참고
 - tokenizing은 동일하게 BERT의 Wordpiece 방식 적용하여 비교
 - (Ref1) 
   BERT pre-trained tokenize -> Embedding layer(num_embeddings=tokenizer의 vocab size+1, embedding dim 64, input_length=128) + LSTM + 1-hidden layer + Adam Optimizer + batch size 32 + epoch 4
 - (Ref2) Ref1 + epoch 50
 - (MODEL) KoELECTRA pre-trained tokenize -> + KoELECTRA + 2-hidden layer + AdamW Optimizer + batch size 32 + epoch 4
 
| Model     | Accuracy|
| :-------- | ------: | 
| Ref1      | 0.5035  |
| Ref2`     | 0.9612  |
| MODEL     | 0.9060  |
 - epoch 50번 수행한 LSTM기반 모델도 높은 성능을 보여주지만, 동일 epoch 비교 시, KoELCTRA가 현저하게 더 높은 성능을 보여줍니다. 


## Reference ##
- [ELECTRA](https://github.com/google-research/electra)
- [KoELECTRA](https://github.com/monologg/KoELECTRA)
- [Huggingface Transformers](https://huggingface.co/transformers/model_doc/electra.html#)
- [Tensorflow Tutorials](https://www.tensorflow.org/official_models/fine_tuning_bert#preprocess_the_data)
