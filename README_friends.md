# BERT-based Sentiment Analysis 
 Friends TV scripts & EmotionPush chat logs의 dialogue에 대해 8가지로 labeling된 emoticon의 Multi-label Classification 수행
 
## 모델 소개 ##
## [BERT](https://openreview.net/pdf?id=r1xMH1BtvB) (Bidirectional Encoder Representations from Transformers) ##
<p float="left" align="center">
    <img width="500" src="https://weeklypythoncode.files.wordpress.com/2020/03/image3.gif?w=640&zoom=2" />  
</p>

 - multi layer의 양방향 Transformer 인코더를 기반으로 한 구조로 MLM(Masked Language Model) 방식의 pre-trained model.
    - 한 문장에 있는 단어들을 차례대로 학습하는 것이 아니라 양방향으로 학습하면서 마스킹처리(Masked out, 모델이 문장을 이해해서 단어를 예측하도록 “가려놓은” 단어들)된 단어들을 예측하는 모델.
    - 입력 시퀀스의 토큰 중 약 15% 정도를 마스킹하고 이를 복원하는 테스크를 통해 학습합니다.
 - **Model Architecture**
    - L = Transformer의 layer의 수(Transformer blocks)
    - H = hidden size
    - A = self-attention의 head 수
    - 4H = Transformer의 feed-forward layer의 첫번째 layer의 unit 수
    - 모델 사용 버전 = bert_en_uncased_L-12_H-768_A-12   (L=12, H=768, A=12의 구조)
 - **Input Representation**
    - 주어진 token에 대해서 3가지 값 (Token의 Embedding, Segment의 Embedding, Position의 Embedding)을 더함으로써 입력값으로 사용.
![BERT_Input](https://user-images.githubusercontent.com/71807390/102978166-6e45e100-4547-11eb-8a3a-b991d2447e08.jpg)
    - 30,000개의 token vcabulary를 가지는 WordPiece 임베딩 값(Wu et al., 2016) 사용, split word의 경우 ##으로 사용
    - Positional 임베딩 값으로는 512 토큰 길이에 맞게 학습된 임베딩 값 사용
    - 모든 문장의 첫 토큰은 [CLS](special classification embedding)값을 넣어주고, 해당 토큰에 대응하는 마지막 hidden state(Transformer의 출력값)는 분류 task에서 사용됩니다.
    - 문장 쌍은 하나로 묶여서 하나의 문장으로 만들어지는데, 실제로 다른 이 문장 쌍을 두가지 방법으로 구별합니다. 첫 번쨰는 두 문장 사이에 [SEP](special token)를 넣어주고, 
      .두 번째 방법은 학습된 문장 A 임베딩 값을 앞쪽 문장 모든 token에 더해주고 문장 B 임베딩 값을 뒤쪽 문장 모든 token에 더해줍니다. 
       단일 문장 입력값의 경우 문장 A 임베딩 값만을 사용합니다.


## Data 분석 과정 ##

### Data Set ###
 - EmotionLines dataset ==> http://doraemon.iis.sinica.edu.tw/emotionlines/
 - labeled emotion on every utterance in dialogues from Friends TV scripts and EmotionPush chat logs.
 - [Data 수] 총 3개로 구분, train :dev : test = 720 : 80 : 200
 - Column 정보
    - speaker, utterance, emoticon, annotation 총 4개의 열로 구성 
    - emotion label 총 8개 [non-neutral, neutral, joy, sadness, fear, anger, surprise, disgust]
 
    ![Friends_Data_Format1](https://user-images.githubusercontent.com/71807390/102965738-c96cd900-4531-11eb-9f44-b464e13898fd.png)
 
 
### Preprocessing ###
### Tokenizing & Encoding ###
*Tokenizer*
```python
*Tensorflow v2 Model*

from tensorflow.keras.preprocessing.text import Tokenizer

def create_tokenizer_from_hub_module():
  return bert.tokenization.FullTokenizer(
    vocab_file=os.path.join(gs_folder_bert, "vocab.txt"),
     do_lower_case=True)
```



### Building Model ###
 - bert layer 이후 Dropout 0.5 추가  (*bert layer에 기본적으로 dropout이 0.1로 모두 적용*) 
 
| Layer (type)                     | Output Shape         | Param # | Connected to         |
| ---------------------------------| -------------------: | -------:| -------------------: |
| input_word_ids (InputLayer)      | [(None, None)]       |     0   |                      |
| input_mask (InputLayer)          | [(None, None)]       |     0   |                      |
| input_type_ids (InputLayer)      | [(None, None)]       |     0   |                      |
| keras_layer_2 (KerasLayer)       | [(None, 768), |(None, 109482241|input_word_ids[0][0]  |
|                                  |                      |         |input_mask[0][0]      |            
|                                  |                      |         |input_type_ids[0][0]  |
| dropout_12 (Dropout)             | (None, 768      )    |   0     |keras_layer_2[0][0]   |
| dense_4 (Dense)                  | (None, 128)          |   98342 |dropout_12[0][0]      |
| dense_5 (Dense)                  | (None, 256)          |   33024 |dense_4[0][0]         |
| classifier (Dense)               | (None, 8)            |    2056 |dense_5[0][0]         |

### Training Model ###
 - Optimizer : AdamW  (BERT adopts the Adam optimizer with weight decay) 
    -> 매 weight 업데이트마다 learning rate를 일정 비율 감소시켜주는 learning rate schedule 적용
    -> weight decay: gradient descent에서 weight 업데이트 할 때, 이전 weight의 크기를 일정 비율 감소시켜줌으로써 over-fitting 방지
 - 추가로 먼저 0에서 워밍업 한 다음 0으로 감소하는 learning rate schedule 적용합니다.
 - learning rate : 2e-5
 - Multi-class Classification이므로 손실함수는 CategoricalCrossentropy, 활성화 함수는 softmax 적용.
 - 평가지표는 Accuracy 
 - Batch Size : 32   (몇 개의 데이터를 보고 한 번의 가중치 업데이트를 수행할 것인지 결정)
 - Epoch : 4         (전체 학습데이터를 한 번씩 학습에 활용한 것이 1 epoch)
 - Hyperparameter Tuning은 별도없이 config의 셋팅을 그대로 사용 
 **info**
    - train data size : 10561
    - steps per epochs: 330
    - num train steps : 1320 
    - warmup steps : 132  

### Performance ###
 - Accuracy 0.6031
 - hyperparameter tuning을 통해 추가 성능 개선 필요 (dropout, learning rate, batch size, epoch) 
## Reference ##
- [Huggingface Transformers](https://huggingface.co/transformers/model_doc/bert.html?highlight=bert##)
- [Tensorflow Tutorials](https://www.tensorflow.org/official_models/fine_tuning_bert#preprocess_the_data)
