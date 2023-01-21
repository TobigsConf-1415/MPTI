# 📽MPTI💬

[![youtube](https://img.shields.io/badge/Youtube-Link-red)](https://www.youtube.com/watch?v=9yn07h3KN-s)
[![googledrive](https://img.shields.io/badge/report-Link-lightgrey)](https://drive.google.com/file/d/16M2B50oNXQ6xA_wA7wd9vZaDsEUrHC8S/view?usp=sharing)

<p align="center"><img src="https://user-images.githubusercontent.com/68625698/125901312-d549f7e1-e7da-4b35-9ba7-aa4ddff46e40.PNG"></p>
<br>

**MPTI**는 **Meet Persona Through AI**의 줄임말로 영화 속 캐릭터와의 대화형 챗봇 서비스입니다.

MPTI_Sherlock은 DialoGPT를 사용하여, MPTI_Ironman은 GPT2를 사용하여 만들었습니다.

서비스에 대한 자세한 내용은 그림 위의 링크를 참고해주세요.

## Data

학습에 사용된 데이터는 대본을 바탕으로 구성되었으며, 상업적으로 시용할 의도가 전혀 없음을 밝힙니다:)

각각의 챗봇에 사용한 데이터의 형식이 서로 다릅니다. 데이터의 형식은 개별 README를 통해 확인할 수 있습니다.

* [Sherlock](https://github.com/TobigsConf-1415/MPTI/tree/main/DialoGPT)
* [Iron Man](https://github.com/TobigsConf-1415/MPTI/tree/main/TransferTransfo)

## Usage

모델의 용량 문제로 인해, 16GB 이상의 GPU 환경이나 Colab에서 GPU를 사용하기를 적극 권장합니다. Colab 환경의 경우, [런타임] - [런타임 유형 변경] - [GPU]로 설정을 바꿔주세요.

### 1. Installation
```
git clone https://github.com/TobigsConf-1415/MPTI.git
cd MPTI
pip install -r requirements.txt
```

### 2. Train

각 모델을 아래의 순서대로 학습시켜주세요
```
cd DialoGPT
python main.py
```

```
cd TransferTransfo
python -m spacy download en
python train.py
```

### 3. Inference
```
python demo.py
```

## Result

<img src="https://user-images.githubusercontent.com/68625698/125901038-09626877-2371-423e-8533-bbec6a1880f2.PNG" width="800"><br>

## Demo

| Sherlock | Iron Man |
|---|---|
|<img src="images/sherlock_sample.gif" width="300" height="600">|<img src="images/ironman_sample.gif" width="300" height="600">|

## Reference
* Golsun, DialogRPT, 2020, https://github.com/golsun/DialogRPT
* HuggingFace, transfer-learning-conv-ai, 2020, https://github.com/huggingface/transfer-learning-conv-ai
* Wolf, T., Sanh, V., Chaumond, J., & Delangue, C. (2019). Transfertransfo: A transfer learning approach 
for neural network based conversational agents.
* Zhang, Y., Sun, S., Galley, M., Chen, Y. C., Brockett, C., Gao, X., ... & Dolan, B. (2019). Dialogpt: Large-scale generative pre-training for conversational response generation.
* Thomas Wolf, How to build a State-of-the-Art Conversational AI with Transfer Learning, 2019, https://medium.com/huggingface/how-to-build-a-state-of-the-art-conversational-ai-with-transfer-learning-2d818ac26313
* https://towardsdatascience.com/make-your-own-rick-sanchez-bot-with-transformers-and-dialogpt-fine-tuning-f85e6d1f4e30

## Contributor 🕵️‍♂️
<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->

<table>
  <tr>
    <td align="center"><a href="https://github.com/Seyoung-Jung"><img src="https://user-images.githubusercontent.com/68625698/125951458-637a621f-e823-4b96-95d6-1a5cc62b3714.jpg" width="200" height="200"><br /><sub><b>Seyoung Jung</b></sub></td>
    <td align="center"><a href="https://github.com/Jeong-JaeYoon"><img src="https://user-images.githubusercontent.com/68625698/125892690-46621db4-d033-4fa3-a320-eceb52610eb8.jpg" width="200" height="200"><br /><sub><b>Jaeyoon Jeong</b></sub></td>
    <td align="center"><a href="https://github.com/treblenalto"><img src="https://user-images.githubusercontent.com/68283760/125950085-509a9fe9-4dac-48dc-a8a2-ded2a4bd9f63.jpg" width="200" height="200"><br /><sub><b>Taehee Kim</b></sub></td>
  </tr>
</table>

<table>
  <tr>
    <td align="center"><a href="https://github.com/ltnalsxl"><img src="https://user-images.githubusercontent.com/68283760/125949586-bcf6297e-4840-4b0d-8eda-bad8b90d54b1.jpg" width="200" height="200"><br /><sub><b>Sumin Lee</b></sub></td>
    <td align="center"><a href="https://github.com/Junhyeok1015"><img src="https://user-images.githubusercontent.com/68625698/125951568-a6a08603-b5b0-4230-8a04-5cc80641cab4.jpg" width="200" height="200"><br /><sub><b>Junhyeok Cho</b></sub></td>
    <td align="center"><a href="https://github.com/hbjk0305"><img src="https://user-images.githubusercontent.com/68283760/125949229-81d9fad7-aba3-4754-af14-342ca9e22d7e.jpg"
 width="200" height="200"><br /><sub><b>JinKyoung Hwangbo</b></sub></td>
  </tr>
</table>
