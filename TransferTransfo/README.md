# MPTI - Iron Man🤖
Meet Persona Through AI

해당 모델은 MPTi_Ironman 챗봇을 만들 때 사용한 모델입니다. 

데이터 포맷만 맞춰서 모델을 학습시키면 챗봇에 Ironman이 아닌 다른 페르소나도 입힐 수 있습니다:)

## Data format
#### 구체적인 전처리 방법은 영상과 sample로 올려둔 데이터를 확인해주세요.
![data_presentation](https://user-images.githubusercontent.com/63901494/126024999-46c550f1-169c-45c8-b079-342f91ef1648.png)
<br>
* input으로 persona가 들어가기 때문에 다양한 persona와 대화를 한 데이터가 필요합니다.
* 해당 모델을 학습시킬 시 huggingface에서 해당 모델용으로 공개한 데이터셋을 활용하였습니다.
* utterance의 candidate는 챗봇의 답변 후보를 담고 있으며, history는 대화가 진행될수록 상대방의 답변을 저장합니다.<br>

데이터 형식 
```
{
    "personality": ["sentence1", "sentence2", ...],
    "utterances": [
        {"candidates": ["sentence1", "sentence2", ...],
        "history": ["sentence1",],
        },
        {"candidates": ["sentence1", "sentence2", ...],
         "history": ["sentence1", "sentence2",],
         },
        ... 
    ]
}
```

## TransferTransfo
