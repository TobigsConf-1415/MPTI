# MPTI - Iron Man๐ค
Meet Persona Through AI

ํด๋น ๋ชจ๋ธ์ MPTI_Ironman ์ฑ๋ด์ ๋ง๋ค ๋ ์ฌ์ฉํ ๋ชจ๋ธ์๋๋ค. 

๋ฐ์ดํฐ ํฌ๋งท๋ง ๋ง์ถฐ์ ๋ชจ๋ธ์ ํ์ต์ํค๋ฉด ์ฑ๋ด์ Ironman์ด ์๋ ๋ค๋ฅธ ํ๋ฅด์๋๋ ์ํ ์ ์์ต๋๋ค:)

## Data format
#### ๊ตฌ์ฒด์ ์ธ ์ ์ฒ๋ฆฌ ๋ฐฉ๋ฒ์ ์์๊ณผ sample๋ก ์ฌ๋ ค๋ ๋ฐ์ดํฐ๋ฅผ ํ์ธํด์ฃผ์ธ์.
![ironman_data](https://user-images.githubusercontent.com/63901494/126025433-948e19d1-48a2-4af8-b101-751772dc7b23.jpg)
<br>
* input์ผ๋ก persona๊ฐ ๋ค์ด๊ฐ๊ธฐ ๋๋ฌธ์ ๋ค์ํ persona์ ๋ํ๋ฅผ ํ ๋ฐ์ดํฐ๊ฐ ํ์ํฉ๋๋ค.
* ํด๋น ๋ชจ๋ธ์ ํ์ต์ํฌ ์ huggingface์์ ํด๋น ๋ชจ๋ธ์ฉ์ผ๋ก ๊ณต๊ฐํ ๋ฐ์ดํฐ์์ ํ์ฉํ์์ต๋๋ค.
* utterance์ candidate๋ ์ฑ๋ด์ ๋ต๋ณ ํ๋ณด๋ฅผ ๋ด๊ณ  ์์ผ๋ฉฐ, history๋ ๋ํ๊ฐ ์งํ๋ ์๋ก ์๋๋ฐฉ์ ๋ต๋ณ์ ์ ์ฅํฉ๋๋ค.<br>

๋ฐ์ดํฐ ํ์ 
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
![ironman_model1](https://user-images.githubusercontent.com/63901494/126025511-d5931657-287a-4cbd-a260-3339c494ff46.jpg)
![ironman_model2](https://user-images.githubusercontent.com/63901494/126025512-f88e7e9f-f6a0-4a58-a0b1-bf6b735b180f.jpg)
![ironman_model3](https://user-images.githubusercontent.com/63901494/126025514-f756decd-5d22-47d0-8f2f-d9e7c7862acc.jpg)
