from torch.utils.data import Dataset
from typing import Union
from pathlib import Path
from torchaudio.datasets import LIBRISPEECH
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import os

nltk.download("stopwords")
nltk.download("punkt")      # tokenize


class LibriSpeech(Dataset):
    def __init__(self, dcfg, prompt, type="train"):           # train = "train-clean-100" or "train-clean-360"/ dev(test) = "dev(test)-clean" or "dev(test)-other"
        super(LibriSpeech, self).__init__()

        if(type != "train") & (type != "valid") & (type != "test"):
            raise TypeError("train/valid/test 중 하나를 입력해주세요.")

        self.type = type

        if not os.path.isdir(dcfg.dataset_path):
            os.makedirs(dcfg.dataset_path, exist_ok=True)

        self.trainset = LIBRISPEECH(dcfg.dataset_path, dcfg.train_url, download=True)
        self.validset = LIBRISPEECH(dcfg.dataset_path, dcfg.valid_url, download=True)
        self.testset = LIBRISPEECH(dcfg.dataset_path, dcfg.test_url, download=True)

        self.train = {"Waveform": list(), "Sample_rate": list(), "Transcript": list(),
                      "Speaker_ID": list(), "Chapter_ID": list(), "Utterance_ID": list(),
                      "Target": list()}
        self.valid = {"Waveform": list(), "Sample_rate": list(), "Transcript": list(),
                      "Speaker_ID": list(), "Chapter_ID": list(), "Utterance_ID": list(),
                      "Target": list()}
        self.test = {"Waveform": list(), "Sample_rate": list(), "Transcript": list(),
                     "Speaker_ID": list(), "Chapter_ID": list(), "Utterance_ID": list(),
                     "Target": list()}

        self.prompt = prompt.upper()   # prompt: "speech recognition: "

        self.extract_element()
        self.remove_stopwords()

    def __len__(self):
        if self.type == "train":
            return len(self.train["Waveform"])
        if self.type == "valid":
            return len(self.valid["Waveform"])
        if self.type == "test":
            return len(self.test["Waveform"])

    def __getitem__(self, idx):
        if self.type == "train":
            return self.train["Waveform"][idx], self.train["Target"][idx]
        if self.type == "valid":
            return self.valid["Waveform"][idx], self.valid["Target"][idx]
        if self.type == "test":
            return self.test["Waveform"][idx], self.test["Target"][idx]

    def remove_stopwords(self):
        nltk.download("stopwords")
        stop_words = set(stopwords.words("english"))
        stop_words = {w.upper() for w in stop_words}

        for i, sentence in enumerate(self.train["Transcript"]):
            res = list()
            tokens = word_tokenize(sentence.upper())
            for w in tokens:
                if w not in stop_words:
                    res.append(w)
            res = " ".join(res)
            self.train["Target"].append(res)

        for i, sentence in enumerate(self.valid["Transcript"]):
            res = list()
            tokens = word_tokenize(sentence.upper())
            for w in tokens:
                if w not in stop_words:
                    res.append(w)
            res = " ".join(res)
            self.valid["Target"].append(res)

        for i, sentence in enumerate(self.test["Transcript"]):
            res = list()
            tokens = word_tokenize(sentence.upper())
            for w in tokens:
                if w not in stop_words:
                    res.append(w)
            res = " ".join(res)
            self.test["Target"].append(res)



    def extract_element(self):
        for tr, vl, ts in zip(self.trainset, self.validset, self.testset):
            self.train["Waveform"].append(tr[0])
            self.train["Sample_rate"].append(tr[1])
            self.train["Transcript"].append(tr[2])
            self.train["Speaker_ID"].append(tr[3])
            self.train["Chapter_ID"].append(tr[4])
            self.train["Utterance_ID"].append(tr[5])

            self.valid["Waveform"].append(vl[0])
            self.valid["Sample_rate"].append(vl[1])
            self.valid["Transcript"].append(vl[2])
            self.valid["Speaker_ID"].append(vl[3])
            self.valid["Chapter_ID"].append(vl[4])
            self.valid["Utterance_ID"].append(vl[5])


            self.test["Waveform"].append(ts[0])
            self.test["Sample_rate"].append(ts[1])
            self.test["Transcript"].append(ts[2])
            self.test["Speaker_ID"].append(ts[3])
            self.test["Chapter_ID"].append(ts[4])
            self.test["Utterance_ID"].append(ts[5])

    def get_prompt(self):
        return self.prompt


from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch):
    aud, trans = zip(*batch)
    
    # 모델 내에서 어탠션 마스크 씌울 예정이므로 아래 코드는 삭제함
    # aud = list(aud)
    # aud = [a.transpose(1,0) for a in aud]
    # audio = pad_sequence(aud, padding_value=0.0)
    # audio = audio.permute(1,2,0)
    # audio = audio.squeeze(1)
    
    # huggingface feature extractor에 넣기 위한 용도
    audio = list()
    for a in aud:
        audio.extend(a.tolist())


    return audio, trans
