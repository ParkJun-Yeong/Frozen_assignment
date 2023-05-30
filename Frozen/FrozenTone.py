import os, sys
import torch
import torch.nn as nn

CURRENT_DIR = os.path.dirname(__file__)
sys.path.append(CURRENT_DIR)
from audio_encoder import AudioEncoder
from language_decoder import LanguageModel

CPU = torch.device("cpu")
A100 = torch.device("cuda")

class FrozenTone(nn.Module):
    # def __init__(self, cfg):
    #     super().__init__(self, FrozenTone)
    #
    #     self.audio_encoder = AudioEncoder(cfg)
    #     self.language_decoder = LanguageModel(cfg)
    def __init__(self, cfg):
        super(FrozenTone, self).__init__()

        self.audio_encoder = AudioEncoder(cfg.audio_model).to(CPU)
        self.language_decoder = LanguageModel(cfg.language_model).to(A100)
        self.softmax = nn.Softmax(dim=-1)
        self.cfg = cfg
        self.prompt = cfg.train.prompt.upper()

    def forward(self, audio, transcript, infer: bool, sampling_rate=16000):
        audio_prefix = self.audio_encoder(audio, sampling_rate)

        logit, transcript = self.language_decoder(audio_prefix.to(A100), transcript, self.prompt, infer, self.cfg.audio_model.n_tokens)


        # logit = self.softmax(logit)
        # logit = torch.argmax(logit, dim=-1)

        # audio_prefix = audio_output

        return logit.logits[:, -transcript.shape[-1]:, :], transcript

