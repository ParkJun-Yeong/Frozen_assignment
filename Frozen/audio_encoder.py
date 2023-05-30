import torch
import torch.nn as nn

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CPU = torch.device("cpu")
A100 = torch.device("cuda")

class AudioEncoder(nn.Module):
    def __init__(self, acfg):
        super(AudioEncoder, self).__init__()
        self.model = None
        self.audio_processor = None
        self.n_tokens = acfg.n_tokens
        self.acfg = acfg

        # cfg = {"audio_model": {"architecture": "wav2vec2_conformer"}}

        architecture = acfg.architecture

        if (architecture == "rnn") | (architecture == "lstm") | (architecture == "gru"):
            self._rnn(architecture)

        if architecture == "wav2vec2_conformer":
            self.load_wav2vec2_conformer()
        if architecture == "wav2vec2":
            self.load_wav2vec2()
        if architecture == "hubert":
            self.load_hubert()

        # opt: 5120, gpt2-medium: 1024
        self.lan_model_dim = 5120
        self.max_seq_len = 1

        if acfg.down_sample.method == "lstm":
            # self.proj_lstm = nn.LSTM(input_size=acfg.hidden_dim, hidden_size=self.lan_model_dim, num_layers=1, bidirectional=False)
            self.proj_lstm = nn.LSTM(input_size=acfg.hidden_dim, hidden_size=acfg.hidden_dim, num_layers=1, bidirectional=False)
            self.proj_fc = nn.Linear(in_features=self.max_seq_len * acfg.hidden_dim, out_features=acfg.n_tokens * self.lan_model_dim)
        if acfg.down_sample.method == "cnn":
            self.proj_cnn = nn.Conv1d(acfg.hidden_dim,self.lan_model_dim,
                                      kernel_size=acfg.down_sample.reduce_factor, stride=acfg.down_sample.reduce_factor)

    def forward(self, audios: list, sampling_rate):
        padded_audios = self.audio_processor(audio=audios, sampling_rate=sampling_rate, padding=True,  return_tensors="pt")
        # if (inputs["input_values"].shape[0] == 1) & (len(inputs["input_values"].shape) >= 3):     # no batch
        #     inputs["input_values"] = inputs["input_values"].view(1, -1).to(device)
        # elif len(inputs["input_values"] >= 4):                                      # batch
        #     inputs["input_values"] = inputs["input_values"].view(inputs["input_values"].shape[-3], 1, -1).to(device)
        #
        # inputs["attention_mask"] = inputs["attention_mask"].to(device)
        # outputs = self.model(**inputs)
        # padded_audios["input_values"] = padded_audios["input_values"].squeeze(0)
        # padded_audios["attention_mask"] = padded_audios["attention_mask"].squeeze(0)
        padded_audios = padded_audios.to(A100)
        self.model = self.model.to(A100)
        outputs = self.model(**padded_audios)       # (batch, seq, dim)
        if self.acfg.down_sample.method == "lstm":
            out = self.proj_lstm(outputs.last_hidden_state.to(A100))[0][:, -1, :]         # 마지막 30개 시퀀스 추출
            out = out.view(out.shape[0], -1)                                        # (batch, 30*1024)
            out = self.proj_fc(out)                                                  # (batch, 30*1024) -> (batch, n_tokens*1024)
            res = out.view(out.shape[0], -1, self.lan_model_dim)
        elif self.acfg.down_sample.method == "cnn":
            out = outputs.last_hidden_state.to(A100)
            out = out.transpose(1,2)
            out = self.proj_cnn(out)
            res = out.transpose(1,2)
        else:
            exit()

        return res

    def load_wav2vec(self):
        pass

    def load_conformer(self):
        pass

    def _rnn(self, architecture):
        if architecture == "rnn":
            pass        # rnn 코드

        if architecture == "lstm":
            pass

        if architecture == "gru":
            pass

    def load_wav2vec2_conformer(self):
        from transformers import Wav2Vec2ConformerModel, AutoProcessor, Wav2Vec2ConformerConfig
        self.audio_processor = AutoProcessor.from_pretrained("facebook/wav2vec2-conformer-rope-large-960h-ft")
        self.model = Wav2Vec2ConformerModel.from_pretrained("facebook/wav2vec2-conformer-rope-large-960h-ft").to(CPU)

    def load_wav2vec2(self):
        from transformers import Wav2Vec2Model, AutoProcessor
        self.audio_processor = AutoProcessor.from_pretrained("facebook/wav2vec2-large-960h")
        self.model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-large-960h")

    def load_hubert(self):      # wav2vec2와 크기 거의 동일
        from transformers import HubertModel, AutoProcessor
        self.audio_processor = AutoProcessor.from_pretrained("facebook/hubert-large-ls960-ft")
        self.model = HubertModel.from_pretrained("facebook/hubert-large-ls960-ft")
