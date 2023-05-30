import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
# from model.gpt2 import GPT2LMHeadModel

device = torch.device("cuda")


class LanguageModel(nn.Module):
    def __init__(self, lcfg):
        super(LanguageModel, self).__init__()

        self.model = None
        self.tokenizer = None
        self.lcfg = lcfg

        architecture = lcfg.architecture

        if architecture == "gpt2":
            self.load_gpt2()

        if architecture == "opt":
            self.load_opt()

    def load_gpt2(self):
        from model.gpt2 import GPT2LMHeadModel
        from transformers import GPT2Tokenizer, GPT2Config

        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2-" + self.lcfg.size)

        # gpt2_config = GPT2Config.from_pretrained(self.cfg.architecture + "-" + self.cfg.size)
        # gpt2_config["max_position_embeddings"] = 50000
        self.model = GPT2LMHeadModel.from_pretrained("gpt2-" + self.lcfg.size, pad_token_id=self.tokenizer.eos_token_id)

        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # model freeze
        for para in self.model.parameters():
            para.requires_grad = False

        # config = torch.hub.load('huggingface/pytorch-transformers', 'config', 'gpt2')

    def load_opt(self):
        from model.opt import OPTForCausalLM
        from transformers import AutoTokenizer, AutoConfig

        self.tokenizer = AutoTokenizer.from_pretrained("facebook/opt-"+self.lcfg.size)
        # config = AutoConfig.from_pretrained("facebook/opt-"+self.lcfg.size)
        # config.vocab_size += 1
        self.model = OPTForCausalLM.from_pretrained("facebook/opt-"+self.lcfg.size)

        self.model.config.vocab_size += 1

        for para in self.model.parameters():
            para.requires_grad = False

    def forward(self, audio_prefix, transcripts, prompt, infer, n_tokens):
        # self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})

        prompts = [prompt + demons for demons in transcripts]
        padded_transcripts = self.tokenizer(prompts, padding=True, return_tensors="pt")
        # padded_transcripts["input_ids"] = torch.tensor(padded_transcripts["input_ids"])
        # padded_transcripts["attention_mask"] = torch.tensor(padded_transcripts["attention_mask"])
        target_tokenized = self.tokenizer(transcripts, padding=True, return_tensors="pt")

        if infer:
            if self.lcfg.architecture == "gpt2":
                special_transcripts = [prompt + ('<|endoftext|>' * self.lcfg.valid_special_input_len)] * audio_prefix.shape[0]      # batch 수 만큼
            elif self.lcfg.architecture == "opt":
                special_transcripts = [prompt + ('</s>' * self.lcfg.valid_special_input_len)] * audio_prefix.shape[0]      # batch 수 만큼
            padded_transcripts = self.tokenizer(special_transcripts)
            padded_transcripts["input_ids"] = torch.tensor(padded_transcripts["input_ids"])
            padded_transcripts["attention_mask"] = torch.tensor(padded_transcripts["attention_mask"])

        padded_transcripts = padded_transcripts.to(device)

        # add EOS BOS (22.05.01. 우선 추가하지 말고 해보자)
        # padded_transcripts["input_ids"] = [[self.tokenizer.bos_token_id] + x + [self.tokenizer.eos_token_id] for x in X["input_ids"]]
        
        # tokenizer에서 padding 추가해서 attention mask 차원 맞추차
        # transcript = [torch.tensor(x).unsqueeze(-1) for x in X["input_ids"]]
        # transcript = pad_sequence(transcript)
        # transcript = transcript.permute(1, 2, 0).to(device)


        output = self.model(**padded_transcripts, audio_prefix=audio_prefix, infer=False)
        # attention_mask = torch.ones((audio_prefix.shape[0], n_tokens)).to(device)
        # output = self.model(audio_prefix=audio_prefix, attention_mask=attention_mask, infer=True)

        # generate_ids = self.model.generate(padded_transcripts.input_ids, max_length=100)
        # sentence = self.tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        # for s in sentence:
        #     print("preds: ", s)

        return output, target_tokenized["input_ids"]
