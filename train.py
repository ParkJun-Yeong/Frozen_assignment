import os, sys
import warnings
import logging
import random
import gc
from dadaptation.dadapt_sgd import DAdaptSGD

import hydra
from omegaconf import OmegaConf, ListConfig
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import SGD, Adam
from torch.nn.functional import pad

from tqdm import tqdm
from omegaconf import DictConfig
import wandb
from torchmetrics import CharErrorRate
from datetime import datetime

# CURRENT_DIR = os.path.dirname(__file__)
# sys.path.append(os.path.join(CURRENT_DIR, "/"))
from FrozenTune.FrozenTone import FrozenTone
from dataset.librispeech.librispeech import LibriSpeech, collate_fn

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

CPU = torch.device("cpu")
A100 = torch.device("cuda")

random_seed = 2000
torch.manual_seed(random_seed)
random.seed(random_seed)

# config_path = os.path.join(CURRENT_DIR, "configs")
config_path = os.path.join(os.path.dirname(__file__), "configs")
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logger = logging.getLogger("run.log")
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
# ch.setLevel((logging.INFO))


def train(model, optimizer, train_dataloader, valid_dataloader, loss_fn, epochs, CER, cfg, lr_scheduler, pad_token_id, start_epoch):
    best_model = None
    best_loss = None
    best_cer = None

    # model, optimizer = torch.cuda.amp.initialize(model, optimizer, opt_level="O1")

    # prev_epoch_loss = None

    for epoch in range(start_epoch, epochs):
        loss_hist = list()
        cer_hist = list()
        # i =0

        model.train()

        # if ((optimizer.param_groups[0]["lr"] >= cfg.train.minimum_lr) & (epoch % cfg.train.lr_schedule_epoch == 0) & (epoch != 1)) | (epoch == 6):         # minimum
        # if (epoch % cfg.train.lr_schedule_epoch == 0) & (epoch != 0):  # minimum
        #     lr_scheduler.step()
        if cfg.train.warmup_exist & (cfg.train.optimizer != "DadaptSGD"):
            if epoch == cfg.train.warmup_epoch:
                optimizer.param_groups[0]["lr"] = cfg.train.lr


        print(f"********* epoch {epoch} *********")
        for X, y in tqdm(train_dataloader, desc="Train iteration starts"):
            # if i >= 0:
            #     continue
            # i += 1
            if cfg.train.amp:
                with torch.cuda.amp.autocast(enabled=True):
                    output, y_ids = model(audio=X, transcript=y, infer=False)
                # try:
                #     output, y_ids = model(audio=X, transcript=y, infer=False)
                # except RuntimeError as e:
                #     print(e)
                #     print(f"Dataset (y): {y}")
                #     continue

                # padding = (0, output.shape[-2] - y_ids.shape[-1])

                # y_pads = pad(y_ids, padding, mode="constant", value=pad_token_id).to(A100)
                    output = output.squeeze(1)
                    output = output.permute(0, 2, 1)
                # y_pads = y_pads.squeeze(1)

                    if cfg.train.loss_fn == "cross_entropy":
                        loss = loss_fn(output.to(A100), y_ids.to(A100))

                    if cfg.train.loss_fn == "ctc":
                        output = output.permute(1, 0, 2)
                        input_lengths = torch.full((output.shape[1],), fill_value=output.shape[0], dtype=torch.long)

                        lengths = list()
                        for y_id in y_ids:
                            lengths.append(len([i for i in y_id if i != pad_token_id]))

                        target_lengths = torch.tensor(lengths)
                        loss = loss_fn(output.to(A100), y_ids.to(CPU), input_lengths.to(A100), target_lengths.to(CPU))

                        generated_text = generate(model, X, y_ids)

                loss_hist.append(loss.item())

                scaler = torch.cuda.amp.GradScaler()
                optimizer.zero_grad()
                # scaler.scale(loss).backward()
                # optimizer.step()
                scaler.step(optimizer)
                scaler.update()
            else:
                output, y_ids = model(audio=X, transcript=y, infer=False)
                # try:
                #     output, y_ids = model(audio=X, transcript=y, infer=False)
                # except RuntimeError as e:
                #     print(e)
                #     print(f"Dataset (y): {y}")
                #     continue

                # padding = (0, output.shape[-2] - y_ids.shape[-1])

                # y_pads = pad(y_ids, padding, mode="constant", value=pad_token_id).to(A100)
                output = output.squeeze(1)
                output = output.permute(0, 2, 1)
                # y_pads = y_pads.squeeze(1)

                if cfg.train.loss_fn == "cross_entropy":
                    loss = loss_fn(output.to(A100), y_ids.to(A100))

                if cfg.train.loss_fn == "ctc":
                    output = output.permute(1, 0, 2)
                    input_lengths = torch.full((output.shape[1],), fill_value=output.shape[0], dtype=torch.long)

                    lengths = list()
                    for y_id in y_ids:
                        lengths.append(len([i for i in y_id if i != pad_token_id]))

                    target_lengths = torch.tensor(lengths)
                    loss = loss_fn(output.to(A100), y_ids.to(CPU), input_lengths.to(A100), target_lengths.to(CPU))

                loss_hist.append(loss.item())

                optimizer.zero_grad()
                loss.backward()
                # Add gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
                optimizer.step()


            wandb.log({"train lr": optimizer.param_groups[0]["lr"]})

            decoded_strs = generate(model, X, y_ids)
            # decoded_strs = decode_to_string(cfg.language_model.architecture, output)
            # # decoded_strs = [s.upper() for s in decoded_strs]
            # decoded_strs = [s.lower() for s in decoded_strs]


            print("-- decoded string in this iteration --")
            for i, s in zip(y, decoded_strs):
                print("target: ", i)
                print("preds: ", s)
            cer = CER(decoded_strs, y)
            cer_hist.append(cer)
            wandb.log({"train_loss/iter": loss.item(), "train_cer/iter": cer})

            # print(f"train_loss/iter: {loss}, train_cer/iter: {cer}")

            if best_model == None:
                best_model = model
                best_loss = loss.item()
                best_cer = cer

            if (best_loss > loss.item()) | (best_cer > cer):
                best_model = model

            # i += 1
        # if prev_epoch_loss is None:
        #     prev_epoch_loss = loss.item()

        loss_hist = torch.tensor(loss_hist)
        cer_hist = torch.tensor(cer_hist)

        avg_loss = torch.mean(loss_hist)
        avg_cer = torch.mean(cer_hist)
        wandb.log({"train_loss/epoch": avg_loss, "train_cer/epoch: cer_": avg_cer})
        print(f"train_loss/epoch: {avg_loss}, train_cer/epoch: {avg_cer}")

        if epoch % 5 == 0:
            valid(model=model, valid_dataloader=valid_dataloader, loss_fn=loss_fn, CER=CER,
                  lm_model=cfg.language_model.architecture, pad_token_id=pad_token_id,
                  loss_type=cfg.train.loss_fn)
        
        # 중간 저장 용
        if not os.path.isdir(cfg.train.checkpoint_path):
            os.mkdir(cfg.train.checkpoint_path)

        now = datetime.now()

        if (epoch % 10 == 0) & (epoch != 0):
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.audio_encoder.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": loss,
            }, os.path.join(cfg.train.checkpoint_path, now.strftime("%Y-%m-%d-%H-%M") + "-e" + str(epoch) + ".pt"))

    # best model
    torch.save({
        "model_state_dict": best_model.audio_encoder.state_dict(),
    }, os.path.join(cfg.train.checkpoint_path, "best_model", now.strftime("%Y-%m-%d-%H-%M") + ".pt"))

    wandb.finish()



def valid(model, valid_dataloader, loss_fn, CER, lm_model, pad_token_id, loss_type):
    print("======== validation start =========")
    model.eval()
    loss_hist = list()
    cer_hist = list()

    with torch.no_grad():
        for X, y in tqdm(valid_dataloader, desc="Valid iteration starts"):
            output, y_ids = model(audio=X, transcript=y, infer=True)

            # try:
            #     output, y_ids = model(audio=X, transcript=y, infer=True)
            # except RuntimeError as e:
            #     print(e)
            #     continue

            padding = (0, output.shape[-2] - y_ids.shape[-1])
            y_pads = pad(y_ids, padding, mode="constant", value=pad_token_id)

            output = output.squeeze(1)
            output = output.permute(0, 2, 1)
            y_pads = y_pads.squeeze(1)

            if loss_type == "cross_entropy":
                # padding = (0, output.shape[-2] - y_ids.shape[-1])
                # y_pads = pad(y_ids, padding, mode="constant", value=pad_token_id)
                #
                # output = output.squeeze(1)
                # output = output.permute(0, 2, 1)
                # y_pads = y_pads.squeeze(1)

                loss = loss_fn(output.to(CPU), y_pads.to(CPU))

            if loss_type == "ctc":
                output = output.permute(1, 0, 2)
                input_lengths = torch.full((output.shape[1],), fill_value=output.shape[0], dtype=torch.long).to(A100)

                lengths = list()
                for y_id in y_ids:
                    lengths.append(len([i for i in y_id if i != pad_token_id]))

                target_lengths = torch.tensor(lengths)
                loss = loss_fn(output.to(CPU), y_pads.to(CPU), input_lengths.to(CPU), target_lengths.to(CPU))

            decoded_strs = decode_to_string(lm_model, output)

            print("-- decoded string in this iteration --")
            for i, s in zip(y, decoded_strs):
                print("target: ", i)
                print("preds: ", s)

            decoded_strs = [s.capitalize() for s in decoded_strs]
            cer = CER(decoded_strs, y)

            loss_hist.append(loss)
            cer_hist.append(cer)
            wandb.log({"valid_loss/iter": loss, "valid_cer/iter": cer})
        loss_hist = torch.tensor(loss_hist)
        cer_hist = torch.tensor(cer_hist)
        avg_loss = torch.mean(loss_hist)
        avg_cer = torch.mean(cer_hist)

        wandb.log({"valid_loss/epoch": avg_loss, "valid_cer/epoch": avg_cer})
        print(f"valid_loss/epoch: {avg_loss}, valid_cer/epoch: {avg_cer}")

        # calculate_cer()

def generate(model, X, y_ids):
    prompt = "what did the speaker say?"
    max_length = 100
    output_ids = torch.LongTensor([]).view(max_length, 0).to(A100)
    top_k=50
    top_p=0.95

    model.eval()

    with torch.no_grad():
        audio_prefix = model.audio_encoder(X.to(CPU), sampling_rate=16000)

        for i in range(max_length):
            if i == 0:
                logits, _ = model.language_decoder(audio_prefix.to(A100), output_ids,
                                                   prompt, infer=False, n_tokens=8)
            else:
                logits, _ = model.language_decoder(torch.empty(()), output_ids,
                                                   "", infer=False, n_tokens=8)
            logits = logits[:,-1,:]
            logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
            # next_token = torch.argmax(next_token_logit, dim=-1)
            probabilities = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1)
            output_ids = torch.cat((output_ids, next_token), dim=1)

             # = torch.cat([model_inputs["inputs_embeds"], self.get_input_embeddings()(next_token)], dim=1)
            # model_inputs["attention_mask"] = torch.cat([model_inputs["attention_mask"], torch.ones((model_inputs["attention_mask"].size(0), 1)).to(model_inputs["attention_mask"])], dim=1)


    return output_ids

from transformers.generation.logits_process import (TopKLogitsWarper, TopPLogitsWarper)
def top_k_top_p_filtering(
    logits: torch.FloatTensor,
    top_k: int = 0,
    top_p: float = 1.0,
    filter_value: float = -float("Inf"),
    min_tokens_to_keep: int = 1,
) -> torch.FloatTensor:
    """
    Filter a distribution of logits using top-k and/or nucleus (top-p) filtering

    Args:
        logits: logits distribution shape (batch size, vocabulary size)
        top_k (`int`, *optional*, defaults to 0):
            If > 0, only keep the top k tokens with highest probability (top-k filtering)
        top_p (`float`, *optional*, defaults to 1.0):
            If < 1.0, only keep the top tokens with cumulative probability >= top_p (nucleus filtering). Nucleus
            filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        min_tokens_to_keep (`int`, *optional*, defaults to 1):
            Minimumber of tokens we keep per batch example in the output.

    From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    if top_k > 0:
        logits = TopKLogitsWarper(top_k=top_k, filter_value=filter_value, min_tokens_to_keep=min_tokens_to_keep)(
            None, logits
        )

    if 0 <= top_p <= 1.0:
        logits = TopPLogitsWarper(top_p=top_p, filter_value=filter_value, min_tokens_to_keep=min_tokens_to_keep)(
            None, logits
        )

    return logits

def decode_to_string(lm_model, y_hat):
    decoded_strs = list()
    if lm_model == "gpt2":
        from transformers import GPT2Tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")
        selected_tokens = torch.argmax(y_hat, dim=-2)
        selected_tokens = [s[s != tokenizer.eos_token_id] for s in selected_tokens]
        decoded_strs = [tokenizer.decode(tokens, skip_special_tokens=True) for tokens in selected_tokens]

    if lm_model == "opt":
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("facebook/opt-13b")
        selected_tokens = torch.argmax(y_hat, dim=-2)
        selected_tokens = [s[s != tokenizer.pad_token_id] for s in selected_tokens]
        decoded_strs = [tokenizer.decode(tokens, skip_special_tokens=True) for tokens in selected_tokens]

    return decoded_strs

def get_pad_token_id(lm_model, size):
    if lm_model == "gpt2":
        from transformers import GPT2Tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2-"+size)
        pad_token_id = tokenizer.eos_token_id       # 모델 내부에서 이걸로 지정함

        return pad_token_id

    if lm_model == "opt":
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("facebook/opt-"+size)
        pad_token_id = tokenizer.pad_token_id

        return pad_token_id


@hydra.main(config_path=config_path, config_name="default")
def main(cfg: DictConfig):
    warnings.filterwarnings("ignore")
    # cfg = OmegaConf.load(cfg)

    wandb.init(
        project="FrozenTune",
        config=cfg
        # config=dict(**self.config["train"], **self.config["model"], **self.config["audio"])
    )


    model = FrozenTone(cfg)

    resume_data = None
    if cfg.train.resume:
        resume_data = torch.load(os.path.join(cfg.train.resume_from, "2023-05-06-21-23-e10.pt"), map_location="cpu")
        model = model.load_state_dict(resume_data["model_state_dict"])

    #     if cfg.train.optimizer == "SGD":
    #         optimizer = SGD(model.parameters(), lr=cfg.train.lr, momentum=cfg.train.momentum)
    #     if cfg.train.optimizer == "Adam":
    #         optimizer = Adam(model.parameters(), lr=cfg.train.lr)

    if cfg.train.resume:
        lr_init = cfg.train.lr ** 0.25
    else:
        if cfg.train.warmup_exist:
            lr_init = cfg.train.warmup_lr
        else:
            lr_init = cfg.train.lr



    if cfg.train.optimizer == "SGD":
        optimizer = SGD(model.parameters(), lr=lr_init, momentum=cfg.train.momentum)
    if cfg.train.optimizer == "Adam":
        optimizer = Adam(model.parameters(), lr=lr_init)
    if cfg.train.optimizer == "DadaptSGD":
        optimizer = DAdaptSGD(model.parameters(), lr=1.0, growth_rate=1.04)             # optimizer, growth_rate, warmup 셋 중 하나만 쓰면 됨

    if cfg.train.resume:
        optimizer.load_state_dict(resume_data["optimizer_state_dict"])
        start_epoch = resume_data["epoch"] + 1

        gc.collect()
        torch.cuda.empty_cache()

    else:
        start_epoch = 0


    train_dataset = LibriSpeech(type="train", dcfg=cfg.dataset, prompt=cfg.train.prompt)
    valid_dataset = LibriSpeech(type="valid", dcfg=cfg.dataset, prompt=cfg.train.prompt)
    train_dataloader = DataLoader(train_dataset, batch_size=cfg.train.train_batch, shuffle=True, collate_fn=collate_fn)
    valid_dataloader = DataLoader(valid_dataset, batch_size=5, shuffle=False, collate_fn=collate_fn)

    # prompt = train_dataset.get_prompt()



    lambda1 = lambda epoch: 0.25 ** epoch
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)

    pad_token_id = get_pad_token_id(cfg.language_model.architecture, cfg.language_model.size)
    # loss_fn = nn.CrossEntropyLoss(ignore_index=pad_token_id)
    loss_fn = nn.CrossEntropyLoss()


    # blank_token_id = model.language_decoder.model.config.vocab_size - 1
    # loss_fn = nn.CTCLoss(blank=blank_token_id, reduction="mean", zero_infinity=False)

    CER = CharErrorRate()


    train(model=model.to(A100), optimizer=optimizer, train_dataloader=train_dataloader,
          valid_dataloader=valid_dataloader, loss_fn=loss_fn, epochs=cfg.train.epochs, CER=CER, cfg=cfg,
          lr_scheduler=lr_scheduler, pad_token_id=pad_token_id, start_epoch=start_epoch)



if __name__ == "__main__":
    main()

