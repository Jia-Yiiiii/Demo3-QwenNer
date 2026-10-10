import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
from utils import NerUtils, set_seed, compute_metrics
from data_process import NERSFTDataset
from accelerate import Accelerator
import swanlab
import os
import json
import argparse


class Trainer:
    def __init__(self, config):
        set_seed(config.get("seed", 42))
        self.accelerator = Accelerator(
            mixed_precision="bf16",
            gradient_accumulation_steps=config["gradient_accumulation_steps"]
        )
        self.device = self.accelerator.device
        self.config = config
        self.tokenizer = None
        self.model = None
        self.train_dataset = None
        self.train_loader = None
        self.dev_dataset = None
        self.dev_loader = None
        self.eval_dataset = None
        self.eval_loader = None
        self.best_f1 = 0.0
        self.counter = 0

    def load_data(self):
        self.train_dataset = NERSFTDataset(
            file=self.config["train_data"],
            tokenizer=self.tokenizer,
            max_seq_length=self.config["max_len"],
            prompt_template=self.config["prompt_template"],
            is_train=True,
        )
        self.train_loader = self.train_dataset.get_data_loader(batch_size=self.config["batch_size"])

        self.dev_dataset = NERSFTDataset(
            file=self.config["dev_data"],
            tokenizer=self.tokenizer,
            max_seq_length=self.config["max_len"],
            prompt_template=self.config["prompt_template"],
            is_train=False,
        )
        self.dev_loader = self.dev_dataset.get_data_loader(batch_size=self.config["dev_batch_size"])

        self.eval_dataset = NERSFTDataset(
            file=self.config["eval_data"],
            tokenizer=self.tokenizer,
            max_seq_length=self.config["max_len"],
            prompt_template=self.config["prompt_template"],
            is_train=False,
        )
        self.eval_loader = self.eval_dataset.get_data_loader(batch_size=self.config["eval_batch_size"])

    def init_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config["model_path"], trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = NerUtils.load_model(self.config)
        self.model.config.use_cache = False
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = AdamW(trainable_params, lr=self.config["learning_rate"])
        steps_per_epoch = len(self.train_loader) // self.config["gradient_accumulation_steps"]
        total_steps = steps_per_epoch * self.config["epochs"]
        warmup_steps = self.config.get("warmup_steps", int(total_steps * 0.1))
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )

        self.model, self.optimizer, self.train_loader, self.scheduler = self.accelerator.prepare(
            self.model, self.optimizer, self.train_loader, self.scheduler
        )

    def save_model(self):
        self.accelerator.wait_for_everyone()
        if not self.accelerator.is_main_process:
            return

        save_dir = os.path.join(self.config["output_dir"], f"best-f1_{self.best_f1:.4f}")
        os.makedirs(save_dir, exist_ok=True)

        self.accelerator.unwrap_model(self.model).save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)

    def save_checkpoint(self, epoch):
        self.accelerator.wait_for_everyone()
        if not self.accelerator.is_main_process:
            return

        checkpoint = {
            "epoch": epoch,
            "best_f1": self.best_f1,
            "counter": self.counter,
            "model_state_dict": self.accelerator.unwrap_model(self.model).state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
        }
        save_path = os.path.join(self.config["output_dir"], "checkpoint.pt")
        torch.save(checkpoint, save_path)

    def load_checkpoint(self):
        checkpoint_path = os.path.join(self.config["output_dir"], "checkpoint.pt")
        if not os.path.exists(checkpoint_path):
            return 0

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.accelerator.unwrap_model(self.model).load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.best_f1 = checkpoint["best_f1"]
        self.counter = checkpoint["counter"]
        start_epoch = checkpoint["epoch"] + 1
        return start_epoch

    def train(self):
        swanlab.init(project="SFT-NER", experiment_name="SFT-NER")

        start_epoch = self.load_checkpoint()

        for epoch in range(start_epoch, self.config["epochs"]):
            self.model.train()
            total_loss, step_count = 0.0, 0
            bar = tqdm(self.train_loader,
                       disable=not self.accelerator.is_local_main_process,
                       desc=f"Epoch {epoch + 1}")

            for batch in bar:
                with self.accelerator.accumulate(self.model):
                    loss = self.model(**batch).loss
                    total_loss += loss.item()
                    step_count += 1

                    self.accelerator.backward(loss)
                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(self.model.parameters(), self.config["max_grad_norm"])
                        self.optimizer.step()
                        self.scheduler.step()
                        self.optimizer.zero_grad()

            avg_loss = total_loss / step_count
            metrics = self.evaluate(self.dev_loader, f"Epoch {epoch + 1} [Eval]")

            swanlab.log({"train/loss": avg_loss, "eval/f1": metrics["f1"]})

            if metrics["f1"] > self.best_f1:
                self.best_f1 = metrics["f1"]
                self.counter = 0
                self.save_model()
            else:
                self.counter += 1
                if self.counter >= self.config["patience"]:
                    break

            self.save_checkpoint(epoch)

        metrics = self.evaluate(self.eval_loader, "[Test]")
        swanlab.log({"test/f1": metrics["f1"]})
        swanlab.finish()

    def evaluate(self, loader, desc="[Eval]"):
        self.model.eval()
        model = self.accelerator.unwrap_model(self.model)
        model.config.use_cache = True

        all_preds, all_golds = [], []
        with torch.no_grad():
            for batch in tqdm(loader, desc=desc, disable=not self.accelerator.is_local_main_process):
                generated = model.generate(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    max_length=self.config["max_len"],  # 新增：限制总长度
                    max_new_tokens=self.config["max_new_tokens"],
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

                prompt_lens = batch["prompt_len"]  # 新增：用prompt_len定位
                for i in range(batch["input_ids"].shape[0]):
                    start = prompt_lens[i].item()
                    pred_text = self.tokenizer.decode(generated[i][start:], skip_special_tokens=True)
                    all_preds.append(NerUtils.get_entities(pred_text))
                    all_golds.append(batch["entities"][i])  # 新增：直接用原始entities

        model.config.use_cache = False

        all_preds = self.accelerator.gather_for_metrics(all_preds)
        all_golds = self.accelerator.gather_for_metrics(all_golds)
        metrics = compute_metrics(all_preds, all_golds)

        self.accelerator.unwrap_model(self.model).train()
        return metrics

    def run(self):
        self.init_model()
        self.load_data()
        self.train()
        del self.model, self.optimizer, self.scheduler
        torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    args = parser.parse_args()

    with open(args.config_path, "r", encoding="utf-8") as f:
        config_dict = json.load(f)

    trainer = Trainer(config_dict)
    trainer.run()