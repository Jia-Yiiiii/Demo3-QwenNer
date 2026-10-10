from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import torch
import json


class NERSFTDataset(Dataset):

    def __init__(self, file, tokenizer, max_seq_length, prompt_template, is_train=True):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.prompt_template = prompt_template
        self.is_train = is_train
        self.pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id or 0
        with open(file, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        self.samples = []
        self.preprocess(raw_data)

    def get_full_text(self, sentence, entities):
        prompt_text = self.prompt_template.replace("{sentence}", sentence)
        target_text = json.dumps({"entities": entities}, ensure_ascii=False)
        return prompt_text, target_text

    def preprocess(self, raw_data):
        for item in raw_data:
            sentence = item.get("sentence", "").strip()
            entities = item.get("entities", [])
            if not sentence:
                continue
            prompt_text, target_text = self.get_full_text(sentence, entities)
            prompt_ids = self.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
            left_len = self.max_seq_length - len(prompt_ids) - 1  # -1 留给 eos_token
            if left_len <= 0:
                continue

            target_ids_full = self.tokenizer(
                target_text, add_special_tokens=False
            )["input_ids"]

            if len(target_ids_full) > left_len:
                continue
            target_ids = target_ids_full
            input_ids = prompt_ids + target_ids
            if self.tokenizer.eos_token_id:
                input_ids.append(self.tokenizer.eos_token_id)
            labels = [-100] * len(prompt_ids) + input_ids[len(prompt_ids):]
            self.samples.append({
                "input_ids": input_ids,
                "labels": labels,
                "sentence": sentence,
                "entities": entities,
                "prompt_len": len(prompt_ids)
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    def collate_fn(self, batch):
        input_ids_list = [x["input_ids"] for x in batch]
        labels_list = [x["labels"] for x in batch]
        max_len = max(len(x) for x in input_ids_list)
        left_pad = not self.is_train

        padded_input_ids = []
        padded_labels = []
        padded_mask = []

        for ids, lab in zip(input_ids_list, labels_list):
            pad_len = max_len - len(ids)
            if left_pad:
                padded_input_ids.append([self.pad_id] * pad_len + ids)
                padded_labels.append([-100] * pad_len + lab)
                padded_mask.append([0] * pad_len + [1] * len(ids))
            else:
                padded_input_ids.append(ids + [self.pad_id] * pad_len)
                padded_labels.append(lab + [-100] * pad_len)
                padded_mask.append([1] * len(ids) + [0] * pad_len)

        return {
            "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(padded_mask, dtype=torch.long),
            "labels": torch.tensor(padded_labels, dtype=torch.long),
            "sentences": [x["sentence"] for x in batch],
            "entities": [x["entities"] for x in batch],
            "prompt_len": torch.tensor([x["prompt_len"] for x in batch], dtype=torch.long)
        }

    def get_data_loader(self, batch_size, shuffle=None):
        if shuffle is None:
            shuffle = self.is_train
        return DataLoader(
            dataset=self, batch_size=batch_size, shuffle=shuffle,
            collate_fn=self.collate_fn
        )