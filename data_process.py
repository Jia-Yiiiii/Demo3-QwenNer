import json
import torch
from torch.utils.data import DataLoader, Dataset
from template import NER_TEMPLATE

class NERSFTDataset(Dataset):
    def __init__(
        self,
        file,
        tokenizer,
        max_seq_length,
        system_prompt=None,
        is_train=True,
    ):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.system_prompt = system_prompt
        self.is_train = is_train
        self.pad_id = tokenizer.pad_token_id
        if self.pad_id is None:
            if tokenizer.eos_token_id is not None:
                self.pad_id = tokenizer.eos_token_id
            else:
                self.pad_id = 0
        with open(file, 'r', encoding='utf8') as f:
            r_data = json.load(f)
        self.samples = []
        self.preprocess(r_data)

    def preprocess(self, raw_data):
        for item in raw_data:
            if not isinstance(item, dict):
                continue
            sentence = item.get("sentence", "")
            if isinstance(sentence, str):
                sentence = sentence.strip()
            else:
                sentence = ""
            if not sentence:
                continue
            entities = item.get("entities", [])
            output_entities = []
            for entity in entities:
                output_entities.append(
                    {
                        "name": entity["name"],
                        "type": entity["type"],
                    }
                )
            prompt_text = NER_TEMPLATE.replace("{system_content}", self.system_prompt).replace("{sentence}", sentence)
            answer_text = json.dumps({"entities": output_entities}, ensure_ascii=False)
            full_text = prompt_text + answer_text
            full_ids = self.tokenizer(full_text, add_special_tokens=False)["input_ids"]
            prompt_ids = self.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
            if len(full_ids) > self.max_seq_length:
                continue
            prompt_len = len(prompt_ids)
            labels = [-100] * prompt_len + full_ids[prompt_len:]
            gold_entities = []
            for entity in entities:
                gold_entities.append((entity["name"], entity["type"]))
            self.samples.append(
                {
                    "input_ids": full_ids,
                    "prompt_ids": prompt_ids,
                    "labels": labels,
                    "sentence": sentence,
                    "entities": gold_entities,
                    "prompt_len": prompt_len,
                }
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    def collate_fn(self, batch):
        if self.is_train:
            input_ids_list = [sample["input_ids"] for sample in batch]
            labels_list = [sample["labels"] for sample in batch]
        else:
            input_ids_list = [sample["prompt_ids"] for sample in batch]
            labels_list = [[-100] * len(sample["prompt_ids"]) for sample in batch]
        max_len = max(len(ids) for ids in input_ids_list)
        padded_input_ids, padded_labels, padded_mask = [], [], []
        for ids, labels in zip(input_ids_list, labels_list):
            pad_len = max_len - len(ids)
            if self.is_train:
                padded_input_ids.append(ids + [self.pad_id] * pad_len)
                padded_labels.append(labels + [-100] * pad_len)
                padded_mask.append([1] * len(ids) + [0] * pad_len)
            else:
                padded_input_ids.append([self.pad_id] * pad_len + ids)
                padded_labels.append([-100] * pad_len + labels)
                padded_mask.append([0] * pad_len + [1] * len(ids))
        return {
            "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(padded_mask, dtype=torch.long),
            "labels": torch.tensor(padded_labels, dtype=torch.long),
            "sentences": [sample["sentence"] for sample in batch],
            "entities": [sample["entities"] for sample in batch],
            "prompt_len": torch.tensor([sample["prompt_len"] for sample in batch], dtype=torch.long),
        }


    def get_dataloader(self, batch_size, shuffle=None):
        if shuffle is None:
            shuffle = self.is_train
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, collate_fn=self.collate_fn)
