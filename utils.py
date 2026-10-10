import torch
import numpy as np
import random
import json
import logging
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training, PeftModel


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class NerUtils:

    @staticmethod
    def load_model(config):
        if config.train_mode == 'qlora':
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        else:
            bnb_config = None
        model = AutoModelForCausalLM.from_pretrained(
            config.model_path,
            torch_dtype=torch.bfloat16,
            quantization_config=bnb_config,
            device_map=config.device,
            trust_remote_code=True
        )
        if config.train_mode == 'qlora':
            model = prepare_model_for_kbit_training(
                model,
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        peft_config = LoraConfig(
            target_modules=config.lora_target_modules,
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout
        )
        model = get_peft_model(model, peft_config)

        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
        model.print_trainable_parameters()
        return model
    @staticmethod
    def load_trained_model(config):
        model = AutoModelForCausalLM.from_pretrained(
            config.model_path,
            torch_dtype=torch.bfloat16,
            device_map=config.device,
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(model, config.output_dir, device_map=config.device)
        return model

    @staticmethod
    def get_entities(text):
        data = json.loads(text)
        entities = []
        for e in data['entities']:
            name = e['name']
            typ = e['type']
            entities.append((name, typ))
        return entities  

    @staticmethod
    def compute_metrics(pred_list, gold_list):
        tp = 0
        fp = 0
        fn = 0
        for p, g in zip(pred_list, gold_list):
            temp_tp = 0
            gold_copy = list(g)
            for ent_pred in p:
                for ent_gold in gold_copy:
                    if ent_pred == ent_gold:
                        temp_tp += 1
                        gold_copy.remove(ent_gold)
                        break
            temp_fp = len(p) - temp_tp
            temp_fn = len(g) - temp_tp
            tp += temp_tp
            fp += temp_fp
            fn += temp_fn

        if tp + fp != 0:
            prec = tp / (tp + fp)
        else:
            prec = 0.0
        if tp + fn != 0:
            rec = tp / (tp + fn)
        else:
            rec = 0.0
        if prec + rec != 0:
            f1 = 2 * prec * rec / (prec + rec)
        else:
            f1 = 0.0
        return {"precision": prec, "recall": rec, "f1": f1}