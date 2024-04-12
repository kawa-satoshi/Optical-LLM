import torch
import numpy as np
from peft import LoraConfig, get_peft_model
from transformers import AutoImageProcessor, TrainingArguments, Trainer, AutoModelForImageClassification
from datasets import load_dataset
import evaluate
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)

# === prepare image processor

model_name = "google/vit-base-patch16-224-in21k"
image_processor = AutoImageProcessor.from_pretrained(model_name)

# === prepare dataset

dataset = load_dataset("food101", split="train[:5000]")
labels = dataset.features["label"].names
label2id, id2label = dict(), dict()
for i, label in enumerate(labels):
    label2id[label] = i
    id2label[i] = label

normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
train_transforms = Compose(
    [RandomResizedCrop(image_processor.size["height"]), RandomHorizontalFlip(), ToTensor(), normalize]
)
val_transforms = Compose(
    [Resize(image_processor.size["height"]), CenterCrop(image_processor.size["height"]), ToTensor(), normalize]
)


def preprocess_train(example_batch):
    example_batch["pixel_values"] = [train_transforms(image.convert("RGB")) for image in example_batch["image"]]
    return example_batch


def preprocess_val(example_batch):
    example_batch["pixel_values"] = [val_transforms(image.convert("RGB")) for image in example_batch["image"]]
    return example_batch


splits = dataset.train_test_split(test_size=0.1)
train_ds, val_ds = splits["train"], splits["test"]
train_ds.set_transform(preprocess_train)
val_ds.set_transform(preprocess_val)

# == prepare model

model = AutoModelForImageClassification.from_pretrained(
    model_name, label2id=label2id, id2label=id2label, ignore_mismatched_sizes=True
)

# === apply lora config

config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["query", "value", "key"],
    lora_dropout=0.1,
    bias="none",
    modules_to_save=["classifier"],
)
lora_model = get_peft_model(model, config)
print(lora_model)
lora_model.print_trainable_parameters()

# == prepare train

args = TrainingArguments(
    f"{model_name}-finetuned-lora-food101",
    remove_unused_columns=False,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-3,
    per_device_train_batch_size=64,
    gradient_accumulation_steps=4,
    per_device_eval_batch_size=64,
    fp16=True,
    num_train_epochs=5,
    logging_steps=10,
    load_best_model_at_end=False,
    metric_for_best_model="accuracy",
    push_to_hub=False,
    label_names=["labels"],
)


def compute_metrics(eval_pred):
    metric = evaluate.load("accuracy")
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return metric.compute(predictions=predictions, references=eval_pred.label_ids)


def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}


trainer = Trainer(
    model=lora_model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=image_processor,
    compute_metrics=compute_metrics,
    data_collator=collate_fn,
)

train_results = trainer.train()
lora_model.save_pretrained(model_name + "_finetuned_food101")

# === evaluate

trainer.evaluate(val_ds)

# === outputs

"""
PeftModel(
  (base_model): LoraModel(
    (model): ViTForImageClassification(
      (vit): ViTModel(
        (embeddings): ViTEmbeddings(
          (patch_embeddings): ViTPatchEmbeddings(
            (projection): Conv2d(3, 768, kernel_size=(16, 16), stride=(16, 16))
          )
          (dropout): Dropout(p=0.0, inplace=False)
        )
        (encoder): ViTEncoder(
          (layer): ModuleList(
            (0-11): 12 x ViTLayer(
              (attention): ViTAttention(
                (attention): ViTSelfAttention(
                  (query): lora.Linear(
                    (base_layer): Linear(in_features=768, out_features=768, bias=True)
                    (lora_dropout): ModuleDict(
                      (default): Dropout(p=0.1, inplace=False)
                    )
                    (lora_A): ModuleDict(
                      (default): Linear(in_features=768, out_features=8, bias=False)
                    )
                    (lora_B): ModuleDict(
                      (default): Linear(in_features=8, out_features=768, bias=False)
                    )
                    (lora_embedding_A): ParameterDict()
                    (lora_embedding_B): ParameterDict()
                  )
                  (key): lora.Linear(
                    (base_layer): Linear(in_features=768, out_features=768, bias=True)
                    (lora_dropout): ModuleDict(
                      (default): Dropout(p=0.1, inplace=False)
                    )
                    (lora_A): ModuleDict(
                      (default): Linear(in_features=768, out_features=8, bias=False)
                    )
                    (lora_B): ModuleDict(
                      (default): Linear(in_features=8, out_features=768, bias=False)
                    )
                    (lora_embedding_A): ParameterDict()
                    (lora_embedding_B): ParameterDict()
                  )
                  (value): lora.Linear(
                    (base_layer): Linear(in_features=768, out_features=768, bias=True)
                    (lora_dropout): ModuleDict(
                      (default): Dropout(p=0.1, inplace=False)
                    )
                    (lora_A): ModuleDict(
                      (default): Linear(in_features=768, out_features=8, bias=False)
                    )
                    (lora_B): ModuleDict(
                      (default): Linear(in_features=8, out_features=768, bias=False)
                    )
                    (lora_embedding_A): ParameterDict()
                    (lora_embedding_B): ParameterDict()
                  )
                  (dropout): Dropout(p=0.0, inplace=False)
                )
                (output): ViTSelfOutput(
                  (dense): Linear(in_features=768, out_features=768, bias=True)
                  (dropout): Dropout(p=0.0, inplace=False)
                )
              )
              (intermediate): ViTIntermediate(
                (dense): Linear(in_features=768, out_features=3072, bias=True)
                (intermediate_act_fn): GELUActivation()
              )
              (output): ViTOutput(
                (dense): Linear(in_features=3072, out_features=768, bias=True)
                (dropout): Dropout(p=0.0, inplace=False)
              )
              (layernorm_before): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
              (layernorm_after): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
            )
          )
        )
        (layernorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
      )
      (classifier): ModulesToSaveWrapper(
        (original_module): Linear(in_features=768, out_features=101, bias=True)
        (modules_to_save): ModuleDict(
          (default): Linear(in_features=768, out_features=101, bias=True)
        )
      )
    )
  )
)

trainable params: 520,037 || all params: 86,396,362 || trainable%: 0.6019200206601292

{'eval_loss': 0.2534804046154022, 'eval_accuracy': 0.942, 'eval_runtime': 8.3425, 'eval_samples_per_second': 59.934, 'eval_steps_per_second': 0.24, 'epoch': 4.44}
{'train_runtime': 174.96, 'train_samples_per_second': 128.601, 'train_steps_per_second': 0.114, 'train_loss': 1.3298248767852783, 'epoch': 4.44}
"""
