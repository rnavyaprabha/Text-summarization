# -*- coding: utf-8 -*-

import datasets
import transformers
from datasets import load_dataset
import evaluate

from transformers import BertTokenizerFast
import torch
torch.backends.cuda.matmul.allow_tf32 = True


tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
tokenizer.bos_token = tokenizer.cls_token
tokenizer.eos_token = tokenizer.sep_token

train_data = load_dataset("cnn_dailymail", "3.0.0", split="train")
val_data = load_dataset("cnn_dailymail", "3.0.0", split="validation[:10%]")

batch_size=4  # change to 16 for full training
encoder_max_length=512
decoder_max_length=128

def process_data_to_model_inputs(batch):
  # tokenize the inputs and labels
  inputs = tokenizer(batch["article"], padding="max_length", truncation=True, max_length=encoder_max_length)
  outputs = tokenizer(batch["highlights"], padding="max_length", truncation=True, max_length=decoder_max_length)

  batch["input_ids"] = inputs.input_ids
  batch["attention_mask"] = inputs.attention_mask
  batch["decoder_input_ids"] = outputs.input_ids
  batch["decoder_attention_mask"] = outputs.attention_mask
  batch["labels"] = outputs.input_ids.copy()

  # because BERT automatically shifts the labels, the labels correspond exactly to `decoder_input_ids`. 
  # We have to make sure that the PAD token is ignored
  batch["labels"] = [[-100 if token == tokenizer.pad_token_id else token for token in labels] for labels in batch["labels"]]

  return batch

# only use 32 training examples for notebook - DELETE LINE FOR FULL TRAINING
#train_data = train_data.select(range(32))

train_data = train_data.map(
    process_data_to_model_inputs, 
    batched=True, 
    batch_size=batch_size, 
    remove_columns=["article", "highlights", "id"],
    num_proc=64
)
train_data.set_format(
    type="torch", columns=["input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask", "labels"],
)


# only use 16 training examples for notebook - DELETE LINE FOR FULL TRAINING
#val_data = val_data.select(range(16))

val_data = val_data.map(
    process_data_to_model_inputs, 
    batched=True, 
    batch_size=batch_size, 
    remove_columns=["article", "highlights", "id"],
    num_proc=64
)
val_data.set_format(
    type="torch", columns=["input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask", "labels"],
)

"""### **Warm-starting the Encoder-Decoder Model**"""

from transformers import EncoderDecoderModel

bert2bert = EncoderDecoderModel.from_encoder_decoder_pretrained("bert-base-uncased", "bert-base-uncased")

# set special tokens
bert2bert.config.decoder_start_token_id = tokenizer.bos_token_id
bert2bert.config.eos_token_id = tokenizer.eos_token_id
bert2bert.config.pad_token_id = tokenizer.pad_token_id

# sensible parameters for beam search
bert2bert.config.vocab_size = bert2bert.config.decoder.vocab_size
bert2bert.config.max_length = 142
bert2bert.config.min_length = 56
bert2bert.config.no_repeat_ngram_size = 3
bert2bert.config.early_stopping = True
bert2bert.config.length_penalty = 2.0
bert2bert.config.num_beams = 4

"""### **Fine-Tuning Warm-Started Encoder-Decoder Models**

For the `EncoderDecoderModel` framework, we will use the `Seq2SeqTrainingArguments` and the `Seq2SeqTrainer`. Let's import them.
"""

from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer

"""Also, we need to define a function to correctly compute the ROUGE score during validation. ROUGE is a much better metric to track during training than only language modeling loss."""

# load rouge for validation
rouge = evaluate.load("rouge")

def compute_metrics(pred):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    # all unnecessary tokens are removed
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = tokenizer.pad_token_id
    label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

    rouge_output = rouge.compute(predictions=pred_str, references=label_str)

    return rouge_output

"""Cool! Finally, we start training."""

# set training arguments - these params are not really tuned, feel free to change
training_args = Seq2SeqTrainingArguments(
    output_dir="./",
    evaluation_strategy="steps",
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    predict_with_generate=True,
    logging_steps=1000,  # set to 1000 for full training
    save_steps=500,  # set to 500 for full training
    eval_steps=8000,  # set to 8000 for full training
    warmup_steps=2000,  # set to 2000 for full training
    #max_steps=16, # delete for full training
    overwrite_output_dir=True,
    save_total_limit=3,
    bf16=True,
    gradient_accumulation_steps=16,
    gradient_checkpointing=True,
    tf32=True,
    optim="adafactor"
)

# instantiate trainer
trainer = Seq2SeqTrainer(
    model=bert2bert,
    tokenizer=tokenizer,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_data,
    eval_dataset=val_data,
)
trainer.train()

"""### **Evaluation**

Awesome, we finished training our dummy model. Let's now evaluated the model on the test data. We make use of the dataset's handy `.map()` function to generate a summary of each sample of the test data.
"""

import datasets
from transformers import BertTokenizer, EncoderDecoderModel

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
#model = EncoderDecoderModel.from_pretrained("./checkpoint-16")
#model.to("cuda")

test_data = datasets.load_dataset("cnn_dailymail", "3.0.0", split="test")

# only use 16 training examples for notebook - DELETE LINE FOR FULL TRAINING
#test_data = test_data.select(range(16))

batch_size = 16  # change to 64 for full evaluation

# map data correctly
def generate_summary(batch):
    # Tokenizer will automatically set [BOS] <text> [EOS]
    # cut off at BERT max length 512
    inputs = tokenizer(batch["article"], padding="max_length", truncation=True, max_length=512, return_tensors="pt")
    input_ids = inputs.input_ids.to("cuda")
    attention_mask = inputs.attention_mask.to("cuda")

    outputs = bert2bert.generate(input_ids, attention_mask=attention_mask)

    # all special tokens including will be removed
    output_str = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    batch["pred"] = output_str

    return batch

results = test_data.map(generate_summary, batched=True, batch_size=batch_size, remove_columns=["article"])

pred_str = results["pred"]
label_str = results["highlights"]

rouge_output = rouge.compute(predictions=pred_str, references=label_str)

print(rouge_output)

"""Reference [patrickvonplaten/bert2bert_cnn_daily_mail](https://huggingface.co/patrickvonplaten/bert2bert_cnn_daily_mail). 

The model achieves a ROUGE-2 score of **18.22**, which is even a little better than reported in the paper.

Reference 2 , [here](https://huggingface.co/patrickvonplaten/bert2bert_cnn_daily_mail).

"""

