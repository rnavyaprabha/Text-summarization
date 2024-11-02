# Exploration of Text Summarization Methods and Techniques
Researching and Building a text summarization AI models 

To create the environment using conda:
```
conda env create -f environment.yml
```

# spACy, HeapQ, TextRank
To use the extractive methods mentioned in the paper and in our presentation, use the following command, methods are `textrank` and `heapq`. If you haven't downloaded the spACy `en_core_web_lg` package, use `True`.
```
python -m spacy_fulldataset --method=[METHOD] --download=[True|False]
```

# BART 
In order to generate a baseline result for BART, the shell script `text_summarization_bart.sh` should be run, but in order to do so, the path should be changed to a local installation of `transformers` from HuggingFace, installed from source. The file `bart_baseline.out` is a text file with the shell script output from a `nohup` command.

## Baseline Results:
```
eval_gen_len            =    83.8787
  eval_loss               =     2.7142
  eval_rouge1             =     44.586
  eval_rouge2             =    21.4773
  eval_rougeL             =    31.0223
  eval_rougeLsum          =    41.5275
  eval_runtime            = 1:45:18.98
  eval_samples            =      13368
  eval_samples_per_second =      2.116
  eval_steps_per_second   =      0.529
```

The relevant metrics are `eval_rouge_1`, `eval_rouge_2`, and `eval_rouge_L`. The values are in line with the original paper from Facebook AI (now META AI).
