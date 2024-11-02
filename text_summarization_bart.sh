# generates a baseline for comparison for fine-tuning
# adapted from https://github.com/huggingface/transformers/tree/main/examples/pytorch/summarization
python /home/marktrovinger/Projects/transformers/examples/pytorch/summarization/run_summarization.py \
    --model_name_or_path facebook/bart-large-cnn \
    --do_eval \
    --dataset_name cnn_dailymail \
    --dataset_config "3.0.0" \
    --source_prefix "summarize: " \
    --output_dir /tmp/tst-summarization \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=4 \
    --overwrite_output_dir \
    --predict_with_generate