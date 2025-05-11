python distillation.py --teacher t5-small --data_dir cnn_dm \
    --student_decoder_layers 3 --student_encoder_layers 6 --tokenizer_name t5-small \
    --learning_rate=3e-4 --freeze_encoder --no_teacher --freeze_embeds \
    --do_train --train_batch_size 32 \
    --do_predict \
    --model_name_or_path t5-small --eval_beams 2 --eval_max_gen_length 142 \
    --val_check_interval 0.25 --n_val 1000 \
    --output_dir distilt5 --gpus 1 --logger_name wandb