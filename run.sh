python code/main.py \
    --gpu "0" \
    --model_name "BiLSTM_DAS" \
    --data_name "filter/cail/total" \
    --batch_size 128 \
    --epoch 100 \
    --sup \
    --kw_name "kw_task_p100" \
    --attn_sup_lambda 0.15 \
    --pretrain_word_emb


python code/main.py \
    --gpu "1" \
    --model_name "BiLSTM_DAS" \
    --data_name "medical/filter/total" \
    --batch_size 128 \
    --epoch 100 \
    --sup \
    --attn_sup_lambda 0.15 \
    --kw_name "kw_task_p100" 
