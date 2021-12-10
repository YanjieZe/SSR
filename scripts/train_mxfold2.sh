SEED=0
TYPE="MixC"
TRANSFORMER=0 # 0 for lstm, 1 for transformer

# model type for mxfold2 can be:
# Turner, Zuker, ZukerC, ZukerL, ZukerS, 
# Mix, MixC

cd novafold

python train.py --model mxfold2 \
                --model-type $TYPE \
                --seed $SEED \
                --device cpu \
                --exp hello_mxfold2 \
                --batch-size 1 \
                --num-transformer-layers $TRANSFORMER \
                --num-transformer-att 4 \
                --wandb \
                --wandb_project RNA_AI \
                --wandb_name $SEED \
                --wandb_group mxfold2_mixc_lstm