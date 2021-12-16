SEED=0
TYPE="Turner"
TRANSFORMER=1 # 0 for lstm, 1 for transformer
GROUP='mxfold2_turner_transformer_hinge_mix'
LOSS='hinge_mix' # or 'hinge_mix'
EXP=$GROUP

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
                --loss-func $LOSS \
                --exp $EXP \
                --epoch 20 
                --wandb \
                --wandb_project RNA_AI \
                --wandb_name $SEED \
                --wandb_group $GROUP
