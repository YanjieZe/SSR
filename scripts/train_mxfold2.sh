SEED=0
TYPE="MixC"

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
                --num-transformer-layers 1 \
                # --wandb \
                # --wandb_project RNA_AI \
                # --wandb_name $SEED \
                # --wandb_group debug