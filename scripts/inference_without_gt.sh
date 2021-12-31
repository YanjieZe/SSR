GROUP='mxfold2_turner_lstm_hinge_mix'
GROUP='mxfold2_turner_lstm_hinge'
GROUP='mxfold2_turner_transformer_hinge_mix'
# GROUP='mxfold2_turner_transformer_hinge'

TRANSFORMER=0
TRANSFORMER=1


LOSS='hinge_mix'
LOSS='hinge'

MAXLEN=600
# set="archiveII_cleaned.lst"
set="test_seq.lst"
# set="archiveII_pseudoknot.lst"

cd novafold
echo "testing ${GROUP}"

alg="linearfold"
# alg="mxfold2"

python inference_without_gt.py --model ${alg} \
                    --epoch 10 \
                    --batch-size 1 \
                    --exp $GROUP \
                    --num-transformer-layers $TRANSFORMER \
                    --num-transformer-att 4 \
                    --device cpu \
                    --train-set ${set} \
                    --seq-max-len $MAXLEN 