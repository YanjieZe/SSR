GROUP='mxfold2_turner_lstm_hinge_mix'
GROUP='mxfold2_turner_lstm_hinge'
GROUP='mxfold2_turner_transformer_hinge_mix'
GROUP='mxfold2_turner_transformer_hinge'

TRANSFORMER=0
TRANSFORMER=1

# lstm hinge mix
# all: Avg. PPV 0.6569927423523803 | Avg. SEN 0.7021672988131752 | Avg. F1 0.6764
# pseudoknot: Avg. PPV 0.5037490263209529 | Avg. SEN 0.5447181112756043 | Avg. F1 0.520741
# free: Avg. PPV 0.7145082344549838 | Avg. SEN 0.7612611901445612 | Avg. F1 0.734899

# lstm hinge
# all: Avg. PPV 0.7256636973221392 | Avg. SEN 0.7445384335932322 | Avg. F1 0.7328
# pseudoknot:  Avg. PPV 0.48887381405985797 | Avg. SEN 0.4996385000040139 | Avg. F1 0.49173
# free: Avg. PPV 0.814535769948959 | Avg. SEN 0.8364543735159533 | Avg. F1 0.8233047

# tf hinge mix
# all: Avg. PPV 0.6569927423523808 | Avg. SEN 0.7021672988131762 | Avg. F1 0.6764
# pseudoknot: Avg. PPV 0.5037490263209531 | Avg. SEN 0.5447181112756042 | Avg. F1 0.520741
# free: Avg. PPV 0.7145082344549829 | Avg. SEN 0.7612611901445611 | Avg. F1 0.734899

# tf hinge
# all: Avg. PPV 0.7256636973221405 | Avg. SEN 0.7445384335932315 | Avg. F1 0.7328
# pseudoknot: Avg. PPV 0.48887381405985797 | Avg. SEN 0.4996385000040139 | Avg. F1 0.49173
# free: Avg. PPV 0.8145357699489602 | Avg. SEN 0.8364543735159531 | Avg. F1 0.823304


LOSS='hinge_mix'
LOSS='hinge'

MAXLEN=600
# set="archiveII_cleaned.lst"
set="archiveII_valid.lst"
# set="archiveII_pseudoknot.lst"

cd novafold
echo "testing ${GROUP}"

python benchmark_archii.py --model mxfold2 \
                    --epoch 10 \
                    --batch-size 1 \
                    --exp $GROUP \
                    --num-transformer-layers $TRANSFORMER \
                    --num-transformer-att 4 \
                    --device cpu \
                    --train-set ${set} \
                    --seq-max-len $MAXLEN 