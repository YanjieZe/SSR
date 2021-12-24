
GROUP='mxfold2_turner_lstm_hinge'
GROUP='mxfold2_turner_lstm_hinge_mix'
GROUP='mxfold2_turner_transformer_hinge'
GROUP='mxfold2_turner_transformer_hinge_mix'

TRANSFORMER=1
LOSS='hinge_mix'
set="finaltest_valid_all.lst"

cd novafold
echo "testing ${GROUP}"

# mxfold2_turner_lstm_hinge:
# epoch=10 
# Avg. PPV 0.44295029084060605 | Avg. SEN 0.7591113775384143 | Avg. F1 0.52558

# mxfold2_turner_lstm_hinge_mix:
# epoch=10
# Avg. PPV 0.43482840420124774 | Avg. SEN 0.8266984273627779 | Avg. F1 0.53609

# mxfold2_turner_transformer_hinge: 
# epoch=10
# Avg. PPV 0.4429502908406062 | Avg. SEN 0.759111377538414 | Avg. F1 0.5255833

# mxfold2_turner_transformer_hinge_mix:
# epoch=10
# Avg. PPV 0.4348284042012478 | Avg. SEN 0.8266984273627777 | Avg. F1 0.536099




python benchmark_archii.py --model mxfold2 \
                    --epoch 10 \
                    --batch-size 1 \
                    --exp $GROUP \
                    --num-transformer-layers $TRANSFORMER \
                    --num-transformer-att 4 \
                    --device cpu \
                    --loss-func $LOSS \
                    --train-set ${set}