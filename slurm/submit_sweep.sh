#!/bin/bash

set -euo pipefail

BITWIDTHS=(4 8)
ACTBITS=(4 8)
SCHEMES=(2)
RATIO=(0.5 0.6 0.7 0.8 0.9)

for bw in "${BITWIDTHS[@]}"; do
  for scheme in "${SCHEMES[@]}"; do
    for abs in "${ACTBITS[@]}"; do
        for ratio in "${RATIO[@]}"; do
            echo "Submitting job: BITWIDTH=$bw SCHEME=$scheme ACTBITS=${abs} RATIO = ${ratio}"
            sbatch \
                --job-name=gen1_evrt-detr-presnet50-lrd-s${scheme}-r${ratio}-w${bw}-a${abs}-eval-alpha095 \
                --output=evrt_detr_presnet50_lrd_s${scheme}_r${ratio}_w${bw}_a${abs}_alpha095.out \
                --error=evrt_detr_presnet50_lrd_s${scheme}_r${ratio}_w${bw}_a${abs}_alpha095.err \
                --export=ALL,BITWIDTH=$bw,SCHEME=$scheme,ACTBITS=${abs},RATIO=${ratio} \
                eval.sbatch
        done
    done
  done
done


