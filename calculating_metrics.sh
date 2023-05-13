#!/bin/bash

# Define preprocessing flag values
DATA_FLAG='covid_data'
MODEL_FLAG='gplvm'
PREPROC_FLAG='rawcounts' #('libnormlogtrans' 'logtranscolumnstd' 'libnormlogtranscolumnstd')
ENCODER_FLAG=('scalynocovars' 'scaly')
KERNEL_FLAG='linear_linear'
LIKELIHOOD_FLAG=('nblikelihoodlearnscalelearntheta' 'nblikelihoodlearnscalefixedtheta1')
SEED_FLAG='42'

# Activate conda environment
source activate gplvm

# Run commands
# for flag in "${preprocessing_flags[@]}"
# do
#     python calc_metrics.py --preprocessing "${flag}" --encoder point --likelihood gaussianlikelihood > "./models/covid_data/seed42/gplvm_${flag}_point_gaussianlikelihood_output.txt"
#     python calc_metrics.py --preprocessing "${flag}" --encoder nnenc --likelihood gaussianlikelihood --bio_metrics nmi ari iso_labels_f1 cellASW iso_labels_asw --batch_metrics batchASW graph_connectivity > "./models/covid_data/seed42/gplvm_${flag}_nnenc_gaussianlikelihood_nolisi_output.txt"
#     python calc_metrics.py --preprocessing "${flag}" --encoder scaly --likelihood gaussianlikelihood > "./models/covid_data/seed42/gplvm_${flag}_scaly_gaussianlikelihood_output.txt"
#     python calc_metrics.py --preprocessing "${flag}" --encoder scalynocovars --likelihood gaussianlikelihood > "./models/covid_data/seed42/gplvm_${flag}_scalynocovars_gaussianlikelihood_output.txt"
# done

constant_flags="-m ${MODEL_FLAG} -d ${DATA_FLAG} -k ${KERNEL_FLAG} -s ${SEED_FLAG}"

for l_flag in "${LIKELIHOOD_FLAG[@]}"
do
    for e_flag in "${ENCODER_FLAG[@]}"
    do
        CMD="python calc_metrics.py ${constant_flags} -e ${e_flag} -l ${l_flag} > ./models/${DATA_FLAG}/seed${SEED_FLAG}/${MODEL_FLAG}_${PREPROC_FLAG}_${e_flag}_${KERNEL_FLAG}_${l_flag}_output.txt"
        echo -e "\nExecuting command:\n==================\n${CMD}\n"
        eval $CMD
    done
done

# Deactivate conda environment
conda deactivate