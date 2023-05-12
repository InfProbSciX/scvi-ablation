#!/bin/bash

# Define preprocessing flag values
preprocessing_flags=('libnormlogtrans' 'logtranscolumnstd' 'libnormlogtranscolumnstd')
nblikelihood_flags=('nblikelihoodnoscalelearntheta' 'nblikelihoodnoscalefixedtheta1') #'nblikelihoodlearnscalelearntheta' 'nblikelihoodlearnscalefixedtheta1'

# Activate conda environment
source activate gplvm

# Run commands
for flag in "${preprocessing_flags[@]}"
do
    python calc_metrics.py --preprocessing "${flag}" --encoder point --likelihood gaussianlikelihood > "./models/covid_data/seed42/gplvm_${flag}_point_gaussianlikelihood_output.txt"
    python calc_metrics.py --preprocessing "${flag}" --encoder nnenc --likelihood gaussianlikelihood --bio_metrics nmi ari iso_labels_f1 cellASW iso_labels_asw --batch_metrics batchASW graph_connectivity > "./models/covid_data/seed42/gplvm_${flag}_nnenc_gaussianlikelihood_nolisi_output.txt"
    python calc_metrics.py --preprocessing "${flag}" --encoder scaly --likelihood gaussianlikelihood > "./models/covid_data/seed42/gplvm_${flag}_scaly_gaussianlikelihood_output.txt"
    python calc_metrics.py --preprocessing "${flag}" --encoder scalynocovars --likelihood gaussianlikelihood > "./models/covid_data/seed42/gplvm_${flag}_scalynocovars_gaussianlikelihood_output.txt"
done

for flag in "${nblikelihood_flags[@]}"
do
    python calc_metrics.py --preprocessing rawcounts --encoder point --likelihood "${flag}" > "./models/covid_data/seed42/gplvm_rawcounts_point_${flag}_output.txt"
    python calc_metrics.py --preprocessing rawcounts --encoder scaly --likelihood "${flag}" > "./models/covid_data/seed42/gplvm_rawcounts_scaly_${flag}_output.txt"
    python calc_metrics.py --preprocessing rawcounts --encoder scalynocovars --likelihood "${flag}" > "./models/covid_data/seed42/gplvm_rawcounts_scalynocovars_${flag}_output.txt"
done

# Deactivate conda environment
conda deactivate