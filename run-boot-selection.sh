#!/bin/bash

data_dir=/path/to/data
exp_dir=/path/to/experiments

TARGETS=( swl-sslc sa-ufal kpv-lattice ta-ttb gl-treegal yue-hk ckt-hse fo-oft te-mtg myv-jr qhe-hiencs qtd-sagt )
GENRES=( spoken fiction fiction news news spoken spoken wiki grammar fiction social spoken )

# train bootstrapping classifier
python classify/boot.py $data_dir/treebanks bert-base-multilingual-cased $exp_dir/boot -s $data_dir/split/40k/split.pkl

# predict instance-level genre
python classify/predict.py $data_dir/treebanks bert-base-multilingual-cased $exp_dir/boot -cd

# select in-genre instances according to predicted genre distribution
for idx in "${!TARGETS[@]}"; do
	tb="${TARGETS[$idx]}";
	lang="${tb%-*}";
	genre="${GENRES[$idx]}";
	python data/filter.py $data_dir/treebanks/ $data_dir/select/not-$lang/filtered.pkl $data_dir/select/boot-$tb/ -dd $exp_dir/boot/All.pkl -xd $genre;
	python data/cat.py $data_dir/treebanks/ $data_dir/select/boot-$tb/filtered.pkl $data_dir/select/boot-$tb/ -nc -nm;
done
