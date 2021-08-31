#!/bin/bash

data_dir=/path/to/data
exp_dir=/path/to/experiments
emb_dir=/path/to/embeddings

TARGETS=( swl-sslc sa-ufal kpv-lattice ta-ttb gl-treegal yue-hk ckt-hse fo-oft te-mtg myv-jr qhe-hiencs qtd-sagt )
GENRES=( spoken fiction fiction news news spoken spoken wiki grammar fiction social spoken )

#
# LDA
#

# cluster treebanks in UD
python cluster/lda.py $data_dir/treebanks/ $exp_dir/lda/rs41/ -cl treebank -vu char -vn 3-6 -rs 41
python cluster/lda.py $data_dir/treebanks/ $exp_dir/lda/rs42/ -cl treebank -vu char -vn 3-6 -rs 42
python cluster/lda.py $data_dir/treebanks/ $exp_dir/lda/rs43/ -cl treebank -vu char -vn 3-6 -rs 43

# create corpora from cluster selection
for idx in "${!TARGETS[@]}"; do
	tb="${TARGETS[$idx]}";
	lang="${tb%-*}";
	genre="${GENRES[$idx]}";
	python cluster/selection.py $data_dir/treebanks/ $exp_dir/emb/mbert/ $exp_dir/lda/rs41/ $data_dir/select/lda-$tb/rs41/ $data_dir/select/$tb/filtered.pkl -ts 100 -us $data_dir/select/$genre-not-$lang/filtered.pkl -cl treebank -rs 41;
	python data/cat.py $data_dir/treebanks/ $data_dir/select/lda-$tb/rs41/selection.pkl $data_dir/select/lda-$tb/rs41/ -nc -nm;
	python cluster/selection.py $data_dir/treebanks/ $exp_dir/emb/mbert/ $exp_dir/lda/rs42/ $data_dir/select/lda-$tb/rs42/ $data_dir/select/$tb/filtered.pkl -ts 100 -us $data_dir/select/$genre-not-$lang/filtered.pkl -cl treebank -rs 42;
	python data/cat.py $data_dir/treebanks/ $data_dir/select/lda-$tb/rs42/selection.pkl $data_dir/select/lda-$tb/rs42/ -nc -nm;
	python cluster/selection.py $data_dir/treebanks/ $exp_dir/emb/mbert/ $exp_dir/lda/rs43/ $data_dir/select/lda-$tb/rs43/ $data_dir/select/$tb/filtered.pkl -ts 100 -us $data_dir/select/$genre-not-$lang/filtered.pkl -cl treebank -rs 43;
	python data/cat.py $data_dir/treebanks/ $data_dir/select/lda-$tb/rs43/selection.pkl $data_dir/select/lda-$tb/rs43/ -nc -nm;
done

#
# GMM
#

# cluster treebanks in UD
python cluster/gmm.py $data_dir/treebanks/ $emb_dir $exp_dir/gmm/rs41/ -cl treebank -rs 41
python cluster/gmm.py $data_dir/treebanks/ $emb_dir $exp_dir/gmm/rs42/ -cl treebank -rs 42
python cluster/gmm.py $data_dir/treebanks/ $emb_dir $exp_dir/gmm/rs43/ -cl treebank -rs 43

# create corpora from cluster selection
for idx in "${!TARGETS[@]}"; do
	tb="${TARGETS[$idx]}";
	lang="${tb%-*}";
	genre="${GENRES[$idx]}";
	python cluster/selection.py $data_dir/treebanks/ $exp_dir/emb/mbert/ $exp_dir/gmm/rs41/ $data_dir/select/gmm-$tb/rs41/ $data_dir/select/$tb/filtered.pkl -ts 100 -us $data_dir/select/$genre-not-$lang/filtered.pkl -cl treebank -rs 41;
	python data/cat.py $data_dir/treebanks/ $data_dir/select/gmm-$tb/rs41/selection.pkl $data_dir/select/gmm-$tb/rs41/ -nc -nm;
	python cluster/selection.py $data_dir/treebanks/ $exp_dir/emb/mbert/ $exp_dir/gmm/rs42/ $data_dir/select/gmm-$tb/rs42/ $data_dir/select/$tb/filtered.pkl -ts 100 -us $data_dir/select/$genre-not-$lang/filtered.pkl -cl treebank -rs 42;
	python data/cat.py $data_dir/treebanks/ $data_dir/select/gmm-$tb/rs42/selection.pkl $data_dir/select/gmm-$tb/rs42/ -nc -nm;
	python cluster/selection.py $data_dir/treebanks/ $exp_dir/emb/mbert/ $exp_dir/gmm/rs43/ $data_dir/select/gmm-$tb/rs43/ $data_dir/select/$tb/filtered.pkl -ts 100 -us $data_dir/select/$genre-not-$lang/filtered.pkl -cl treebank -rs 43;
	python data/cat.py $data_dir/treebanks/ $data_dir/select/gmm-$tb/rs43/selection.pkl $data_dir/select/gmm-$tb/rs43/ -nc -nm;
done
