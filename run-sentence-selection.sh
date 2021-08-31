#!/bin/bash

data_dir=/path/to/data
exp_dir=/path/to/experiments
emb_dir=/path/to/embeddings

# SWL-SSLC (Sen)
python cluster/selection.py $data_dir/treebanks/ $emb_dir None $data_dir/select/sen-swl-sslc/rs42/ $data_dir/select/swl-sslc/filtered.pkl -ts 100 -us $data_dir/select/not-swl/filtered.pkl -cl sentence -tk 35961 -rs 42
python data/cat.py $data_dir/treebanks/ $data_dir/select/sen-swl-sslc/rs42/selection.pkl $data_dir/select/sen-swl-sslc/rs42/ -nc -nm
python cluster/selection.py $data_dir/treebanks/ $emb_dir None $data_dir/select/sen-swl-sslc/rs41/ $data_dir/select/swl-sslc/filtered.pkl -ts 100 -us $data_dir/select/not-swl/filtered.pkl -cl sentence -tk 39542 -rs 41
python data/cat.py $data_dir/treebanks/ $data_dir/select/sen-swl-sslc/rs41/selection.pkl $data_dir/select/sen-swl-sslc/rs41/ -nc -nm
python cluster/selection.py $data_dir/treebanks/ $emb_dir None $data_dir/select/sen-swl-sslc/rs43/ $data_dir/select/swl-sslc/filtered.pkl -ts 100 -us $data_dir/select/not-swl/filtered.pkl -cl sentence -tk 37699 -rs 43
python data/cat.py $data_dir/treebanks/ $data_dir/select/sen-swl-sslc/rs43/selection.pkl $data_dir/select/sen-swl-sslc/rs43/ -nc -nm

# SA-UFAL (Sen)
python cluster/selection.py $data_dir/treebanks/ $emb_dir None $data_dir/select/sen-sa-ufal/rs42/ $data_dir/select/sa-ufal/filtered.pkl -ts 100 -us $data_dir/select/not-sa/filtered.pkl -cl sentence -tk 118766 -rs 42
python data/cat.py $data_dir/treebanks/ $data_dir/select/sen-sa-ufal/rs42/selection.pkl $data_dir/select/sen-sa-ufal/rs42/ -nc -nm
python cluster/selection.py $data_dir/treebanks/ $emb_dir None $data_dir/select/sen-sa-ufal/rs41/ $data_dir/select/sa-ufal/filtered.pkl -ts 100 -us $data_dir/select/not-sa/filtered.pkl -cl sentence -tk 120794 -rs 41
python data/cat.py $data_dir/treebanks/ $data_dir/select/sen-sa-ufal/rs41/selection.pkl $data_dir/select/sen-sa-ufal/rs41/ -nc -nm
python cluster/selection.py $data_dir/treebanks/ $emb_dir None $data_dir/select/sen-sa-ufal/rs43/ $data_dir/select/sa-ufal/filtered.pkl -ts 100 -us $data_dir/select/not-sa/filtered.pkl -cl sentence -tk 115084 -rs 43
python data/cat.py $data_dir/treebanks/ $data_dir/select/sen-sa-ufal/rs43/selection.pkl $data_dir/select/sen-sa-ufal/rs43/ -nc -nm

# KPV-Lattice (Sen)
python cluster/selection.py $data_dir/treebanks/ $emb_dir None $data_dir/select/sen-kpv-lattice/rs42/ $data_dir/select/kpv-lattice/filtered.pkl -ts 100 -us $data_dir/select/not-kpv/filtered.pkl -cl sentence -tk 141563 -rs 42
python data/cat.py $data_dir/treebanks/ $data_dir/select/sen-kpv-lattice/rs42/selection.pkl $data_dir/select/sen-kpv-lattice/rs42/ -nc -nm
python cluster/selection.py $data_dir/treebanks/ $emb_dir None $data_dir/select/sen-kpv-lattice/rs41/ $data_dir/select/kpv-lattice/filtered.pkl -ts 100 -us $data_dir/select/not-kpv/filtered.pkl -cl sentence -tk 119936 -rs 41
python data/cat.py $data_dir/treebanks/ $data_dir/select/sen-kpv-lattice/rs41/selection.pkl $data_dir/select/sen-kpv-lattice/rs41/ -nc -nm
python cluster/selection.py $data_dir/treebanks/ $emb_dir None $data_dir/select/sen-kpv-lattice/rs43/ $data_dir/select/kpv-lattice/filtered.pkl -ts 100 -us $data_dir/select/not-kpv/filtered.pkl -cl sentence -tk 118326 -rs 43
python data/cat.py $data_dir/treebanks/ $data_dir/select/sen-kpv-lattice/rs43/selection.pkl $data_dir/select/sen-kpv-lattice/rs43/ -nc -nm

# TA-TTB (Sen)
python cluster/selection.py $data_dir/treebanks/ $emb_dir None $data_dir/select/sen-ta-ttb/rs42/ $data_dir/select/ta-ttb/filtered.pkl -ts 100 -us $data_dir/select/not-ta/filtered.pkl -cl sentence -tk 350918 -rs 42
python data/cat.py $data_dir/treebanks/ $data_dir/select/sen-ta-ttb/rs42/selection.pkl $data_dir/select/sen-ta-ttb/rs42/ -nc -nm
python cluster/selection.py $data_dir/treebanks/ $emb_dir None $data_dir/select/sen-ta-ttb/rs41/ $data_dir/select/ta-ttb/filtered.pkl -ts 100 -us $data_dir/select/not-ta/filtered.pkl -cl sentence -tk 290252 -rs 41
python data/cat.py $data_dir/treebanks/ $data_dir/select/sen-ta-ttb/rs41/selection.pkl $data_dir/select/sen-ta-ttb/rs41/ -nc -nm
python cluster/selection.py $data_dir/treebanks/ $emb_dir None $data_dir/select/sen-ta-ttb/rs43/ $data_dir/select/ta-ttb/filtered.pkl -ts 100 -us $data_dir/select/not-ta/filtered.pkl -cl sentence -tk 301285 -rs 43
python data/cat.py $data_dir/treebanks/ $data_dir/select/sen-ta-ttb/rs43/selection.pkl $data_dir/select/sen-ta-ttb/rs43/ -nc -nm

# GL-TreeGal (Sen)
python cluster/selection.py $data_dir/treebanks/ $emb_dir None $data_dir/select/sen-gl-treegal/rs42/ $data_dir/select/gl-treegal/filtered.pkl -ts 100 -us $data_dir/select/not-gl/filtered.pkl -cl sentence -tk 282360 -rs 42
python data/cat.py $data_dir/treebanks/ $data_dir/select/sen-gl-treegal/rs42/selection.pkl $data_dir/select/sen-gl-treegal/rs42/ -nc -nm
python cluster/selection.py $data_dir/treebanks/ $emb_dir None $data_dir/select/sen-gl-treegal/rs41/ $data_dir/select/gl-treegal/filtered.pkl -ts 100 -us $data_dir/select/not-gl/filtered.pkl -cl sentence -tk 298449 -rs 41
python data/cat.py $data_dir/treebanks/ $data_dir/select/sen-gl-treegal/rs41/selection.pkl $data_dir/select/sen-gl-treegal/rs41/ -nc -nm
python cluster/selection.py $data_dir/treebanks/ $emb_dir None $data_dir/select/sen-gl-treegal/rs43/ $data_dir/select/gl-treegal/filtered.pkl -ts 100 -us $data_dir/select/not-gl/filtered.pkl -cl sentence -tk 302789 -rs 43
python data/cat.py $data_dir/treebanks/ $data_dir/select/sen-gl-treegal/rs43/selection.pkl $data_dir/select/sen-gl-treegal/rs43/ -nc -nm

# YUE-HK (Sen)
python cluster/selection.py $data_dir/treebanks/ $emb_dir None $data_dir/select/sen-yue-hk/rs42/ $data_dir/select/yue-hk/filtered.pkl -ts 100 -us $data_dir/select/not-yue/filtered.pkl -cl sentence -tk 41945 -rs 42
python data/cat.py $data_dir/treebanks/ $data_dir/select/sen-yue-hk/rs42/selection.pkl $data_dir/select/sen-yue-hk/rs42/ -nc -nm
python cluster/selection.py $data_dir/treebanks/ $emb_dir None $data_dir/select/sen-yue-hk/rs42/ $data_dir/select/yue-hk/filtered.pkl -ts 100 -us $data_dir/select/not-yue/filtered.pkl -cl sentence -tk 36156 -rs 41
python data/cat.py $data_dir/treebanks/ $data_dir/select/sen-yue-hk/rs41/selection.pkl $data_dir/select/sen-yue-hk/rs41/ -nc -nm
python cluster/selection.py $data_dir/treebanks/ $emb_dir None $data_dir/select/sen-yue-hk/rs43/ $data_dir/select/yue-hk/filtered.pkl -ts 100 -us $data_dir/select/not-yue/filtered.pkl -cl sentence -tk 37067 -rs 43
python data/cat.py $data_dir/treebanks/ $data_dir/select/sen-yue-hk/rs43/selection.pkl $data_dir/select/sen-yue-hk/rs43/ -nc -nm

# CKT-HSE (Sen)
python cluster/selection.py $data_dir/treebanks/ $emb_dir None $data_dir/select/sen-ckt-hse/rs42/ $data_dir/select/ckt-hse/filtered.pkl -ts 100 -us $data_dir/select/not-ckt/filtered.pkl -cl sentence -tk 39441 -rs 42
python data/cat.py $data_dir/treebanks/ $data_dir/select/sen-ckt-hse/rs42/selection.pkl $data_dir/select/sen-ckt-hse/rs42/ -nc -nm
python cluster/selection.py $data_dir/treebanks/ $emb_dir None $data_dir/select/sen-ckt-hse/rs41/ $data_dir/select/ckt-hse/filtered.pkl -ts 100 -us $data_dir/select/not-ckt/filtered.pkl -cl sentence -tk 36166 -rs 41
python data/cat.py $data_dir/treebanks/ $data_dir/select/sen-ckt-hse/rs41/selection.pkl $data_dir/select/sen-ckt-hse/rs41/ -nc -nm
python cluster/selection.py $data_dir/treebanks/ $emb_dir None $data_dir/select/sen-ckt-hse/rs43/ $data_dir/select/ckt-hse/filtered.pkl -ts 100 -us $data_dir/select/not-ckt/filtered.pkl -cl sentence -tk 36212 -rs 43
python data/cat.py $data_dir/treebanks/ $data_dir/select/sen-ckt-hse/rs43/selection.pkl $data_dir/select/sen-ckt-hse/rs43/ -nc -nm

# FO-OFT (Sen)
python cluster/selection.py $data_dir/treebanks/ $emb_dir None $data_dir/select/sen-fo-oft/rs42/ $data_dir/select/fo-oft/filtered.pkl -ts 100 -us $data_dir/select/not-fo/filtered.pkl -cl sentence -tk 71166 -rs 42
python data/cat.py $data_dir/treebanks/ $data_dir/select/sen-fo-oft/rs42/selection.pkl $data_dir/select/sen-fo-oft/rs42/ -nc -nm
python cluster/selection.py $data_dir/treebanks/ $emb_dir None $data_dir/select/sen-fo-oft/rs41/ $data_dir/select/fo-oft/filtered.pkl -ts 100 -us $data_dir/select/not-fo/filtered.pkl -cl sentence -tk 71996 -rs 41
python data/cat.py $data_dir/treebanks/ $data_dir/select/sen-fo-oft/rs41/selection.pkl $data_dir/select/sen-fo-oft/rs41/ -nc -nm
python cluster/selection.py $data_dir/treebanks/ $emb_dir None $data_dir/select/sen-fo-oft/rs43/ $data_dir/select/fo-oft/filtered.pkl -ts 100 -us $data_dir/select/not-fo/filtered.pkl -cl sentence -tk 75001 -rs 43
python data/cat.py $data_dir/treebanks/ $data_dir/select/sen-fo-oft/rs43/selection.pkl $data_dir/select/sen-fo-oft/rs43/ -nc -nm

# TE-MTG (Sen)
python cluster/selection.py $data_dir/treebanks/ $emb_dir None $data_dir/select/sen-te-mtg/rs42/ $data_dir/select/te-mtg/filtered.pkl -ts 100 -us $data_dir/select/not-te/filtered.pkl -cl sentence -tk 27943 -rs 42
python data/cat.py $data_dir/treebanks/ $data_dir/select/sen-te-mtg/rs42/selection.pkl $data_dir/select/sen-te-mtg/rs42/ -nc -nm
python cluster/selection.py $data_dir/treebanks/ $emb_dir None $data_dir/select/sen-te-mtg/rs41/ $data_dir/select/te-mtg/filtered.pkl -ts 100 -us $data_dir/select/not-te/filtered.pkl -cl sentence -tk 28378 -rs 41
python data/cat.py $data_dir/treebanks/ $data_dir/select/sen-te-mtg/rs41/selection.pkl $data_dir/select/sen-te-mtg/rs41/ -nc -nm
python cluster/selection.py $data_dir/treebanks/ $emb_dir None $data_dir/select/sen-te-mtg/rs43/ $data_dir/select/te-mtg/filtered.pkl -ts 100 -us $data_dir/select/not-te/filtered.pkl -cl sentence -tk 28143 -rs 43
python data/cat.py $data_dir/treebanks/ $data_dir/select/sen-te-mtg/rs43/selection.pkl $data_dir/select/sen-te-mtg/rs43/ -nc -nm

# MYV-JR (Sen)
python cluster/selection.py $data_dir/treebanks/ $emb_dir None $data_dir/select/sen-myv-jr/rs42/ $data_dir/select/myv-jr/filtered.pkl -ts 100 -us $data_dir/select/not-myv/filtered.pkl -cl sentence -tk 136319 -rs 42
python data/cat.py $data_dir/treebanks/ $data_dir/select/sen-myv-jr/rs42/selection.pkl $data_dir/select/sen-myv-jr/rs42/ -nc -nm
python cluster/selection.py $data_dir/treebanks/ $emb_dir None $data_dir/select/sen-myv-jr/rs41/ $data_dir/select/myv-jr/filtered.pkl -ts 100 -us $data_dir/select/not-myv/filtered.pkl -cl sentence -tk 123133 -rs 41
python data/cat.py $data_dir/treebanks/ $data_dir/select/sen-myv-jr/rs41/selection.pkl $data_dir/select/sen-myv-jr/rs41/ -nc -nm
python cluster/selection.py $data_dir/treebanks/ $emb_dir None $data_dir/select/sen-myv-jr/rs43/ $data_dir/select/myv-jr/filtered.pkl -ts 100 -us $data_dir/select/not-myv/filtered.pkl -cl sentence -tk 132445 -rs 43
python data/cat.py $data_dir/treebanks/ $data_dir/select/sen-myv-jr/rs43/selection.pkl $data_dir/select/sen-myv-jr/rs43/ -nc -nm

# QHE-HIENCS (Sen)
python cluster/selection.py $data_dir/treebanks/ $emb_dir None $data_dir/select/sen-qhe-hiencs/rs42/ $data_dir/select/qhe-hiencs/filtered.pkl -ts 100 -us $data_dir/select/not-qhe/filtered.pkl -cl sentence -tk 22661 -rs 42
python data/cat.py $data_dir/treebanks/ $data_dir/select/sen-qhe-hiencs/rs42/selection.pkl $data_dir/select/sen-qhe-hiencs/rs42/ -nc -nm
python cluster/selection.py $data_dir/treebanks/ $emb_dir None $data_dir/select/sen-qhe-hiencs/rs41/ $data_dir/select/qhe-hiencs/filtered.pkl -ts 100 -us $data_dir/select/not-qhe/filtered.pkl -cl sentence -tk 14718 -rs 41
python data/cat.py $data_dir/treebanks/ $data_dir/select/sen-qhe-hiencs/rs41/selection.pkl $data_dir/select/sen-qhe-hiencs/rs41/ -nc -nm
python cluster/selection.py $data_dir/treebanks/ $emb_dir None $data_dir/select/sen-qhe-hiencs/rs43/ $data_dir/select/qhe-hiencs/filtered.pkl -ts 100 -us $data_dir/select/not-qhe/filtered.pkl -cl sentence -tk 15420 -rs 43
python data/cat.py $data_dir/treebanks/ $data_dir/select/sen-qhe-hiencs/rs43/selection.pkl $data_dir/select/sen-qhe-hiencs/rs43/ -nc -nm

# QTD-SAGT (Sen)
python cluster/selection.py $data_dir/treebanks/ $emb_dir None $data_dir/select/sen-qtd-sagt/rs42/ $data_dir/select/qtd-sagt/filtered.pkl -ts 100 -us $data_dir/select/not-qtd/filtered.pkl -cl sentence -tk 34039 -rs 42
python data/cat.py $data_dir/treebanks/ $data_dir/select/sen-qtd-sagt/rs42/selection.pkl $data_dir/select/sen-qtd-sagt/rs42/ -nc -nm
python cluster/selection.py $data_dir/treebanks/ $emb_dir None $data_dir/select/sen-qtd-sagt/rs41/ $data_dir/select/qtd-sagt/filtered.pkl -ts 100 -us $data_dir/select/not-qtd/filtered.pkl -cl sentence -tk 38930 -rs 41
python data/cat.py $data_dir/treebanks/ $data_dir/select/sen-qtd-sagt/rs41/selection.pkl $data_dir/select/sen-qtd-sagt/rs41/ -nc -nm
python cluster/selection.py $data_dir/treebanks/ $emb_dir None $data_dir/select/sen-qtd-sagt/rs43/ $data_dir/select/qtd-sagt/filtered.pkl -ts 100 -us $data_dir/select/not-qtd/filtered.pkl -cl sentence -tk 35309 -rs 43
python data/cat.py $data_dir/treebanks/ $data_dir/select/sen-qtd-sagt/rs43/selection.pkl $data_dir/select/sen-qtd-sagt/rs43/ -nc -nm