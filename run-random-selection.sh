#!/bin/bash

data_dir=/path/to/data
exp_dir=/path/to/experiments

# SWL-SSLC
python data/filter.py $data_dir/treebanks/ $data_dir/select/not-swl/filtered.pkl $data_dir/select/rand-swl-sslc/rs42/ -r "32075,8032,0" -rs 42
python data/cat.py $data_dir/treebanks/ $data_dir/select/rand-swl-sslc/rs42/filtered.pkl $data_dir/select/rand-swl-sslc/rs42/ -nc -nm
python data/filter.py $data_dir/treebanks/ $data_dir/select/not-swl/filtered.pkl $data_dir/select/rand-swl-sslc/rs41/ -r "30736,7697,0" -rs 41
python data/cat.py $data_dir/treebanks/ $data_dir/select/rand-swl-sslc/rs41/filtered.pkl $data_dir/select/rand-swl-sslc/rs41/ -nc -nm
python data/filter.py $data_dir/treebanks/ $data_dir/select/not-swl/filtered.pkl $data_dir/select/rand-swl-sslc/rs43/ -r "30439,7623,0" -rs 43
python data/cat.py $data_dir/treebanks/ $data_dir/select/rand-swl-sslc/rs43/filtered.pkl $data_dir/select/rand-swl-sslc/rs43/ -nc -nm

# SA-UFAL
python data/filter.py $data_dir/treebanks/ $data_dir/select/not-sa/filtered.pkl $data_dir/select/rand-sa-ufal/rs42 -r "80452,18331,0" -rs 42
python data/cat.py $data_dir/treebanks/ $data_dir/select/rand-sa-ufal/rs42/filtered.pkl $data_dir/select/rand-sa-ufal/rs42/ -nc -nm
python data/filter.py $data_dir/treebanks/ $data_dir/select/not-sa/filtered.pkl $data_dir/select/rand-sa-ufal/rs41 -r "82081,18738,0" -rs 41
python data/cat.py $data_dir/treebanks/ $data_dir/select/rand-sa-ufal/rs41/filtered.pkl $data_dir/select/rand-sa-ufal/rs41/ -nc -nm
python data/filter.py $data_dir/treebanks/ $data_dir/select/not-sa/filtered.pkl $data_dir/select/rand-sa-ufal/rs43 -r "79712,18145,0" -rs 43
python data/cat.py $data_dir/treebanks/ $data_dir/select/rand-sa-ufal/rs43/filtered.pkl $data_dir/select/rand-sa-ufal/rs43/ -nc -nm

python data/filter.py $data_dir/treebanks/ $data_dir/select/not-kpv/filtered.pkl $data_dir/select/rand-kpv-lattice/rs42 -r "88413,20326,0" -rs 42 
python data/cat.py $data_dir/treebanks/ $data_dir/select/rand-kpv-lattice/rs42/filtered.pkl $data_dir/select/rand-kpv-lattice/rs42/ -nc -nm
python data/filter.py $data_dir/treebanks/ $data_dir/select/not-kpv/filtered.pkl $data_dir/select/rand-kpv-lattice/rs41 -r "82899,18947,0" -rs 41
python data/cat.py $data_dir/treebanks/ $data_dir/select/rand-kpv-lattice/rs41/filtered.pkl $data_dir/select/rand-kpv-lattice/rs41/ -nc -nm
python data/filter.py $data_dir/treebanks/ $data_dir/select/not-kpv/filtered.pkl $data_dir/select/rand-kpv-lattice/rs43 -r "80413,18326,0" -rs 43
python data/cat.py $data_dir/treebanks/ $data_dir/select/rand-kpv-lattice/rs43/filtered.pkl $data_dir/select/rand-kpv-lattice/rs43/ -nc -nm

# TA-TTB
python data/filter.py $data_dir/treebanks/ $data_dir/select/not-ta/filtered.pkl $data_dir/select/rand-ta-ttb/rs42 -r "260142,55436,0" -rs 42
python data/cat.py $data_dir/treebanks/ $data_dir/select/rand-ta-ttb/rs42/filtered.pkl $data_dir/select/rand-ta-ttb/rs42/ -nc -nm
python data/filter.py $data_dir/treebanks/ $data_dir/select/not-ta/filtered.pkl $data_dir/select/rand-ta-ttb/rs41 -r "242089,50923,0" -rs 41
python data/cat.py $data_dir/treebanks/ $data_dir/select/rand-ta-ttb/rs41/filtered.pkl $data_dir/select/rand-ta-ttb/rs41/ -nc -nm
python data/filter.py $data_dir/treebanks/ $data_dir/select/not-ta/filtered.pkl $data_dir/select/rand-ta-ttb/rs43 -r "244863,51616,0" -rs 43
python data/cat.py $data_dir/treebanks/ $data_dir/select/rand-ta-ttb/rs43/filtered.pkl $data_dir/select/rand-ta-ttb/rs43/ -nc -nm

# GL-TreeGal
python data/filter.py $data_dir/treebanks/ $data_dir/select/not-gl/filtered.pkl $data_dir/select/rand-gl-treegal/rs42 -r "239403,50152,0" -rs 42
python data/cat.py $data_dir/treebanks/ $data_dir/select/rand-gl-treegal/rs42/filtered.pkl $data_dir/select/rand-gl-treegal/rs42/ -nc -nm
python data/filter.py $data_dir/treebanks/ $data_dir/select/not-gl/filtered.pkl $data_dir/select/rand-gl-treegal/rs41 -r "241625,50708,0" -rs 41
python data/cat.py $data_dir/treebanks/ $data_dir/select/rand-gl-treegal/rs41/filtered.pkl $data_dir/select/rand-gl-treegal/rs41/ -nc -nm
python data/filter.py $data_dir/treebanks/ $data_dir/select/not-gl/filtered.pkl $data_dir/select/rand-gl-treegal/rs43 -r "250049,50709,0" -rs 43
python data/cat.py $data_dir/treebanks/ $data_dir/select/rand-gl-treegal/rs43/filtered.pkl $data_dir/select/rand-gl-treegal/rs43/ -nc -nm

# YUE-HK
python data/filter.py $data_dir/treebanks/ $data_dir/select/not-yue/filtered.pkl $data_dir/select/rand-yue-hk/rs42 -r "31449,7921,0" -rs 42
python data/cat.py $data_dir/treebanks/ $data_dir/select/rand-yue-hk/rs42/filtered.pkl $data_dir/select/rand-yue-hk/rs42/ -nc -nm
python data/filter.py $data_dir/treebanks/ $data_dir/select/not-yue/filtered.pkl $data_dir/select/rand-yue-hk/rs41 -r "30342,7644,0" -rs 41
python data/cat.py $data_dir/treebanks/ $data_dir/select/rand-yue-hk/rs41/filtered.pkl $data_dir/select/rand-yue-hk/rs41/ -nc -nm
python data/filter.py $data_dir/treebanks/ $data_dir/select/not-yue/filtered.pkl $data_dir/select/rand-yue-hk/rs43 -r "29897,7533,0" -rs 43
python data/cat.py $data_dir/treebanks/ $data_dir/select/rand-yue-hk/rs43/filtered.pkl $data_dir/select/rand-yue-hk/rs43/ -nc -nm

# CKT-HSE
python data/filter.py $data_dir/treebanks/ $data_dir/select/not-ckt/filtered.pkl $data_dir/select/rand-ckt-hse/rs42 -r "30781,7754,0" -rs 42
python data/cat.py $data_dir/treebanks/ $data_dir/select/rand-ckt-hse/rs42/filtered.pkl $data_dir/select/rand-ckt-hse/rs42/ -nc -nm
python data/filter.py $data_dir/treebanks/ $data_dir/select/not-ckt/filtered.pkl $data_dir/select/rand-ckt-hse/rs41 -r "30342,7644,0" -rs 41
python data/cat.py $data_dir/treebanks/ $data_dir/select/rand-ckt-hse/rs41/filtered.pkl $data_dir/select/rand-ckt-hse/rs41/ -nc -nm
python data/filter.py $data_dir/treebanks/ $data_dir/select/not-ckt/filtered.pkl $data_dir/select/rand-ckt-hse/rs43 -r "29669,7476,0" -rs 43
python data/cat.py $data_dir/treebanks/ $data_dir/select/rand-ckt-hse/rs43/filtered.pkl $data_dir/select/rand-ckt-hse/rs43/ -nc -nm

# FO-OFT
python data/filter.py $data_dir/treebanks/ $data_dir/select/not-fo/filtered.pkl $data_dir/select/rand-fo-oft/rs42 -r "49499,11006,0" -rs 42
python data/cat.py $data_dir/treebanks/ $data_dir/select/rand-fo-oft/rs42/filtered.pkl $data_dir/select/rand-fo-oft/rs42/ -nc -nm
python data/filter.py $data_dir/treebanks/ $data_dir/select/not-fo/filtered.pkl $data_dir/select/rand-fo-oft/rs41 -r "49483,11003,0" -rs 41
python data/cat.py $data_dir/treebanks/ $data_dir/select/rand-fo-oft/rs41/filtered.pkl $data_dir/select/rand-fo-oft/rs41/ -nc -nm
python data/filter.py $data_dir/treebanks/ $data_dir/select/not-fo/filtered.pkl $data_dir/select/rand-fo-oft/rs43 -r "50859,11346,0" -rs 43
python data/cat.py $data_dir/treebanks/ $data_dir/select/rand-fo-oft/rs43/filtered.pkl $data_dir/select/rand-fo-oft/rs43/ -nc -nm

# TE-MTG
python data/filter.py $data_dir/treebanks/ $data_dir/select/not-te/filtered.pkl $data_dir/select/rand-te-mtg/rs42 -r "21382,4455,0" -rs 42
python data/cat.py $data_dir/treebanks/ $data_dir/select/rand-te-mtg/rs42/filtered.pkl $data_dir/select/rand-te-mtg/rs42/ -nc -nm
python data/filter.py $data_dir/treebanks/ $data_dir/select/not-te/filtered.pkl $data_dir/select/rand-te-mtg/rs41 -r "21641,4520,0" -rs 41
python data/cat.py $data_dir/treebanks/ $data_dir/select/rand-te-mtg/rs41/filtered.pkl $data_dir/select/rand-te-mtg/rs41/ -nc -nm
python data/filter.py $data_dir/treebanks/ $data_dir/select/not-te/filtered.pkl $data_dir/select/rand-te-mtg/rs43 -r "21382,4455,0" -rs 43
python data/cat.py $data_dir/treebanks/ $data_dir/select/rand-te-mtg/rs43/filtered.pkl $data_dir/select/rand-te-mtg/rs43/ -nc -nm

# MYV-JR
python data/filter.py $data_dir/treebanks/ $data_dir/select/not-myv/filtered.pkl $data_dir/select/rand-myv-jr/rs42 -r "87280,20074,0" -rs 42
python data/cat.py $data_dir/treebanks/ $data_dir/select/rand-myv-jr/rs42/filtered.pkl $data_dir/select/rand-myv-jr/rs42/ -nc -nm
python data/filter.py $data_dir/treebanks/ $data_dir/select/not-myv/filtered.pkl $data_dir/select/rand-myv-jr/rs41 -r "84643,19415,0" -rs 41
python data/cat.py $data_dir/treebanks/ $data_dir/select/rand-myv-jr/rs41/filtered.pkl $data_dir/select/rand-myv-jr/rs41/ -nc -nm
python data/filter.py $data_dir/treebanks/ $data_dir/select/not-myv/filtered.pkl $data_dir/select/rand-myv-jr/rs43 -r "86556,19893,0" -rs 43
python data/cat.py $data_dir/treebanks/ $data_dir/select/rand-myv-jr/rs43/filtered.pkl $data_dir/select/rand-myv-jr/rs43/ -nc -nm

# QHE-HIENCS
python data/filter.py $data_dir/treebanks/ $data_dir/select/not-qhe/filtered.pkl $data_dir/select/rand-qhe-hiencs/rs42 -r "13510,3106,0" -rs 42
python data/cat.py $data_dir/treebanks/ $data_dir/select/rand-qhe-hiencs/rs42/filtered.pkl $data_dir/select/rand-qhe-hiencs/rs42/ -nc -nm
python data/filter.py $data_dir/treebanks/ $data_dir/select/not-qhe/filtered.pkl $data_dir/select/rand-qhe-hiencs/rs41 -r "10147,2265,0" -rs 41
python data/cat.py $data_dir/treebanks/ $data_dir/select/rand-qhe-hiencs/rs41/filtered.pkl $data_dir/select/rand-qhe-hiencs/rs41/ -nc -nm
python data/filter.py $data_dir/treebanks/ $data_dir/select/not-qhe/filtered.pkl $data_dir/select/rand-qhe-hiencs/rs43 -r "11163,2519,0" -rs 43
python data/cat.py $data_dir/treebanks/ $data_dir/select/rand-qhe-hiencs/rs43/filtered.pkl $data_dir/select/rand-qhe-hiencs/rs43/ -nc -nm

# QTD-SAGT
python data/filter.py $data_dir/treebanks/ $data_dir/select/not-qtd/filtered.pkl $data_dir/select/rand-qtd-sagt/rs42 -r "31415,7644,0" -rs 42
python data/cat.py $data_dir/treebanks/ $data_dir/select/rand-qtd-sagt/rs42/filtered.pkl $data_dir/select/rand-qtd-sagt/rs42/ -nc -nm
python data/filter.py $data_dir/treebanks/ $data_dir/select/not-qtd/filtered.pkl $data_dir/select/rand-qtd-sagt/rs41 -r "30017,7294,0" -rs 41
python data/cat.py $data_dir/treebanks/ $data_dir/select/rand-qtd-sagt/rs41/filtered.pkl $data_dir/select/rand-qtd-sagt/rs41/ -nc -nm
python data/filter.py $data_dir/treebanks/ $data_dir/select/not-qtd/filtered.pkl $data_dir/select/rand-qtd-sagt/rs43 -r "29133,7073,0" -rs 43
python data/cat.py $data_dir/treebanks/ $data_dir/select/rand-qtd-sagt/rs43/filtered.pkl $data_dir/select/rand-qtd-sagt/rs43/ -nc -nm