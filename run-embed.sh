#!/bin/bash

data_dir=/path/to/data
exp_dir=/path/to/experiments

python data/embed.py $data_dir/treebanks bert-base-multilingual-cased $exp_dir/emb -p mean