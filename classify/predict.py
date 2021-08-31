#!/usr/bin/python3

import argparse, json, logging, os, sys
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import transformers

from collections import defaultdict, OrderedDict

# local imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from lib.data import *
from lib.utils import *


def parse_arguments():
	arg_parser = argparse.ArgumentParser(description='Universal Dependencies - Classifier Prediction')
	arg_parser.add_argument('ud_path', help='path to Universal Dependencies directory')
	arg_parser.add_argument('model', help='model name in the transformers library')
	arg_parser.add_argument('exp_path', help='path to tuning experiment directory')
	arg_parser.add_argument('-bs', '--batch_size', type=int, default=64, help='batch size (default: 64)')
	arg_parser.add_argument('-cd', '--closed_domain', action='store_true', default=False, help='limit prediction to set of known domains (default: False)')
	return arg_parser.parse_args()


def main():
	args = parse_arguments()

	# check if experiment dir exists
	if not os.path.exists(args.exp_path):
		print(r"[Error] Could not find experiment directory '{args.exp_path}'. Exiting.")
		return

	# setup logging
	setup_logging(os.path.join(args.exp_path, 'predict.log'))

	# load Universal Dependencies
	ud = UniversalDependencies.from_directory(args.ud_path, verbose=True)
	domains = ud.get_domains()
	domain_labels = {domain: didx for didx, domain in enumerate(domains)}
	logging.info(f"Loaded {ud} with {len(domains)} domains ({', '.join(domains)}).")

	# load transformer model
	model_type = 'standard'
	tokenizer = transformers.AutoTokenizer.from_pretrained(args.model)
	model = transformers.AutoModelForSequenceClassification.from_pretrained(
		os.path.join(args.exp_path, 'best'),
		num_labels=len(domains),
		return_dict=True
	)
	# set model to inference mode
	model.eval()
	# move to CUDA device if available
	if torch.cuda.is_available(): model.to(torch.device('cuda'))
	logging.info(f"Loaded {model_type}-type '{args.model}' ({model.__class__.__name__} with {tokenizer.__class__.__name__}).")

	# initialize results
	domain_dist = np.zeros((len(ud), len(domains)))
	corpus_map = OrderedDict([('All', OrderedDict())])

	# iterate over UD
	tbf_cursor = 0
	cursor = 0
	while cursor < len(ud):
		# set up batch
		start_idx = cursor
		end_idx = min(start_idx + args.batch_size, len(ud))
		cursor = end_idx
		batch = [s.to_text() for s in ud[start_idx:end_idx]]

		with torch.no_grad():
			# tokenize batch: {input_ids: [[]], token_type_ids: [[]], attention_mask: [[]]}
			tkn_batch = tokenizer(batch, return_tensors='pt', padding=True, truncation=True)
			# move batch to GPU (if available)
			if torch.cuda.is_available():
				tkn_batch = {k: v.to(torch.device('cuda')) for k, v in tkn_batch.items()}

			# perform standard transformer embedding forward pass
			model_out = model(**tkn_batch)
			logits = model_out.logits.cpu() # (batch_size, num_labels)
			# apply domain mask for closed domain prediction
			if args.closed_domain:
				softmax = torch.zeros_like(logits)
				for bidx, udidx in enumerate(range(start_idx, end_idx)):
					sen_labels = [domain_labels[d] for d in ud.get_domains_of_index(udidx)]
					softmax[bidx, sen_labels] = F.softmax(logits[bidx, sen_labels], dim=-1)
			else:
				softmax = F.softmax(logits, dim=-1)

		# append distribution to results
		domain_dist[start_idx:end_idx, :] = np.array(softmax)

		# update corpus map
		for sidx in range(start_idx, end_idx):
			language = ud.get_language_of_index(sidx).replace(' ', '_')
			treebank = ud.get_treebank_name_of_index(sidx)
			tb_key = f'{language}/{treebank}'
			if tb_key not in corpus_map['All']:
				corpus_map['All'][tb_key] = OrderedDict()

			tb_file = ud.get_treebank_file_of_index(sidx)
			if tb_file not in corpus_map['All'][tb_key]:
				corpus_map['All'][tb_key][tb_file] = OrderedDict()
				tbf_cursor = 0

			sentence_key = f'sentence-{tbf_cursor}'
			corpus_map['All'][tb_key][tb_file][sentence_key] = [sidx]
			tbf_cursor += 1

		sys.stdout.write(f"\r[{(cursor*100)/len(ud):.2f}%] Predicting...")
		sys.stdout.flush()

	# export results
	results_path = os.path.join(args.exp_path, 'All.pkl')
	with open(results_path, 'wb') as fp:
		pickle.dump(
			{
				'corpus_map': corpus_map,
				'domain_dist': domain_dist,
				'domains': domains
			},
			fp
		)
	logging.info(f"\rPredicted {'closed-domain' if args.closed_domain else 'open-domain'} labels for {len(ud)} sentences. \
Saved results to '{results_path}'.")


if __name__ == '__main__':
	main()