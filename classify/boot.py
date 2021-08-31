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
	arg_parser = argparse.ArgumentParser(description='Universal Dependencies - Supervised Bootstrap')
	arg_parser.add_argument('ud_path', help='path to Universal Dependencies directory')
	arg_parser.add_argument('model', help='model name in the transformers library')
	arg_parser.add_argument('out_path', help='path to output directory')
	arg_parser.add_argument('-s', '--split', help='path to data split definition pickle (default: None - full UD)')
	arg_parser.add_argument('-e', '--epochs', type=int, default=100, help='maximum number of epochs (default: 100)')
	arg_parser.add_argument('-es', '--early_stop', type=int, default=3, help='maximum number of epochs without improvement (default: 3)')
	arg_parser.add_argument('-bs', '--batch_size', type=int, default=32, help='maximum number of sentences per batch (default: 32)')
	arg_parser.add_argument('-lr', '--learning_rate', type=float, default=1e-7, help='learning rate (default: 1e-7)')
	arg_parser.add_argument('-th', '--threshold', type=float, default=.99, help='classification threshold for bootstrapping sentences (default: .99)')
	arg_parser.add_argument('-rs', '--seed', type=int, default=42, help='seed for probabilistic components (default: 42)')
	return arg_parser.parse_args()


def get_schedule(domain_combinations):
	schedule = []

	domains = {d for dc in domain_combinations for d in dc}
	known_domains = {dc[0] for dc in domain_combinations if len(dc) == 1}
	known_combinations = {dc for dc in domain_combinations if len(dc) == 1}
	prev_num_known_domains = -1

	# iterate until all domains can be resolved or until no further reduction was achieved in the last iteration
	while prev_num_known_domains != len(known_domains):
		environment = {'known': [], 'predict': [], 'disjunct': []}
		prev_num_known_domains = len(known_domains)

		# all truly single domains act as positive training examples in that
		environment['known'] = list(sorted(known_domains))

		# predict known domain sentences from domain combinations containing them
		predict_combinations = {dc for dc in domain_combinations if (len(set(dc) & known_domains) > 0) and (dc not in known_combinations)}
		environment['predict'] = list(sorted(predict_combinations))

		# unknown domains are disjunct from known single domains
		unknown_combinations = {dc for dc in domain_combinations if len(set(dc) & known_domains) == 0}
		environment['disjunct'] = list(sorted(unknown_combinations))

		# determine new known domains
		new_known_domains = {(set(dc) - known_domains).pop() for dc in domain_combinations if len(set(dc) - known_domains) == 1}
		known_domains |= new_known_domains
		new_known_combinations = {dc for dc in domain_combinations if len(set(dc) - known_domains) == 0}
		known_combinations |= new_known_combinations

		schedule.append(environment)

	return schedule


def get_batches(ud, index_label_map, size):
	relevant_indices = set(index_label_map.keys())

	while len(relevant_indices) > 0:
		batch_idcs = list(np.random.choice(list(relevant_indices), min(size, len(relevant_indices)), replace=False))
		batch = [s.to_text() for s in ud[batch_idcs]]

		labels = [index_label_map[idx] for idx in batch_idcs]

		relevant_indices = relevant_indices - set(batch_idcs)
		num_remaining = len(relevant_indices)

		yield batch, labels, num_remaining


def forward(model, model_type, tokenizer, objective, batch, labels=None):
	# tokenize batch: {input_ids: [[]], token_type_ids: [[]], attention_mask: [[]]}
	tkn_batch = tokenizer(batch, return_tensors='pt', padding=True, truncation=True)
	# include labels if available
	if labels: tkn_batch['labels'] = torch.LongTensor(labels) # (batch_size,)
	# move batch to GPU (if available)
	if torch.cuda.is_available():
		tkn_batch = {k: v.to(torch.device('cuda')) for k, v in tkn_batch.items()}

	# perform standard transformer embedding forward pass
	model_out = model(**tkn_batch)
	loss_out = model_out.loss # scalar
	logits = model_out.logits # (batch_size, num_labels)

	return loss_out, logits


def run(model, model_type, tokenizer, objective, optimizer, batch_generator, num_total, mode='train'):
	stats = defaultdict(list)

	# set model to training mode
	if mode == 'train':
		model.train()
	# set model to eval mode
	elif mode == 'eval':
		model.eval()

	# iterate over batches
	for bidx, batch_data in enumerate(batch_generator):
		batch, labels, num_remaining = batch_data

		# when training, perform both forward and backward pass
		if mode == 'train':
			# zero out previous gradients
			optimizer.zero_grad()

			loss_out, logits = forward(model, model_type, tokenizer, objective, batch, labels)

			# propagate loss
			loss_out.backward()
			optimizer.step()

		# when evaluating, perform forward pass without gradients
		elif mode == 'eval':
			with torch.no_grad():
				loss_out, logits = forward(model, model_type, tokenizer, objective, batch, labels)

		# store statistics
		stats[f'{mode}/loss'].append(float(loss_out))

		# print batch statistics
		pct_complete = (1 - (num_remaining/num_total))*100
		sys.stdout.write(f"\r[{mode.capitalize()} | Batch {bidx+1} | {pct_complete:.2f}%] \
Size {tuple(logits.shape)}, Loss: {loss_out:.4f} ({np.mean(stats[f'{mode}/loss']):.4f} mean)")
		sys.stdout.flush()

	# clear line
	print("\r", end='')

	return stats


def predict(ud, model, model_type, tokenizer, predict_indices, domain_labels, known_labels, threshold, batch_size):
	predictions = {}
	predict_indices = set(predict_indices)

	# set model to eval mode
	model.eval()

	# iterate over batches
	num_total = len(predict_indices)
	bidx = 0
	while len(predict_indices) > 0:
		batch_indices = list(predict_indices)[:min(batch_size, len(predict_indices))]
		predict_indices -= set(batch_indices)
		batch = [s.to_text() for s in ud[batch_indices]]

		with torch.no_grad():
			# perform transformer embedding forward pass without labels
			_, logits = forward(model, model_type, tokenizer, objective=None, batch=batch) # (batch_size, num_labels)

			# gather predictions
			for bidx, sidx in enumerate(batch_indices):
				# gather possible domains of sentence
				sen_domains = ud.get_domains_of_index(sidx)
				# map sentence domains to global domains
				sen_domain_labels = [domain_labels[d] for d in sen_domains]
				# softmax over possible domain logits
				sen_domain_probabilities = F.softmax(logits[bidx, sen_domain_labels], dim=-1) # (num_sentence_domains)
				# get most likely domain index
				sen_domain_prediction = int(torch.argmax(sen_domain_probabilities, dim=-1).cpu()) # label index (in sentence_domain_labels)
				domain_prediction = sen_domain_labels[sen_domain_prediction] # label index in global domains

				# check if prediction fulfills requirements
				if (domain_prediction in known_labels) and (sen_domain_probabilities[sen_domain_prediction] >= threshold):
					predictions[sidx] = domain_prediction

		# print batch statistics
		pct_complete = (1 - (len(predict_indices)/num_total))*100
		sys.stdout.write(f"\r[Predict | Batch {bidx+1} | {pct_complete:.2f}%] Size {tuple(logits.shape)}, \
Added {len(predictions)} ({(len(predictions) * 100)/num_total:.2f}%)")
		sys.stdout.flush()
		bidx += 1

	# clear line
	print("\r", end='')

	return predictions


def main():
	args = parse_arguments()

	# check if output dir exists
	setup_output_directory(args.out_path)

	# setup logging
	setup_logging(os.path.join(args.out_path, 'boot.log'))

	# set random seed
	transformers.set_seed(args.seed)
	torch.random.manual_seed(args.seed)
	np.random.seed(args.seed)

	# load data split definition (if supplied)
	ud_filter = None
	split_idcs = None
	relevant_idcs = None
	if args.split:
		with open(args.split, 'rb') as fp:
			split_idcs = pickle.load(fp)
		# create filter to load only relevant indices (train, dev)
		relevant_idcs = set(split_idcs['train']) | set(split_idcs['dev'])
		ud_filter = UniversalDependenciesIndexFilter(relevant_idcs)
		logging.info(f"Loaded data splits {', '.join([f'{s}: {len(idcs)}' for s, idcs in split_idcs.items()])} with filter {ud_filter}.")

	# load Universal Dependencies
	ud = UniversalDependencies.from_directory(args.ud_path, ud_filter=ud_filter, verbose=True)
	domains = ud.get_domains()
	domain_labels = {domain:didx for didx, domain in enumerate(domains)}
	logging.info(f"Loaded {ud} with {len(domains)} domains ({', '.join(domains)}).")

	# use all of UD for each split if none are provided
	if split_idcs is None:
		relevant_idcs = set(range(len(ud)))
		split_idcs = {split: list(range(len(ud))) for split in ['train', 'dev', 'test']}

	# create index maps for easy retrieval by domain combination
	combination_indices = defaultdict(set)
	cursor = 0
	for tb in ud.get_treebanks():
		tb_idcs = set(range(cursor, cursor + len(tb))) & relevant_idcs
		if len(tb_idcs) > 0:
			combination_indices[tuple(sorted(tb.get_domains()))] |= tb_idcs
		cursor += len(tb)
	logging.info(f"Gathered {len(combination_indices)} possible domain combinations.")

	# create schedule
	schedule = get_schedule(list(combination_indices.keys()))
	logging.info(f"Created training schedule with {len(schedule)} environments:")
	for env_idx, environment in enumerate(schedule):
		logging.info(f"  Env {env_idx + 1}:")
		for key in sorted(environment):
			logging.info(f"    {key} ({len(environment[key])}): {environment[key]}")
	if len(schedule[-1]['disjunct']) > 0:
		logging.error(f"[Error] Unable to bootstrap using this schedule. Exiting.")
		return

	# load transformer model
	model_type = 'standard'
	tokenizer = transformers.AutoTokenizer.from_pretrained(args.model)
	model = transformers.AutoModelForSequenceClassification.from_pretrained(
		args.model,
		num_labels=len(domains),
		return_dict=True
	)
	objective = None # use built in loss function
	logging.info(f"Loaded {model_type} '{args.model}' ({model.__class__.__name__} with {tokenizer.__class__.__name__}).")

	# move to CUDA device if available
	if torch.cuda.is_available(): model.to(torch.device('cuda'))

	# initialize optimizer
	optimizer = torch.optim.AdamW(params=model.parameters(), lr=args.learning_rate)
	logging.info(f"Optimizing using {optimizer.__class__.__name__} with learning rate {args.learning_rate}.")

	# initialize known indices from single domain treebanks
	known_indices = {}
	for combination, indices in combination_indices.items():
		if len(combination) > 1: continue
		known_indices.update({idx:domain_labels[combination[0]] for idx in indices})

	# iterate over schedules
	for env_idx, environment in enumerate(schedule):
		# gather known labels
		known_labels = [domain_labels[d] for d in environment['known']]
		# gather training and validation indices
		train_indices = {idx:known_indices[idx] for idx in (set(split_idcs['train']) & set(known_indices))}
		eval_indices = {idx:known_indices[idx] for idx in (set(split_idcs['dev']) & set(known_indices))}
		# gather indices to predict
		predict_indices = set()
		if len(environment['predict']) > 0:
			predict_indices = set.union(*[combination_indices[dc] for dc in environment['predict']])
		logging.info(f"[Env {env_idx + 1}/{len(schedule)}] Covering {len(known_labels)}/{len(domain_labels)} domains. \
{len(known_indices)} sentences with fixed labels ({len(train_indices)} train / {len(eval_indices)} eval). \
Predicting labels for {len(predict_indices)} indices.")

		# train on known domains
		stats = defaultdict(list)
		for ep_idx in range(args.epochs):
			# iterate over batches with known domain labels
			train_batches = get_batches(ud, train_indices, args.batch_size)
			cur_stats = run(
				model, model_type, tokenizer, objective, optimizer,
				train_batches, len(train_indices)
			)
			# store statistics
			stats['train/loss'].append(np.mean(cur_stats['train/loss']))
			# print training statistics
			logging.info(f"[Env {env_idx + 1}/{len(schedule)} | Epoch {ep_idx+1}/{args.epochs}] Training completed with \
Loss: {np.mean(cur_stats['train/loss']):.4f} ({max(cur_stats['train/loss']):.4f} max, {min(cur_stats['train/loss']):.4f} min)")

			# iterate over batches with known domain labels
			eval_batches = get_batches(ud, eval_indices, args.batch_size)
			cur_stats = run(
				model, model_type, tokenizer, objective, optimizer,
				eval_batches, len(eval_indices), mode='eval'
			)
			# store statistics
			stats['eval/loss'].append(np.mean(cur_stats['eval/loss']))
			# print training statistics
			logging.info(f"[Env {env_idx + 1}/{len(schedule)} | Epoch {ep_idx+1}/{args.epochs}] Evaluation completed with \
Loss: {np.mean(cur_stats['eval/loss']):.4f} ({max(cur_stats['eval/loss']):.4f} max, {min(cur_stats['eval/loss']):.4f} min)")

			# save most recent model
			model_path = os.path.join(args.out_path, 'newest')
			if model_type == 'sentence':
				model.save(model_path)
			else:
				model.save_pretrained(model_path)
			logging.info(f"Saved model from epoch {ep_idx + 1} to '{model_path}'.")

			# save best model
			if np.mean(cur_stats['eval/loss']) <= min(stats['eval/loss']):
				model_path = os.path.join(args.out_path, 'best')
				if model_type == 'sentence':
					model.save(model_path)
				else:
					model.save_pretrained(model_path)
				logging.info(f"Saved model with best loss {np.mean(cur_stats['eval/loss']):.4f} to '{model_path}'.")

			# check for early stopping
			if (ep_idx - stats['eval/loss'].index(min(stats['eval/loss']))) >= args.early_stop:
				logging.info(f"No improvement since {args.early_stop} epochs ({min(stats['eval/loss']):.4f} loss). Early stop.")
				break

		# exit if there are no more predictions to make
		if len(predict_indices) < 1:
			logging.info(f"No more predictions to make. Schedule complete.")
			break

		# load previous best model
		model_path = os.path.join(args.out_path, 'best')
		model.from_pretrained(model_path)
		if torch.cuda.is_available(): model.to(torch.device('cuda'))
		
		# get sentences with domain combinations containing known domains
		predictions = predict(ud, model, model_type, tokenizer, predict_indices, domain_labels, known_labels, args.threshold, args.batch_size)
		known_indices.update(predictions)
		# print training statistics
		logging.info(f"[Env {env_idx + 1}/{len(schedule)}] Predicted labels for {len(predictions)} sentences \
above threshold {args.threshold} ({(len(predictions)*100)/len(predict_indices):.2f}%).")

		# gather new single domains
		num_inferred = defaultdict(int)
		for combination in environment['predict']:
			# skip cases where none or more than one ambiguous domain is left
			remaining = set(combination) - set(environment['known'])
			if len(remaining) != 1: continue

			# set all remaining indices of combination to that class
			new_single_domain = remaining.pop()
			inferred_indices = {
				idx: domain_labels[new_single_domain]
				for idx in (combination_indices[combination] - set(known_indices.keys()))
			}
			known_indices.update(inferred_indices)

			num_inferred[new_single_domain] += len(inferred_indices)
		logging.info(f"[Env {env_idx + 1}/{len(schedule)}] Inferred domains {', '.join([f'{n} {d}' for d, n in num_inferred.items()])}.")


if __name__ == '__main__':
	main()
