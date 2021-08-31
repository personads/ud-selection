#!/usr/bin/python3

import argparse, hashlib, json, logging, os, sys

import numpy as np
import torch

# local imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from lib.data import *
from lib.utils import *

def parse_arguments():
	arg_parser = argparse.ArgumentParser(description='Universal Dependencies - Contextual Embedding')
	arg_parser.add_argument('ud_path', help='path to Universal Dependencies directory')
	arg_parser.add_argument('model', help='model name in the transformers library')
	arg_parser.add_argument('out_path', help='path to output directory')
	arg_parser.add_argument('-s', '--split', help='path to data split definition pickle (default: None - full UD)')
	arg_parser.add_argument('-mp', '--model_path', help='path to pretrained model (default: None - default pretrained weights)')
	arg_parser.add_argument('-bs', '--batch_size', type=int, default=64, help='batch size while embedding the corpus (default: 64)')
	arg_parser.add_argument('-pl', '--pooling', default='mean', choices=['mean', 'cls', 'none'], help='strategy for pooling word-level embeddings to sentence-level (default: mean)')
	arg_parser.add_argument('-of', '--out_format', default='numpy', choices=['numpy', 'map'], help='output format (default: numpy)')
	return arg_parser.parse_args()

def main():
	args = parse_arguments()

	# check if output dir exists
	setup_output_directory(args.out_path)

	# setup logging
	setup_logging(os.path.join(args.out_path, 'embed.log'))

	# load data split definition (if supplied)
	ud_idcs, ud_filter = None, None
	if args.split:
		with open(args.split, 'rb') as fp:
			splits = pickle.load(fp)
		# create filter to load only relevant indices (train, dev)
		ud_idcs = set(splits['train']) | set(splits['dev']) | set(splits['test'])
		ud_filter = UniversalDependenciesIndexFilter(ud_idcs)
		ud_idcs = sorted(ud_idcs)
		logging.info(f"Loaded data splits {', '.join([f'{s}: {len(idcs)}' for s, idcs in splits.items()])}.")

	# load Universal Dependencies
	ud = UniversalDependencies.from_directory(args.ud_path, ud_filter=ud_filter, verbose=True)
	ud_idcs = list(range(len(ud))) if ud_idcs is None else sorted(ud_idcs)
	logging.info(f"Loaded {ud} with {len(ud_idcs)} relevant sentences.")

	# load transformer model
	model_type = 'standard'
	if args.model.startswith('sentence/'):
		from sentence_transformers import SentenceTransformer
		model = SentenceTransformer(args.model.replace('sentence/', ''))
		tokenizer = model.tokenizer
		model_type = 'sentence'
	else:
		from transformers import AutoTokenizer, AutoModel
		tokenizer = AutoTokenizer.from_pretrained(args.model)
		model = AutoModel.from_pretrained(args.model_path if args.model_path else args.model, return_dict=True)
		# check CUDA availability
		if torch.cuda.is_available(): model.to(torch.device('cuda'))

	logging.info(f"Loaded {model_type}-type '{args.model}' {model.__class__.__name__} with {tokenizer.__class__.__name__}.")

	# main embedding loop
	embeddings = []
	sen_hashes = []
	tkn_count, unk_count, max_len = 0, 0, 1
	cursor = 0
	while cursor < len(ud_idcs):
		# set up batch
		start_idx = cursor
		end_idx = min(start_idx + args.batch_size, len(ud_idcs))
		cursor = end_idx

		sentences = ud[ud_idcs[start_idx:end_idx]]
		batch = [s.to_words() for s in sentences]
		sen_hashes += [hashlib.md5(' '.join(s).encode('utf-8')).hexdigest() for s in batch]

		if tokenizer:
			# tokenize batch: {input_ids: [[]], token_type_ids: [[]], attention_mask: [[]]}
			enc_batch = tokenizer(batch, return_tensors='pt', is_split_into_words=True, padding=True, truncation=True)
			# count tokens and UNK
			tkn_count += int(torch.sum((enc_batch['input_ids'] != tokenizer.pad_token_id)))
			unk_count += int(torch.sum((enc_batch['input_ids'] == tokenizer.unk_token_id)))
			# set maximum length
			cur_max_len = int(torch.max(torch.sum(enc_batch['attention_mask'], dim=-1)))
			max_len = cur_max_len if cur_max_len > max_len else max_len

		# no gradients required during inference
		with torch.no_grad():
			# embed batch (sentence-level)
			if model_type == 'sentence':
				# SentenceTransformer takes list[str] as input
				emb_sentences = model.encode(batch, convert_to_tensor=True) # (batch_size, hidden_dim)
				embeddings += [emb_sentences[sidx] for sidx in range(emb_sentences.shape[0])]
			# embed batch (token-level)
			else:
				# move input to GPU (if available)
				if torch.cuda.is_available():
					enc_batch = {k: v.to(torch.device('cuda')) for k, v in enc_batch.items()}
				# perform embedding forward pass
				model_outputs = model(**enc_batch)
				emb_batch = model_outputs.last_hidden_state # (batch_size, max_len, hidden_dim)
				att_batch = enc_batch['attention_mask'] > 0 # create boolean mask (batch_size, max_len)
				# perform pooling over sentence tokens with specified strategy
				for sidx in range(emb_batch.shape[0]):
					# mean pooling over tokens in each sentence
					if args.pooling == 'mean':
						# reduce (1, max_length, hidden_dim) -> (1, num_tokens, hidden_dim) -> (hidden_dim)
						embeddings.append(torch.mean(emb_batch[sidx, att_batch[sidx], :], dim=0))
					# get cls token from each sentence
					elif args.pooling == 'cls':
						# reduce (1, max_length, hidden_dim) -> (hidden_dim)
						embeddings.append(emb_batch[sidx, 0, :])
					# no reduction
					elif args.pooling == 'none':
						# (sen_len, hidden_dim)
						embeddings.append(emb_batch[sidx, att_batch[sidx], :])

		sys.stdout.write(f"\r[{(cursor*100)/len(ud_idcs):.2f}%] Embedding...")
		sys.stdout.flush()

	print("\r")
	if tkn_count: logging.info(f"{tokenizer.__class__.__name__} encoded corpus to {tkn_count} tokens with {unk_count} UNK tokens ({unk_count/tkn_count:.4f}).")
	logging.info(f"{model.__class__.__name__} embedded corpus with {len(embeddings)} sentences.")

	# export embeddings to numpy arrays
	if args.out_format == 'numpy':
		# TODO export with padding
		if args.pooling == 'none': raise NotImplementedError
		# split embedded corpus by language
		start_idx = 0
		for sidx, uidx in enumerate(ud_idcs):
			cur_language = ud.get_language_of_index(uidx)
			embeddings[sidx] = list(embeddings[sidx])
			# check if next index is new language
			if (sidx == len(ud_idcs) - 1) or (cur_language != ud.get_language_of_index(ud_idcs[sidx+1])):
				# save tensors to disk as numpy arrays
				tensor_path = os.path.join(args.out_path, f'{cur_language.replace(" ", "_")}.npy')
				np.save(tensor_path, np.array(embeddings[start_idx:sidx+1]))
				start_idx = sidx+1
				logging.info(f"Saved embeddings to '{tensor_path}' as numpy array.")

	# export embeddings to hash->emb map
	elif args.out_format == 'map':
		hash_emb_map = {sen_hashes[sidx]: embeddings[sidx] for sidx in range(len(embeddings))}
		# save to disk
		out_file = os.path.join(args.out_path, 'map.pkl')
		with open(out_file, 'wb') as fp:
			pickle.dump(hash_emb_map, fp)
		logging.info(f"Saved 'sentence hash' -> 'embedding tensor' map to '{out_file}'.")


if __name__ == '__main__':
	main()