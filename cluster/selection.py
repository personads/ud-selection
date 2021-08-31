#!/usr/bin/python3

import argparse, logging, os, pickle, random, sys

from collections import OrderedDict

import numpy as np

from scipy.spatial.distance import cosine

# local imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from lib.data import *
from lib.utils import *


def parse_arguments():
	arg_parser = argparse.ArgumentParser(description='Universal Dependencies - Cluster Data Selection')
	arg_parser.add_argument('ud_path', help='path to Universal Dependencies directory')
	arg_parser.add_argument('emb_path', help='path to embedding directory')
	arg_parser.add_argument('exp_path', help='path to clustering experiment directory')
	arg_parser.add_argument('out_path', help='path to output directory')
	arg_parser.add_argument('tgt_split', help='path to split pickle defining the target data')
	arg_parser.add_argument(
		'-ts', '--target_size', type=int, default=0, help='number of sentences from target data to use (default: 0 - all)'
	)
	arg_parser.add_argument(
		'-us', '--ud_split', help='path to UD split definition from which to select data (default: None - all of UD)'
	)
	arg_parser.add_argument(
		'-cl', '--cluster_level', choices=['all', 'language', 'treebank', 'sentence'], default='language',
		help='data granularity at which to retrieve clustering results (default: language)'
	)
	arg_parser.add_argument(
		'-tk', '--top_k', type=int, default=0, help='top k clusters to select data from (default: 0 - all clusters)'
	)
	arg_parser.add_argument(
		'-op', '--output_proportions', default='.8,.2',
		help='train and dev proportions as comma-separated floats (default: ".8,.2")'
	)
	arg_parser.add_argument(
		'-rs', '--seed', type=int, default=42, help='seed for probabilistic components (default: 42)'
	)
	return arg_parser.parse_args()


def main():
	args = parse_arguments()

	setup_success = setup_output_directory(args.out_path)

	# setup logging
	log_path = os.path.join(args.out_path, 'selection.log')
	setup_logging(log_path)

	# setup random seeds
	random.seed(args.seed)
	np.random.seed(args.seed)

	# parse output proportions
	proportions = [float(pstr) for pstr in args.output_proportions.split(',')]
	assert sum(proportions) == 1, f"[Error] Split proportions {proportions} do not sum up to 1."
	assert len(proportions) == 2, f"[Error] There must be 2 proportions (i.e. train, dev)."

	# load Universal Dependencies
	ud_filter = UniversalDependenciesIndexFilter(set()) # no need to load textual content of any sentence
	ud = UniversalDependencies.from_directory(args.ud_path, ud_filter=ud_filter, verbose=True)
	logging.info(f"Loaded {ud}.")

	# load pre-computed embeddings if provided
	embeddings = load_embeddings(args.emb_path, 'All')
	logging.info(f"Loaded {embeddings.shape[0]} pre-computed embeddings from '{args.emb_path}'.")
	assert embeddings.shape[0] == len(ud), f"[Error] Number of embeddings (n={embeddings.shape[0]}) and sentences in UD (n={len(ud)}) do not match."

	# load data generator for appropriate level
	data_generator = None
	if args.cluster_level == 'all':
		data_generator = [('All', ud[0:len(ud)])]
	elif args.cluster_level == 'language':
		data_generator = ud.get_sentences_by_language()
	elif args.cluster_level == 'treebank':
		data_generator = ud.get_sentences_by_treebank()
	if args.cluster_level == 'sentence':
		data_generator = [(f'{sidx}/{ud.get_treebank_file_of_index(sidx)}', [s]) for sidx, s in enumerate(ud)]

	# get target data embedding
	with open(args.tgt_split, 'rb') as fp:
		target_split = pickle.load(fp)
	target_idcs = []
	if len(target_split['train']) > 0:
		target_idcs = target_split['train']
	elif len(target_split['test']) > 0:
		target_idcs = target_split['test']
		logging.warning("[Warning] No train split for target data. Falling back to test.")
	else:
		logging.error("[Error] Target split definitions seem to be empty. Exiting.")
		return
	# subsample target indices
	if (args.target_size > 0) and (len(target_idcs) > args.target_size):
		target_idcs = np.random.choice(target_idcs, args.target_size, replace=False)
	# mean pool target sentence embeddings
	target_embedding = np.mean(embeddings[target_idcs], axis=0)
	# gather target metadata
	target_languages, target_treebanks, target_genres = set(), set(), set()
	for tidx in target_idcs:
		target_languages.add(ud.get_language_of_index(tidx))
		target_treebanks.add(ud.get_treebank_name_of_index(tidx))
		target_genres |= set(ud.get_domains_of_index(tidx))
	logging.info(f"Gathered target embedding from {len(target_idcs)} sentences:")
	logging.info(f"  Language(s): {', '.join(sorted(target_languages))}")
	logging.info(f"  Treebank(s): {', '.join(sorted(target_treebanks))}")
	logging.info(f"  Genre(s): {', '.join(sorted(target_genres))}")

	# by default, choose data from all of UD
	relevant_idcs = set(range(len(ud)))
	# if provided, load existing split
	if args.ud_split:
		with open(args.ud_split, 'rb') as fp:
			split_idcs = pickle.load(fp)
		# construct index-based filter
		relevant_idcs = set(split_idcs['train']) | set(split_idcs['dev']) | set(split_idcs['test'])
		logging.info(f"Loaded selection data splits {', '.join([f'{s}: {len(idcs)}' for s, idcs in split_idcs.items()])}).")
	# remove target instances from selection pool
	relevant_idcs -= set(target_idcs)
	logging.info(f"Loaded {len(relevant_idcs)} instances relevant for data selection.")

	# iterate over cluster results
	logging.info("---")
	logging.info("Gathering data candidates from clustering results:")
	candidates = OrderedDict()
	cursor = 0
	for lvl_key, sentences in data_generator:
		# special case: sentence-level selection
		if args.cluster_level == 'sentence':
			# skip non-relevant sentences
			if cursor in relevant_idcs:
				# add sentence to candidates
				candidates[lvl_key] = {
					'ud_idcs': [cursor],
					'cosine': cosine(embeddings[cursor], target_embedding)
				}
			cursor += 1
			continue

		# standard cases
		# load results dict
		with open(os.path.join(args.exp_path, f'{lvl_key.replace(" ", "_")}.pkl'), 'rb') as fp:
			results = pickle.load(fp)
		# check if results contain cluster assignments
		if 'domain_dist' not in results: continue
		assignments = results['domain_dist'].argmax(axis=-1)
		num_clusters = results['domain_dist'].shape[-1]
		num_sentences = results['domain_dist'].shape[0]
		assert num_sentences == len(sentences), f"Mismatch between number of clustering results (n={num_sentences}) and {args.cluster_level} sentences (n={len(sentences)}) "

		# mean pool each cluster
		closest_cluster = -1
		for cidx in range(num_clusters):
			cluster_idcs = np.where(assignments == cidx)[0]
			cluster_idcs += np.ones_like(cluster_idcs) * cursor # increment local idcs by current offset

			# filter out non-relevant instances in clusters
			cluster_idcs = sorted(set(cluster_idcs) & relevant_idcs)
			if len(cluster_idcs) < 1: continue

			emb_cluster = np.mean(embeddings[cluster_idcs], axis=0)
			cos_dist = cosine(emb_cluster, target_embedding)
			# update closest cluster of current set
			if (lvl_key not in candidates) or (cos_dist < candidates[lvl_key]['cosine']):
				candidates[lvl_key] = {
					'ud_idcs': cluster_idcs,
					'cosine': cos_dist
				}
				closest_cluster = cidx

		# increment cursor
		cursor += num_sentences
		if lvl_key in candidates: logging.info(f"  Added cluster ID={closest_cluster} from {num_clusters} clusters in {lvl_key} to candidates.")

	# select K closest clusters (or all)
	logging.info(f"Selected {f'top {args.top_k}' if args.top_k > 0 else 'all'} clusters:")
	selection = []
	for cidx, (candidate, criteria) in enumerate(sorted(candidates.items(), key=lambda el: el[1]['cosine'], reverse=True)):
		if (args.top_k > 0) and (cidx >= args.top_k): break
		selection += list(criteria['ud_idcs'])
		logging.info(f"  {candidate}: {len(criteria['ud_idcs'])} instances, {criteria['cosine']:.2f} cosine")

	# create appropriate data splits
	random.shuffle(selection)
	selection_split = {
		'train': sorted(selection[:int(len(selection) * proportions[0])]),
		'dev': sorted(selection[int(len(selection) * proportions[0]):]),
		'test': []
	}
	selection_split_path = os.path.join(args.out_path, 'selection.pkl')
	with open(selection_split_path, 'wb') as fp:
		pickle.dump(selection_split, fp)
	logging.info(f"Saved selection split indices to '{selection_split_path}' \
with {', '.join([f'{s}: {len(idcs)}' for s, idcs in selection_split.items()])}.")


if __name__ == '__main__':
	main()
