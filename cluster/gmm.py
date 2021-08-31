#!/usr/bin/python3

import argparse, copy, json, logging, os, pickle, sys

import numpy as np

from collections import defaultdict, OrderedDict

from sklearn.mixture import GaussianMixture

# local imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from lib.data import *
from lib.utils import *


def parse_arguments():
	arg_parser = argparse.ArgumentParser(description='Universal Dependencies - Gaussian Mixture Model Clustering')
	arg_parser.add_argument('ud_path', help='path to Universal Dependencies directory')
	arg_parser.add_argument('emb_path', help='path to embedding directory')
	arg_parser.add_argument('out_path', help='path to output directory')
	arg_parser.add_argument('-cl', '--cluster_level', choices=['all', 'language', 'treebank'], default='language', help='data granularity at which to run clustering (default: language)')
	arg_parser.add_argument('-rs', '--seed', type=int, default=42, help='random seed for all probabilistic components (default: 42)')
	return arg_parser.parse_args()


def run_gmm(embeddings, num_domains, random_state=42):
	# perform clustering using GMM
	gmm = GaussianMixture(n_components=num_domains, random_state=random_state, verbose=True)
	gmm.fit(embeddings) # embeddings: (corpus_size, hidden_dim)
	domain_dist = gmm.predict_proba(embeddings) # (corpus_size, num_domains)

	return domain_dist


def main():
	args = parse_arguments()

	setup_success = setup_output_directory(args.out_path)
	if not setup_success: return

	# setup logging
	log_path = os.path.join(args.out_path, 'clustering.log')
	setup_logging(log_path)

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
	else:
		logging.error(f"[Error] Unknown cluster level '{args.cluster_level}'. Exiting.")
		return

	# iterate over languages
	logging.info("---")
	num_gmms = 0
	cursor = 0
	for lvl_key, sentences in data_generator:
		# set up data
		ud_idcs = list(range(cursor, cursor+len(sentences)))
		cur_embeddings = embeddings[ud_idcs]
		genres = {g for i in ud_idcs for g in ud.get_domains_of_index(i)}
		logging.info(f"Clustering {args.cluster_level.capitalize()} '{lvl_key}' ({', '.join(sorted(genres))}):")

		# check if there is more than one domain in the corpus
		if len(genres) < 2:
			logging.info(f"  {lvl_key} only has {len(genres)} domain(s). Assigning to one cluster.")
			domain_dist = np.ones((len(sentences), 1))
		# run clustering
		else:
			logging.info("  Fitting GMMs...")
			domain_dist = run_gmm(cur_embeddings, len(genres), random_state=args.seed)

		# store results on disk
		results_path = os.path.join(args.out_path, f'{lvl_key.replace(" ", "_")}.pkl')
		with open(results_path, 'wb') as fp:
			pickle.dump(
				{
					# 'corpus_map': corpus_map,
					'domain_dist': domain_dist
				},
				fp
			)
		logging.info(f"  Clustered into {domain_dist.shape[1]} domains and stored results in '{results_path}'.\n---")

		# count number of items analyzed
		num_gmms += 1

		cursor += len(sentences)

	logging.info(f"Ran GMMs for {num_gmms} {args.cluster_level}s.")


if __name__ == '__main__':
	main()
