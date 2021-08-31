#!/usr/bin/python3

import argparse, logging, os, pickle, sys

from collections import defaultdict, OrderedDict

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# local imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from lib.data import *
from lib.utils import *


def parse_arguments():
	arg_parser = argparse.ArgumentParser(description='Universal Dependencies - Latent Dirichlet Allocation')
	arg_parser.add_argument('ud_path', help='path to Universal Dependencies directory')
	arg_parser.add_argument('out_path', help='path to output directory')
	arg_parser.add_argument(
		'-cl', '--cluster_level', choices=['all', 'language', 'treebank'], default='language',
		help='data granularity at which to run clustering (default: language)')
	arg_parser.add_argument(
		'-vu', '--vectorizer_unit', choices=['word', 'char'], default='word',
		help='unit of the sentence vectorizer (default: "word")')
	arg_parser.add_argument(
		'-vn', '--vectorizer_ngrams', default='1-1',
		help='ngram range on the sentence vectorizer (default: "1-1")')
	arg_parser.add_argument(
		'-vm', '--max_df', type=float, default=.3,
		help='maximum fraction of sentences a token may appear in before being excluded from vocabulary (default: 0.3)')
	arg_parser.add_argument(
		'-rs', '--seed', type=int, default=42,
		help='random seed for all probabilistic components (default: 42)')
	args = arg_parser.parse_args()
	# parse ngram range
	args.vectorizer_ngrams = tuple([int(i) for i in args.vectorizer_ngrams.split('-')])
	return args


def identity(sentence):
	return sentence


def run_lda(corpus, vectorizer, num_domains, random_state=42):
	# feature extraction
	word_matrix = vectorizer.fit_transform(corpus)
	idx_word_map = vectorizer.get_feature_names()

	# perform LDA
	lda = LatentDirichletAllocation(n_components=num_domains, max_iter=100, random_state=random_state)
	lda.fit(word_matrix)
	topic_dist = lda.transform(word_matrix)

	# analyse topics
	topics = []
	for topic in lda.components_:
		topic_words = np.flip(topic.argsort())
		topics.append(' | '.join([idx_word_map[widx] for widx in topic_words]))

	return topics, topic_dist


def main():
	args = parse_arguments()

	setup_success = setup_output_directory(args.out_path)
	if not setup_success: return

	# setup logging
	log_path = os.path.join(args.out_path, 'clustering.log')
	setup_logging(log_path)

	# load Universal Dependencies
	ud = UniversalDependencies.from_directory(args.ud_path, verbose=True)
	logging.info(f"Loaded {ud}.")

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
	num_ldas = 0
	for lvl_key, sentences in data_generator:
		# set up data
		corpus = []
		genres = set()
		for sentence in sentences:
			corpus.append(sentence.to_words() if args.vectorizer_unit == 'words' else sentence.to_text())
			genres |= set(ud.get_domains_of_index(sentence.idx))
		logging.info(f"Clustering {args.cluster_level.capitalize()} '{lvl_key}' ({', '.join(sorted(genres))}):")

		# check if there is more than one domain in the corpus
		if len(genres) < 2:
			logging.info(f"  {lvl_key} only has {len(genres)} domain(s). Assigning to one cluster.")
			domains = [next(iter(genres))]
			domain_dist = np.ones((len(sentences), 1))
		# run clustering
		else:
			logging.info("  Fitting LDA...")
			# check if number of items in corpus is too low
			max_df = args.max_df
			if len(corpus) * args.max_df < 2:
				max_df = 2/len(corpus)
				logging.warning(f"  [Warning] Corpus is too small to apply max_df={args.max_df}. Increased to {max_df}.")

			# setup vectorizer
			vectorizer = CountVectorizer(
				analyzer=args.vectorizer_unit,
				tokenizer=identity,
				preprocessor=identity,
				token_pattern=None,
				ngram_range=args.vectorizer_ngrams,
				max_df=args.max_df,
				min_df=2
			)
			logging.info(
				f"  Vectorizing input using {vectorizer.__class__.__name__} on {args.vectorizer_ngrams} "
				f"{args.vectorizer_unit}-ngrams and max_df={args.max_df}."
			)
			# run LDA
			domains, domain_dist = run_lda(corpus, vectorizer, len(genres), random_state=args.seed)

		# store results on disk
		results_path = os.path.join(args.out_path, f'{lvl_key.replace(" ", "_")}.pkl')
		with open(results_path, 'wb') as fp:
			pickle.dump(
				{
					'domains': domains,
					'domain_dist': domain_dist
				},
				fp
			)
		logging.info(f"  Clustered into {domain_dist.shape[1]} domains and stored results in '{results_path}'.\n---")

		# count number of items analyzed
		num_ldas += 1

	logging.info(f"Ran LDA for {num_ldas} {args.cluster_level}s.")


if __name__ == '__main__':
	main()