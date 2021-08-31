import copy, itertools, logging, os, pickle, re, sys

from collections import defaultdict, OrderedDict

import numpy as np

from lib.data import *

#
# Experiment Setup Functions
#


def setup_output_directory(out_path):
	if os.path.exists(out_path):
		response = None
		while response not in ['y', 'n']:
			response = input(f"Path '{out_path}' already exists. Overwrite? [y/n] ")
		if response == 'n':
			exit(1)
	# if output dir does not exist, create it
	else:
		print(f"Path '{out_path}' does not exist. Creating...")
		os.mkdir(out_path)
	return True


def setup_logging(log_path):
	log_format = '%(message)s'
	log_level = logging.INFO
	logging.basicConfig(filename=log_path, filemode='w', format=log_format, level=log_level)

	logger = logging.getLogger()
	logger.addHandler(logging.StreamHandler(sys.stdout))


#
# Data Loading and Manipulation Functions
#


def load_corpus(ud_path, lang, ud_stats, group='sentences', strict_groups=False, tokenized=True):
	# corpus should be a list of str-lists where each str-list is one document in the eyes of LDA
	corpus = []
	# the corpus map stores 'lang' -> 'tb_name' -> 'tb_file' -> 'group' -> [sent_idx, ...]
	corpus_map = OrderedDict()
	corpus_map[lang] = OrderedDict()
	# set of domains in all treebanks
	domains = set()

	# iterate over treebanks
	for tb_name in ud_stats[lang]['treebanks']:
		logging.info(f"  Loading treebank '{tb_name}':")
		tb_stats = ud_stats[lang]['treebanks'][tb_name]
		# check if there are files associated with the current treebank
		if 'files' not in tb_stats:
			logging.warning(f"    [Warning] No files associated with treebank '{tb_name}'. Skipped.")
			continue

		# iterate over files
		corpus_map[lang][tb_name] = OrderedDict()
		for tb_file in tb_stats['files']:
			tb_file_path = os.path.join(ud_path, tb_file)
			tb_filename = os.path.basename(tb_file_path)
			# check if file exists on local machine
			if not os.path.exists(tb_file_path):
				logging.warning(f"    [Warning] Could not find treebank file '{tb_file_path}'. Skipped.")
				continue

			# load treebank file
			treebank = UniversalDependenciesTreebank.from_conllu(tb_file_path, name=tb_filename)
			logging.info(f"    Loaded {treebank}.")

			# if cluster level is specified, try parsing the group generator
			cluster_group = copy.deepcopy(group)
			if cluster_group != 'sentences':
				if ('groups' in tb_stats) and (cluster_group in tb_stats['groups']):
					grouper = parse_grouper(tb_stats['groups'][cluster_group])
					if grouper is None:
						logging.warning(f"      [Warning] Could not parse grouper '{tb_stats['groups'][cluster_group]}' for group '{cluster_group}'.")
						# if strict grouping is enforced, but the generator could not be parsed, skip treebank
						if strict_groups:
							logging.warning(f"    [Warning] Could not parse grouper for '{cluster_group}' for treebank '{tb_name}' (strict group enforcement). Skipped treebank.")
							break
						# if group generator could not be parsed, fall back to sentence level clustering
						else:
							logging.warning(f"      [Warning] Could not parse grouper for '{cluster_group}' for treebank '{tb_name}'. Falling back to sentence-level.")
							cluster_group = 'sentences'
				# if treebank has no appropriate grouper and strict grouping is enforced, skip treebank
				elif strict_groups:
					logging.warning(f"    [Warning] Treebank '{tb_name}' lacks the cluster group '{cluster_group}' (strict group enforcement). Skipped treebank.")
					break
				# if treebank has no appropriate grouper, fall back to sentence level grouping
				else:
					logging.warning(f"    [Warning] Treebank '{tb_name}' lacks the cluster group '{cluster_group}'. Falling back to sentence-level.")
					cluster_group = 'sentences'

			# default cluster level is sentence-level
			tb_sentences = treebank.get_sentences()
			corpus_map[lang][tb_name][tb_filename] = OrderedDict()
			if cluster_group == 'sentences':
				# append sentences in this file to the joint corpus
				for sentence in tb_sentences:
					# append list of str words to corpus
					corpus.append(sentence.to_words() if tokenized else sentence.to_text())
					# each sentence is its own group
					corpus_map[lang][tb_name][tb_filename][f'sentence-{sentence.idx}'] = [len(corpus)-1]
			# if cluster-level other than sentences is specified
			else:
				# group sentences
				tb_groups = grouper(tb_sentences)
				logging.info(f"      Split treebank file into {len(tb_groups)} {cluster_group}.")
				# append each group as a concatenated list of sentences to corpus
				for tb_group in sorted(tb_groups):
					# handle ungrouped items
					if tb_group == 'unknown' :
						# if strict grouping is enforced, skip the 'unknown' sentences
						if strict_groups:
							logging.warning(f"      [Warning] {len(tb_groups[tb_group])} sentences are not assigned to any group. Discarded.")
							continue
						# if group is 'unknown' and strict grouping is not enforced, add sentences as stand-alone documents
						else:
							logging.warning(f"      [Warning] {len(tb_groups[tb_group])} sentences are not assigned to any group. Adding as stand-alone groups.")
					# if group is set, init list of sentence indices
					else:
						corpus_map[lang][tb_name][tb_filename][tb_group] = []

					# iterate over sentences in current group
					for tbg_sentence in tb_groups[tb_group]:
						# append sentence to corpus
						corpus.append(tbg_sentence.to_words() if tokenized else tbg_sentence.to_text())
						# for special group 'unknown', add sentence as stand-alone documents
						if tb_group == 'unknown':
							corpus_map[lang][tb_name][tb_filename][f'sentence-{tbg_sentence.idx}'] = [len(corpus) - 1]
						# for all other groups, store sentence indices in relevant group
						else:
							corpus_map[lang][tb_name][tb_filename][tb_group].append(len(corpus) - 1)

			# check if current treebank file contributed documents to the corpus
			if len(corpus_map[lang][tb_name][tb_filename]) < 1:
				logging.warning(f"      [Warning] No {cluster_group} from '{tb_filename}' loaded into corpus. Discarded.")
				del(corpus_map[lang][tb_name][tb_filename])
		# check if current treebank contributed documents to the corpus
		if len(corpus_map[lang][tb_name]) < 1:
			logging.warning(f"  [Warning] No {group} from '{tb_name}' loaded into corpus. Discarded.")
			del(corpus_map[lang][tb_name])
			continue

		domains |= set(tb_stats['genres'])

	return corpus, corpus_map, domains


def reduce_groups(corpus, corpus_map):
	red_corpus = []
	red_corpus_map = copy.deepcopy(corpus_map)

	for lang in corpus_map:
		for tb in corpus_map[lang]:
			for tbf in corpus_map[lang][tb]:
				for grp in corpus_map[lang][tb][tbf]:
					# reduction loop
					red_group = []
					# gather relevant sentences
					for sidx in corpus_map[lang][tb][tbf][grp]:
						red_group.append(corpus[sidx])
					# check if list of str or simply str
					if type(red_group[0]) is str:
						# concatenate texts to one large document separated by spaces
						red_corpus.append(' '.join(red_group))
					else:
						# concatenate all lists of words into one large list
						red_corpus.append(list(itertools.chain(*red_group)))
					# reduce relevant indices in corpus map to just one
					red_corpus_map[lang][tb][tbf][grp] = [len(red_corpus) - 1]

	return red_corpus, red_corpus_map


def load_embeddings(emb_path, lang):
	# in special case of all languages, concatenate all embeddings
	if lang == 'All':
		emb_corpus = merge_embeddings(emb_path)
	# otherwise, only load embeddings for current language
	else:
		emb_corpus = np.load(os.path.join(emb_path, f'{lang.replace(" ", "_")}.npy'))

	return emb_corpus


def merge_embeddings(emb_path):
	emb_corpus = None
	for file_path in os.listdir(emb_path):
		# skip non-numpy files
		if not file_path.endswith('.npy'): continue
		lang = os.path.basename(file_path).replace('_', ' ')
		# load language embeddings
		emb_lang = np.load(os.path.join(emb_path, file_path))
		# concatenate with all embeddings
		if emb_corpus is None:
			emb_corpus = emb_lang
		else:
			emb_corpus = np.concatenate((emb_corpus, emb_lang), axis=0)
	return emb_corpus


def filter_treebanks(tb_filter, corpus_map, corpus, meta=None):
	filt_corpus_map = copy.deepcopy(corpus_map)
	filt_corpus = []
	filt_domains = set()

	filt_indices = []
	for lang in corpus_map:
		for tb in corpus_map[lang]:
			# delete treebanks not matching the filter
			if not re.match(tb_filter, tb):
				del(filt_corpus_map[lang][tb])
				continue
			# collect relevant domains
			if meta: filt_domains |= set(meta[lang]['treebanks'][tb]['domains'])
			# collect relevant indices and rescale corpus map
			for tbf in corpus_map[lang][tb]:
				for grp in corpus_map[lang][tb][tbf]:
					filt_corpus_map[lang][tb][tbf][grp] = []
					for idx in corpus_map[lang][tb][tbf][grp]:
						filt_indices.append(idx)
						filt_corpus.append(corpus[idx])
						filt_corpus_map[lang][tb][tbf][grp].append(len(filt_corpus) - 1)

		if len(filt_corpus_map[lang]) < 1:
			del(filt_corpus_map[lang])

	return filt_indices, filt_corpus_map, filt_corpus, filt_domains


def load_experiment_data(ud_path, meta, lang, group='sentences', strict_groups=False, tokenized=False, emb_path=None, filter_tb=None):
	corpus, corpus_map, lang_domains = load_corpus(
		ud_path, lang, meta,
		group=group, strict_groups=strict_groups,
		tokenized=tokenized
	)
	# reduce corpus if cluster group is not sentence-level
	if group != 'sentences':
		corpus, corpus_map = reduce_groups(corpus, corpus_map)

	# check if corpus contains data
	if len(corpus) < 2:
		logging.warning(
			f"  [Warning] The joint corpus for language '{lang}' is too small for clustering (n={len(corpus)}). Skipped.")
		return False

	# load embeddings
	emb_corpus = []
	if emb_path:
		emb_corpus = load_embeddings(emb_path, lang)
		logging.info(f"  Loaded {emb_corpus.shape[0]} embeddings with dimensionality {emb_corpus.shape[1]}.")
		# check if corpus size and number of embeddings match
		if len(corpus) != emb_corpus.shape[0]:
			logging.warning(f"  [Warning] UD corpus with n={len(corpus)} does not match embeddings with n={emb_corpus.shape[0]}. Skipped language.")
			return False

	# filter corpus
	if filter_tb:
		filt_indices, corpus_map, corpus, lang_domains = filter_treebanks(filter_tb, corpus_map, corpus, meta)
		if emb_path: emb_corpus = emb_corpus[filt_indices, :]

	return corpus, corpus_map, lang_domains, emb_corpus


def load_results(exp_path, ud_path, meta, lang, emb_path=None, filter_tb=None):
	# set up data
	exp_data = load_experiment_data(ud_path, meta, lang, emb_path=emb_path, filter_tb=filter_tb)
	if exp_data == False: return False
	corpus, corpus_map, lang_domains, emb_corpus = exp_data

	# load results
	res_path = os.path.join(exp_path, f'{lang.replace(" ", "_")}.pkl')
	if not os.path.exists(res_path):
		logging.warning(f"  [Warning] Could not find results for '{lang}' at '{res_path}'. Skipped.")
		return False

	with open(res_path, 'rb') as fp:
		results = pickle.load(fp)

	# sanity check if corpus maps align
	if results['corpus_map'] != corpus_map:
		logging.warning(f"  [Warning] Corpus map for evaluation ('{ud_path}') and corpus map from experiment ('{exp_path}') do not align.")
		return False

	corpus_map = results['corpus_map']
	domain_dist = results['domain_dist']
	domains = results['domains'] if 'domains' in results else [f'cluster-{idx}' for idx in range(domain_dist.shape[0])]
	logging.info(f"  Loaded {domain_dist.shape[1]} domains.")

	return domain_dist, corpus, corpus_map, domains, emb_corpus
