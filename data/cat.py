#!/usr/bin/python3

import argparse, os, pickle, sys

# local imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from lib.data import *
from lib.utils import *


def parse_arguments():
	arg_parser = argparse.ArgumentParser(description='Universal Dependencies - Concatenated Corpus Creation')
	arg_parser.add_argument('ud_path', help='path to Universal Dependencies directory')
	arg_parser.add_argument('split', help='path to split definition (pickle)')
	arg_parser.add_argument('out_path', help='path to output directory')
	arg_parser.add_argument('-nc', '--no_comments', action='store_true', default=False, help='if set removes comments')
	arg_parser.add_argument('-nm', '--no_multiword', action='store_true', default=False, help='if set resolves multiword tokens into constituent words')
	return arg_parser.parse_args()


def main():
	args = parse_arguments()

	# check if output dir exists
	setup_success = setup_output_directory(args.out_path)
	if not setup_success: return

	# setup logging
	setup_logging(os.path.join(args.out_path, 'cat.log'))

	# load existing split
	with open(args.split, 'rb') as fp:
		split_idcs = pickle.load(fp)
	relevant_idcs = set(split_idcs['train']) | set(split_idcs['dev']) | set(split_idcs['test'])
	ud_filter = UniversalDependenciesIndexFilter(relevant_idcs)
	logging.info(f"Loaded data splits {', '.join([f'{s}: {len(idcs)}' for s, idcs in split_idcs.items()])}.")

	# load Universal Dependencies
	ud = UniversalDependencies.from_directory(args.ud_path, ud_filter=ud_filter, verbose=True)
	domains = ud.get_domains()
	logging.info(f"Loaded {ud} with {len(ud.get_treebanks())} treebanks in {len(domains)} domains ({', '.join(domains)}).")

	# export splits
	for split, idcs in split_idcs.items():
		split_path = os.path.join(args.out_path, f'{split}.conllu')
		logging.info(f"Exporting {split} split with {len(idcs)} sentences to '{split_path}'...")
		with open(split_path, 'w', encoding='utf8') as fp:
			for idx in sorted(idcs):
				if not args.no_comments:
					fp.write(f'# ud_file = {ud.get_treebank_file_of_index(idx)}\n')
					fp.write(f'# ud_index = {idx}\n')
				fp.write(ud[idx].to_conllu(comments=(not args.no_comments), resolve=args.no_multiword))
		logging.info(f"Completed export.")


if __name__ == '__main__':
	main()
