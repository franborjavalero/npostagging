# Copyright (C) 2018 Francisco de Borja Valero <franborjavalero@gmail.com>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import os
import sys
import argparse
import npostagging.utils

sos = "<sos>"
eos = "<eos>"
threshold = 0.01
unk_word = "unk_word"
pad_word = "pad_word"
open_tags = ['NN', 'NNS', 'NNP','NNPS','VB','VBD','VBG','VBN', 'VBP', 'VBZ', 'JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS']

#	python preprocess.py clean_wsj_treebank source_directory --destiny_directory (--sections_train --sections_dev --sections_test)
#	python preprocess.py get_ambiguities source_directory
#	python preprocess.py generate_corpus source_directory --set

parser = argparse.ArgumentParser()
parser.add_argument('mode', type=str, help='Modes: clean_wsj_treebank, get_ambiguities or generate_corpus')
parser.add_argument('source_directory', type=str, help='Directory original WSJ Treebank')
parser.add_argument('--destiny_directory', type=str, help='Directory where cleaned WSJ Treebank will be saved')
parser.add_argument('--sections_train', default="0-18",type=str, help='Sections original WSJ Treebank form part of train set, section_source-section_destiny')
parser.add_argument('--sections_dev', default="19-21", type=str, help='Sections original WSJ Treebank form part of development set, section_source-section_destiny')
parser.add_argument('--sections_test', default="22-24", type=str, help='Sections original WSJ Treebank form part of test set, section_source-section_destiny')
parser.add_argument('--set', type=str, help='Type set of disambiguate corpus generated: train, dev or test')
parser.add_argument('--synthetic', help='Preprocessing for synthetic data', action='store_true')
args = parser.parse_args()

def get_arguments():
	
	modes = ["clean_wsj_treebank", "get_ambiguities", "generate_corpus"]
	get_ambiguities_corpus_files = ["train_disambiguate_raw.txt"]
	corpus_sections = ["train", "dev", "test"]
	generate_corpus_files = ["word_to_id.json", "id_word_to_id_ambiguity.json", "tag_to_id.json"]

	if args.mode not in modes:
		print("Error argument mode, expected {}".format(", ".join(modes)))
		sys.exit(1)
	
	if not os.path.isdir(args.source_directory):
		print("Error, argument source_directory: {} does not exist".format(args.source_directory))
		sys.exit(1)
	
	if args.mode == "clean_wsj_treebank":

		if not args.destiny_directory:
			print("Error, expected destiny_directory argument in clean_wsj_treebank mode")
			sys.exit(1)

		if not os.path.isdir(args.destiny_directory):
			os.makedirs(args.destiny_directory)

		#	optional arguments
		if args.sections_train or args.sections_dev or args.sections_test:
			if not args.sections_train or not args.sections_dev or not args.sections_test:
				print("Error, expected three arguments sections: sections_train, sections_dev sections_test in clean_wsj_treebank")
				sys.exit(1)
	
	elif args.mode == "get_ambiguities":
		
		for file in get_ambiguities_corpus_files:
			file_path = os.path.join(args.source_directory, file)
			if not os.path.exists(file_path):
				print("Error, file {} does not exist".format(file_path))
				sys.exit(1)

	else:	#	generate_corpus

		if not args.set:
			print("Error, expected set argument in generate_corpus mode")
			sys.exit(1)

		if args.set not in corpus_sections:
			print("Error, expected set argument: {}".format(", ".join(corpus_sections)))
			sys.exit(1)
		else:
			if args.set == "train":
				generate_corpus_files = generate_corpus_files[:-1]
			for file in generate_corpus_files:
				file_path = os.path.join(args.source_directory, file)
				if not os.path.exists(file_path):
					print("Error,  file {} does not exist".format(file_path))
					sys.exit(1)

	return args

def main():
	
	arguments = get_arguments()

	if arguments.mode == "clean_wsj_treebank":
		
		# generate_raw_disambiguate_corpus 
		filename_train_disambiguate_raw = os.path.join(arguments.destiny_directory, "train_disambiguate_raw.txt")
		filename_dev_disambiguate_raw = os.path.join(arguments.destiny_directory, "dev_disambiguate_raw.txt")
		filename_test_disambiguate_raw = os.path.join(arguments.destiny_directory, "test_disambiguate_raw.txt")
		
		train_sections = arguments.sections_train.split("-")
		npostagging.utils.clean_directory_wsj(arguments.source_directory, arguments.destiny_directory, int(train_sections[0]), int(train_sections[1]))
		npostagging.utils.build_disambiguate_corpus_wsj(arguments.destiny_directory, filename_train_disambiguate_raw, int(train_sections[0]), int(train_sections[1]))

		dev_sections = arguments.sections_dev.split("-")
		npostagging.utils.clean_directory_wsj(arguments.source_directory, arguments.destiny_directory, int(dev_sections[0]), int(dev_sections[1]))
		npostagging.utils.build_disambiguate_corpus_wsj(arguments.destiny_directory, filename_dev_disambiguate_raw, int(dev_sections[0]), int(dev_sections[1]))

		test_sections = arguments.sections_test.split("-")
		npostagging.utils.clean_directory_wsj(arguments.source_directory, arguments.destiny_directory, int(test_sections[0]), int(test_sections[1]))
		npostagging.utils.build_disambiguate_corpus_wsj(arguments.destiny_directory, filename_test_disambiguate_raw, int(test_sections[0]), int(test_sections[1]))

	elif arguments.mode == "get_ambiguities":

		filename_train_disambiguate_raw = os.path.join(arguments.source_directory, "train_disambiguate_raw.txt")
		filename_tag_to_id = os.path.join(arguments.source_directory, "tag_to_id.json")
		filename_word_to_id = os.path.join(arguments.source_directory, "word_to_id.json")
		filename_id_word_to_id_ambiguity = os.path.join(arguments.source_directory, "id_word_to_id_ambiguity.json")
		filename_ambiguity_matrix = os.path.join(arguments.source_directory, "ambiguity_matrix.npy")
		filename_word_to_tag_threshold = os.path.join(arguments.source_directory, "word_to_tag_threshold.json")

		#	get dictionary_tags_to_id
		tag_to_id, _ = npostagging.utils.get_tags(filename_train_disambiguate_raw)
		npostagging.utils.add_key_to_dict(tag_to_id, sos)
		npostagging.utils.add_key_to_dict(tag_to_id, eos)
		npostagging.utils.add_key_to_dict(tag_to_id, pad_word)
		npostagging.utils.export_json(filename_tag_to_id, tag_to_id)

		#	get dictionary_word_to_id
		word_to_id, _ = npostagging.utils.get_words(filename_train_disambiguate_raw)
		npostagging.utils.add_key_to_dict(word_to_id, unk_word)
		npostagging.utils.add_key_to_dict(word_to_id, sos)
		npostagging.utils.add_key_to_dict(word_to_id, eos)
		npostagging.utils.add_key_to_dict(word_to_id, pad_word)
		npostagging.utils.export_json(filename_word_to_id, word_to_id)

		#	get dictionary_id_word_to_ambiguity
		word_multihot = npostagging.utils.get_array_tags(filename_train_disambiguate_raw, tag_to_id, threshold=threshold, filename=filename_word_to_tag_threshold)
		npostagging.utils.add_special_keys_array_tags(word_multihot, tag_to_id, unk_word=unk_word, sos=sos, eos=eos, pad_word=pad_word, open_tags=open_tags, synthetic=arguments.synthetic)
		array_tags_string_to_id_ambiguity, id_word_to_id_ambiguity = npostagging.utils.get_ambiguities(word_to_id, word_multihot)
		npostagging.utils.export_json(filename_id_word_to_id_ambiguity, id_word_to_id_ambiguity)

		#	get ambiguity_matrix
		id_ambiguity_to_array_tags = npostagging.utils.get_id_ambiguity_to_array_tags(array_tags_string_to_id_ambiguity)
		ambiguity_matrix = npostagging.utils.get_ambiguity_matrix(id_ambiguity_to_array_tags)
		npostagging.utils.export_numpy_array(filename_ambiguity_matrix, ambiguity_matrix)

		# generate_clean_disambiguate_corpus: where all tags belongs to one ambiguity class have a frecuency per word/token higher than 1%
		word_to_tag_threshold = npostagging.utils.load_json(filename_word_to_tag_threshold)
		filename_dev_disambiguate_raw = os.path.join(arguments.source_directory, "dev_disambiguate_raw.txt")
		filename_test_disambiguate_raw = os.path.join(arguments.source_directory, "test_disambiguate_raw.txt")
		filename_train_disambiguate = os.path.join(arguments.source_directory, "train_disambiguate.txt")
		filename_dev_disambiguate = os.path.join(arguments.source_directory, "dev_disambiguate.txt")
		filename_test_disambiguate = os.path.join(arguments.source_directory, "test_disambiguate.txt")
		
		npostagging.utils.clean_disambiguate_corpus(filename_train_disambiguate_raw, filename_train_disambiguate, word_to_tag_threshold)
		npostagging.utils.clean_disambiguate_corpus(filename_dev_disambiguate_raw, filename_dev_disambiguate, word_to_tag_threshold)
		npostagging.utils.clean_disambiguate_corpus(filename_test_disambiguate_raw, filename_test_disambiguate, word_to_tag_threshold)

	else: 
		
		#	generate_corpus

		filename_disambiguate = os.path.join(arguments.source_directory, "{}_disambiguate.txt".format(arguments.set))
		filename_word_to_id = os.path.join(arguments.source_directory, "word_to_id.json")
		filename_id_word_to_id_ambiguity = os.path.join(arguments.source_directory, "id_word_to_id_ambiguity.json")
		filename_words = os.path.join(arguments.source_directory, "{}_words.txt".format(arguments.set))
		filename_X = os.path.join(arguments.source_directory,"X_{}.txt".format(arguments.set))
		filename_y_ambiguity = os.path.join(arguments.source_directory,"y_ambiguity_{}.txt".format(arguments.set))

		word_to_id = npostagging.utils.load_json(filename_word_to_id)
		id_word_to_id_ambiguity = npostagging.utils.load_json(filename_id_word_to_id_ambiguity)
					
		filename_tag_to_id = os.path.join(arguments.source_directory,"tag_to_id.json")
		filename_y_tag = os.path.join(arguments.source_directory,"y_tag_{}.txt".format(arguments.set))
		tag_to_id = npostagging.utils.load_json(filename_tag_to_id)
		
		npostagging.utils.gen_corpus_set(filename_disambiguate, word_to_id, id_word_to_id_ambiguity, filename_X, tag_to_id=tag_to_id, filename_y_tag=filename_y_tag, 
			filename_y_ambiguity=filename_y_ambiguity, filename_words=filename_words)


if __name__ == "__main__":
	main()