# Copyright (C) 2018  Fran-Borja Valero <franborjavalero@gmail.com>
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
import npostagging.baselines

#	python baselines.py get source_directory
#	python baselines.py evaluate source_directory (--set)

parser = argparse.ArgumentParser()
parser.add_argument('mode', type=str, help='Modes: get,  or evaluate')
parser.add_argument('source_directory', type=str, help='Directory source datasets')
parser.add_argument('--set', default="test", type=str, help='Set to evaluate dev or test')
args = parser.parse_args()

def get_arguments():

	get_mode_files = ["train_disambiguate.txt", "ambiguity_matrix.npy", "tag_to_id.json", "word_to_id.json", "id_word_to_id_ambiguity.json"]
	evaluate_mode_files = ["baseline_tag_level.npy", "baseline_ambiguity_level.npy", "baseline_word_level.npy"]
	evaluate_mode_files = ["baseline_tag_level.npy"]
	section = ["dev", "test"]
	modes = ["get", "evaluate"]

	if args.mode not in modes:
		print("Error, expected mode {}".format(", ".join(section)))
		sys.exit(1)

	if not os.path.isdir(args.source_directory):
		print("Error, source_directory {} does not exist".format(args.source_directory))
		sys.exit(1)

	if args.mode == "get":
				
		for file in get_mode_files:
			path_file = os.path.join(args.source_directory, file)
			if not os.path.exists(path_file):
				print("Error, file {} does not exist".format(path_file))
				sys.exit(1)

	else: #	evaluate

		for file in evaluate_mode_files:
			path_file = os.path.join(args.source_directory, file)
			if not os.path.exists(os.path.join(args.source_directory, file)):
				print("Error, file {} does not exist".format(path_file))
				sys.exit(1)
		
		if args.set not in section:
			print("Error, expected argument {}".format(", ".join(section)))
			sys.exit(1)

	return args

def main():
	
	arguments = get_arguments()

	filename_model_tag_level = os.path.join(arguments.source_directory, "baseline_tag_level.npy")
	filename_model_ambiguity_level = os.path.join(arguments.source_directory, "baseline_ambiguity_level.npy")
	filename_model_word_level = os.path.join(arguments.source_directory, "baseline_word_level.npy")

	filename_ambiguity_matrix = os.path.join(arguments.source_directory, "ambiguity_matrix.npy")
	filename_tag_to_id = os.path.join(arguments.source_directory, "tag_to_id.json")
	filename_word_to_id = os.path.join(arguments.source_directory, "word_to_id.json")

	if args.mode == "get":

		filename_train = os.path.join(arguments.source_directory, "train_disambiguate.txt")
		filename_id_word_to_id_ambiguity = os.path.join(arguments.source_directory, "id_word_to_id_ambiguity.json")

		tag_level = npostagging.baselines.BaselineTagLevel(filename_train, filename_ambiguity_matrix)
		ambiguity_level = npostagging.baselines.BaselineAmbiguityLevel(filename_train, filename_tag_to_id, filename_word_to_id, filename_id_word_to_id_ambiguity)
		word_level = npostagging.baselines.BaselineWordLevel(filename_tag_to_id, filename_word_to_id, disambiguate_corpus=filename_train)
		
		tag_level.export(filename_model_tag_level)
		ambiguity_level.export(filename_model_ambiguity_level)
		word_level.export(filename_model_word_level)

	else:
		
		models_name = ["tag", "ambiguity", "word"]
		results = []

		filename_X_ambiguity = os.path.join(arguments.source_directory, "X_{}.txt".format(arguments.set))
		filename_X_words = os.path.join(arguments.source_directory, "{}_words.txt".format(arguments.set))
		filename_y = os.path.join(arguments.source_directory, "y_tag_{}.txt".format(arguments.set))
		
		tag_level = npostagging.baselines.BaselineTagLevel(filename_model_tag_level, filename_ambiguity_matrix, filename_tag_to_id)
		total_acc, ambiguity_acc, unk_acc = tag_level.evaluate(filename_X_ambiguity, filename_y)
		results.append((total_acc, ambiguity_acc, unk_acc))

		ambiguity_level = npostagging.baselines.BaselineAmbiguityLevel(filename_model_ambiguity_level, filename_tag_to_id)
		total_acc, ambiguity_acc, unk_acc = ambiguity_level.evaluate(filename_X_ambiguity, filename_y)	
		results.append((total_acc, ambiguity_acc, unk_acc))

		word_level = npostagging.baselines.BaselineWordLevel(filename_tag_to_id, filename_word_to_id, words_apparitions=filename_model_word_level)
		total_acc, ambiguity_acc, unk_acc = word_level.evaluate(filename_X_words, filename_y)
		results.append((total_acc, ambiguity_acc, unk_acc))

		print("Results {} set".format(arguments.set))
		for model, result in zip(models_name, results):
			print("Baseline: {} level - total_acc: {:.3f} ambiguity_acc: {:.3f} unk_acc: {:.3f}".format(model, result[0], result[1], result[2]))

if __name__ == "__main__":
	main()