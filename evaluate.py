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
import numpy as np
import npostagging.utils
import npostagging.rnn_arch
import npostagging.seq2seq_arch
import npostagging.two_phases_arch

#	python evaluate.py corpus_directory arch_desc_file.json model_name (--set)

sos = "<sos>"
eos = "<eos>"
unk_word = "unk_word"
pad_word = "pad_word"

parser = argparse.ArgumentParser()
parser.add_argument('corpus_directory', type=str, help='Source directory contains X and y dev and test corpus files')
parser.add_argument('description_file', type=str, help='JSON file contains features of rnn_arch or rnn_nn_arch')
parser.add_argument('model_name', type=str, help='Model name of trained model to evaluate')
parser.add_argument('--set', type=str, default="test", help='Corpus to evaluate, dev or test')	
args = parser.parse_args()

def get_arguments():
	
	if not os.path.isdir(args.corpus_directory):
		print("Error, corpus_directory: {} does not exist".format(args.corpus_directory))
		sys.exit(1)

	if not os.path.exists(args.description_file):
		print("Error, file {} does not exist".format(args.description_file))
		sys.exit(1)

	directory_model = os.path.join("models", args.model_name)
	if not os.path.isdir(directory_model):
		print("Error, directory_model {} does not exist".format(directory_model))
		sys.exit(1)

	if args.set not in ["dev", "test"]:
		print("Error, 'set' argument, expected 'dev' or 'set'")
		sys.exit(1)

	arch_desc = npostagging.utils.load_json(args.description_file)
	arch_desc["model_name"] = args.model_name
	arch_desc["set"] = args.set
	arch_desc["corpus_directory"] = args.corpus_directory
	arch_desc["training"] = False
	npostagging.utils.check_parameters_model(arch_desc)
	
	return arch_desc

def main():
	
	arch_desc = get_arguments()

	#	corpus to evaluate
	filename_X = os.path.join(arch_desc["corpus_directory"], "X_{}.txt".format(arch_desc['set']))
	filename_y_ambiguity = os.path.join(arch_desc["corpus_directory"], "y_ambiguity_{}.txt".format(arch_desc['set']))
	filename_y_tag = os.path.join(arch_desc["corpus_directory"], "y_tag_{}.txt".format(arch_desc['set']))
	#	auxiliar data
	filename_ambiguity_matrix = os.path.join(arch_desc["corpus_directory"], "ambiguity_matrix.npy")
	filename_id_word_to_id_ambiguity = os.path.join(arch_desc["corpus_directory"], "id_word_to_id_ambiguity.json")
	filename_tag_to_id = os.path.join(arch_desc["corpus_directory"], "tag_to_id.json")
	filename_word_to_id = os.path.join(arch_desc["corpus_directory"], "word_to_id.json")

	files = [filename_ambiguity_matrix, filename_id_word_to_id_ambiguity, filename_tag_to_id, filename_word_to_id, filename_X, filename_y_tag, filename_y_tag]

	if arch_desc['arch'] == 'rnn':
		files = files[:-1]

	for f in files:
		if not os.path.exists(f):
			print("Error, file {} does not exist".format(f))
			sys.exit(1)

	id_word_to_id_ambiguity = npostagging.utils.load_json(filename_id_word_to_id_ambiguity)
	word_to_id = npostagging.utils.load_json(filename_word_to_id)

	matrix_multihot = npostagging.utils.import_numpy_array(filename_ambiguity_matrix)
	num_ambiguities, num_tags = matrix_multihot.shape
	
	id_unk_token = id_word_to_id_ambiguity[str(word_to_id[unk_word])]
	
	tag_to_id = npostagging.utils.load_json(filename_tag_to_id)
	start_token = tag_to_id[sos]
	end_token = tag_to_id[eos]

	ids_ambiguous = np.array([np.sum(matrix_multihot[idx]) > 1 for idx in range(num_ambiguities)])

	print("Trained model: {} set: {}".format(arch_desc['model_name'], arch_desc['set']))
	
	if arch_desc['arch'] == 'rnn':
		  
		  rnn_arch = npostagging.rnn_arch.RnnArchitecture(arch_desc['model_name'], arch_desc['hidden_rnn'], num_ambiguities, num_tags, 
		  	bidirectional=arch_desc["bidirectional_rnn"], unit_type=arch_desc["unit_type"])

		  rnn_arch.forward()
		  
		  rnn_arch.evaluate(filename_X, filename_y_tag, matrix_multihot, ids_ambiguous, id_unk_token)

	elif arch_desc['arch'] == 'two_phases':

		rnn_nn_arch = npostagging.two_phases_arch.TwoPhasesArchitecture(arch_desc['model_name'], arch_desc['hidden_rnn'], arch_desc["hidden_nn"], 
			num_ambiguities, num_tags, bidirectional=arch_desc["bidirectional_rnn"], unit_type=arch_desc["unit_type"])

		rnn_nn_arch.forward()

		rnn_nn_arch.evaluate(filename_X, filename_y_ambiguity, filename_y_tag, matrix_multihot, ids_ambiguous, id_unk_token)
	
	else:
		
		seq2seq_arch = npostagging.seq2seq_arch.Seq2SeqArchitecture(arch_desc['model_name'], arch_desc['hidden_encoder'], arch_desc['hidden_decoder'], 
			num_ambiguities, num_tags, start_token, end_token, arch_desc['attention_size'], unit_type=arch_desc['unit_type'], 
			bidirectional_encoder=arch_desc['bidirectional_encoder'], beam_width=arch_desc['beam_width'])
		
		seq2seq_arch.forward()

		seq2seq_arch.evaluate(filename_X, filename_y_tag, matrix_multihot, ids_ambiguous, id_unk_token)

if __name__ == "__main__":
	main()
