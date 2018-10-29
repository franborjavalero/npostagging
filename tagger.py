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
import npostagging.utils
import npostagging.rnn_arch
import npostagging.seq2seq_arch
import npostagging.two_phases_arch

#	python tagger.py corpus_directory arch_desc_file.json model_name input_file output_file

sos = "<sos>"
eos = "<eos>"
pad_word = "pad_word"
unk_word = "unk_word"

parser = argparse.ArgumentParser()
parser.add_argument('corpus_directory', type=str, help='Source directory contains auxilriar dictionaries and matrix files')
parser.add_argument('description_file', type=str, help='JSON file contains features of rnn_arch or rnn_nn_arch')
parser.add_argument('model_name', type=str, help='Model name of trained model to evaluate')
parser.add_argument('input_file', type=str, help='Corpus to disambiguate')	
parser.add_argument('output_file', type=str, help='Disambiguated corpus')	
args = parser.parse_args()

def get_arguments():
	
	if not os.path.isdir(args.corpus_directory):
		print("Error, corpus_directory: {} does not exist".format(args.corpus_directory))
		sys.exit(1)

	files_ = [args.description_file, args.input_file]
	for f in files_:
		if not os.path.exists(f):
			print("Error, file {} does not exist".format(f))
			sys.exit(1)

	directory_model = os.path.join("models", args.model_name)
	if not os.path.isdir(directory_model):
		print("Error, directory_model {} does not exist".format(directory_model))
		sys.exit(1)


	arch_desc = npostagging.utils.load_json(args.description_file)
	arch_desc["model_name"] = args.model_name
	arch_desc["corpus_directory"] = args.corpus_directory
	arch_desc["training"] = False
	arch_desc["input_file"] = args.input_file
	arch_desc["output_file"] = args.output_file
	
	npostagging.utils.check_parameters_model(arch_desc)
	
	return arch_desc

def main():
	
	arch_desc = get_arguments()

	filename_ambiguity_matrix = os.path.join(arch_desc["corpus_directory"], "ambiguity_matrix.npy")
	filename_id_word_to_id_ambiguity = os.path.join(arch_desc["corpus_directory"], "id_word_to_id_ambiguity.json")
	filename_tag_to_id = os.path.join(arch_desc["corpus_directory"], "tag_to_id.json")
	filename_word_to_id = os.path.join(arch_desc["corpus_directory"], "word_to_id.json")

	files = [filename_ambiguity_matrix, filename_id_word_to_id_ambiguity, filename_tag_to_id, filename_word_to_id]
	
	for f in files:
		if not os.path.exists(f):
			print("Error, file {} does not exist".format(f))
			sys.exit(1)
		
	id_word_to_id_ambiguity = npostagging.utils.load_json(filename_id_word_to_id_ambiguity)
	tag_to_id = npostagging.utils.load_json(filename_tag_to_id)
	id_to_tag = npostagging.utils.reverse_dictionary(tag_to_id)
	word_to_id = npostagging.utils.load_json(filename_word_to_id)

	filename_aux_file = os.path.join(arch_desc["corpus_directory"], "processed_raw_input_tagger.txt")
	npostagging.utils.raw_corpus_to_processed_corpus(arch_desc['input_file'], filename_aux_file, word_to_id, id_word_to_id_ambiguity)

	matrix_multihot = npostagging.utils.import_numpy_array(filename_ambiguity_matrix)
	num_ambiguities, num_tags = matrix_multihot.shape
	start_token = tag_to_id[sos]
	end_token = tag_to_id[eos]

	if arch_desc['arch'] == 'rnn':
		  
		  rnn_arch = npostagging.rnn_arch.RnnArchitecture(arch_desc['model_name'], arch_desc['hidden_rnn'], num_ambiguities, num_tags, 
		  	bidirectional=arch_desc["bidirectional_rnn"], unit_type=arch_desc["unit_type"])

		  rnn_arch.forward()

		  predictions = rnn_arch.predict(filename_aux_file, matrix_multihot)

	elif arch_desc['arch'] == 'two_phases':

		rnn_nn_arch = npostagging.two_phases_arch.TwoPhasesArchitecture(arch_desc['model_name'], arch_desc['hidden_rnn'], arch_desc["hidden_nn"], num_ambiguities, 
			num_tags, bidirectional=arch_desc["bidirectional_rnn"], unit_type=arch_desc["unit_type"])

		rnn_nn_arch.forward()

		predictions = rnn_nn_arch.predict(filename_aux_file, matrix_multihot)
	
	else: # seq2seq
		
		seq2seq_arch = npostagging.seq2seq_arch.Seq2SeqArchitecture(arch_desc['model_name'], arch_desc['hidden_encoder'], arch_desc['hidden_decoder'], num_ambiguities, num_tags, 
			start_token, end_token, arch_desc['attention_size'], unit_type=arch_desc['unit_type'], bidirectional_encoder=arch_desc['bidirectional_encoder'], beam_width=arch_desc['beam_width'])
		
		seq2seq_arch.forward()

		predictions = seq2seq_arch.predict(filename_aux_file, matrix_multihot)

	string_predictions = npostagging.utils.predictions_to_text(predictions[0], id_to_tag)

	npostagging.utils.generate_disambiguate_file_prediction(arch_desc['input_file'], string_predictions, arch_desc['output_file'])

if __name__ == "__main__":
	main()