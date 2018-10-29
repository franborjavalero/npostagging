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
import shutil
import argparse
import numpy as np
import npostagging.utils
import npostagging.rnn_arch
import npostagging.seq2seq_arch
import npostagging.two_phases_arch

#	python train.py corpus_directory arch_desc_file.json model_name

sos = "<sos>"
eos = "<eos>"
unk_word = "unk_word"
pad_word = "pad_word"

parser = argparse.ArgumentParser()
parser.add_argument('corpus_directory', type=str, help='Source directory contains X and y training and dev corpus files')
parser.add_argument('description_file', type=str, help='JSON file contains features of rnn_arch or rnn_nn_arch')
parser.add_argument('model_name', type=str, help='Model name, different trained_models could have same architecture')
args = parser.parse_args()

def get_arguments():
	
	if not os.path.isdir(args.corpus_directory):
		print("Error, corpus_directory: {} does not exist".format(args.corpus_directory))
		sys.exit(1)

	if not os.path.exists(args.description_file):
		print("Error, file {} does not exist".format(args.description_file))
		sys.exit(1)

	arch_desc = npostagging.utils.load_json(args.description_file)
	
	arch_desc["model_name"] = args.model_name
	arch_desc["corpus_directory"] = args.corpus_directory
	arch_desc["training"] = True
	
	npostagging.utils.check_parameters_model(arch_desc)

	for m_ in [arch_desc["model_name"], arch_desc["model_name"]+"_aux"]:
		directory_model = os.path.join("models", m_)
		if not os.path.isdir(directory_model):
			os.makedirs(directory_model)
	
	return arch_desc

def main():

	arch_desc = get_arguments()

	filename_X_train = os.path.join(arch_desc["corpus_directory"], "X_train.txt")
	filename_y_ambiguity_train = os.path.join(arch_desc["corpus_directory"], "y_ambiguity_train.txt")
	filename_y_tag_train = os.path.join(arch_desc["corpus_directory"], "y_tag_train.txt")

	filename_X_dev = os.path.join(arch_desc["corpus_directory"], "X_dev.txt")
	filename_y_ambiguity_dev = os.path.join(arch_desc["corpus_directory"], "y_ambiguity_dev.txt")
	filename_y_tag_dev = os.path.join(arch_desc["corpus_directory"], "y_tag_dev.txt")
	
	filename_ambiguity_matrix = os.path.join(arch_desc["corpus_directory"], "ambiguity_matrix.npy")
	filename_id_word_to_id_ambiguity = os.path.join(arch_desc["corpus_directory"], "id_word_to_id_ambiguity.json")
	filename_tag_to_id = os.path.join(arch_desc["corpus_directory"], "tag_to_id.json")
	filename_word_to_id = os.path.join(arch_desc["corpus_directory"], "word_to_id.json")

	files = [filename_ambiguity_matrix, filename_id_word_to_id_ambiguity, filename_tag_to_id, filename_word_to_id, filename_X_train, filename_X_dev, 
	filename_y_tag_dev, filename_y_ambiguity_train, filename_y_ambiguity_dev]

	if arch_desc['arch'] != 'two_phases':
		files = files[:-2]

	for f in files:
		if not os.path.exists(f):
			print("Error, file {} does not exist".format(f))
			sys.exit(1)

	id_word_to_id_ambiguity = npostagging.utils.load_json(filename_id_word_to_id_ambiguity)
	word_to_id = npostagging.utils.load_json(filename_word_to_id)
	tag_to_id = npostagging.utils.load_json(filename_tag_to_id)
	start_token = tag_to_id[sos]
	end_token = tag_to_id[eos]

	matrix_multihot = npostagging.utils.import_numpy_array(filename_ambiguity_matrix)
	num_ambiguities, num_tags = matrix_multihot.shape
	id_unk_token = id_word_to_id_ambiguity[str(word_to_id[unk_word])]
	
	ids_ambiguous = np.array([np.sum(matrix_multihot[idx]) > 1 for idx in range(num_ambiguities)])

	
	if arch_desc['arch'] == 'rnn':
		
		npostagging.rnn_arch.train_rnn_arch(arch_desc, filename_X_train, filename_y_tag_train, filename_X_dev, filename_y_tag_dev, arch_desc['num_epochs'], 
			arch_desc['batch_size'], matrix_multihot, ids_ambiguous, id_unk_token, num_ambiguities, num_tags, max_stopping_step=arch_desc['max_stopping_step'])

	elif arch_desc['arch'] == 'two_phases':

		npostagging.two_phases_arch.train_rnn_nn_arch_phase1(arch_desc, filename_X_train, filename_y_ambiguity_train, filename_X_dev, filename_y_ambiguity_dev, 
			arch_desc['num_epochs'], arch_desc['batch_size'], num_ambiguities, num_tags, max_stopping_step=arch_desc['max_stopping_step'])
		
		npostagging.two_phases_arch.train_rnn_nn_arch_phase2(arch_desc, filename_X_train, filename_y_tag_train, filename_X_dev, filename_y_tag_dev, 
			arch_desc['num_epochs'], arch_desc['batch_size'], matrix_multihot, ids_ambiguous, id_unk_token, num_ambiguities, num_tags, max_stopping_step=arch_desc['max_stopping_step'])
	else:

		npostagging.seq2seq_arch.train_Seq2Seq(arch_desc, filename_X_train, filename_y_tag_train, filename_X_dev, filename_y_tag_dev, arch_desc['num_epochs'], arch_desc['batch_size'],
			matrix_multihot, ids_ambiguous, id_unk_token, start_token, end_token, num_ambiguities, num_tags, beam_width=arch_desc['beam_width'], max_stopping_step=arch_desc['max_stopping_step'])
		
	#	remove folder contains auxiliar checkpoints
	shutil.rmtree(os.path.join("models", arch_desc["model_name"]+"_aux"))
	
if __name__ == "__main__":
	main()
