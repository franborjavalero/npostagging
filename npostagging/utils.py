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
import json
import random
import numpy as np
from nltk import word_tokenize

sos = "<sos>"
eos = "<eos>"
unk_word = "unk_word"
pad_word = "pad_word"
# hyperparameters architecture neural model
archs = ['two_phases', 'rnn', "seq2seq"]
unit_types = ['lstm_cpu', 'gru_cpu', 'lstm_gpu', 'gru_gpu']
open_tags = ['NN', 'NNS', 'NNP','NNPS','VB','VBD','VBG','VBN', 'VBP', 'VBZ', 'JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS']

# get_ambiguities_corpus

def get_tags(filename_disambiguate_corpus):
	tag_to_id = {}
	tag_to_num_app = {}
	id_tag = 0
	try:
		with open(filename_disambiguate_corpus) as file:
			for line in file:
				tokens = line.strip().split(' ') # line: word tag
				if tokens[1] not in tag_to_id:
					tag_to_id[tokens[1]] = id_tag
					tag_to_num_app[tokens[1]] = 1
					id_tag += 1
				else:
					tag_to_num_app[tokens[1]] = tag_to_num_app[tokens[1]] + 1
	except IOError:
		print("Error, file: {} not found".format(filename_disambiguate_corpus))

	# dict['tag'] = id_tag; dict['tag'] = num_apparitions
	return tag_to_id, tag_to_num_app

def get_words(filename_disambiguate_corpus):
	word_to_id = {}
	word_to_num_app = {}
	id_word = 0
	try:
		with open(filename_disambiguate_corpus) as file:
			for line in file:
				tokens = line.strip().split(' ')
				if tokens[0].lower() not in word_to_id:
					word_to_id[tokens[0].lower()] = id_word
					word_to_num_app[tokens[0].lower()] = 1
					id_word += 1
				else:
					word_to_num_app[tokens[0].lower()] = word_to_num_app[tokens[0].lower()] + 1
	except IOError:
		print("Error, file: {} not found".format(filename_disambiguate_corpus))
	# dict['word'] = id_word; dict['word'] = num_apparitions
	return word_to_id, word_to_num_app 

def add_key_to_dict(dict_, key):
	# used for unknown and padded
	if key not in dict_:
		dict_[key] = len(dict_)

def get_array_tags(filename_disambiguate_corpus, tags_to_id, threshold = 0.01, filename=None):
	word_to_array_tags = {} # dict['word'] = [array_tags_appears_n_else_0] => counter
	word_multihot = {} # dict['word'] = [array_tags_appears_1_else_0]
	try:
		with open(filename_disambiguate_corpus) as file:
			for line in file:
				tokens = line.strip().split(' ')
				if tokens[0].lower() in word_to_array_tags:
					aux = word_to_array_tags[tokens[0].lower()]
					aux[tags_to_id[tokens[1]]] += 1
					word_to_array_tags[tokens[0].lower()] = aux 
				elif tokens[0].lower() not in word_to_array_tags:
					aux = np.zeros(len(tags_to_id), dtype=np.int32)
					aux[tags_to_id[tokens[1]]] = 1
					word_to_array_tags[tokens[0].lower()] = aux
		id_to_tag = reverse_dictionary(tags_to_id)
		if threshold > 0.0:
			word_tags_threshold = {}
			word_multihot = {}
			for w, arr in word_to_array_tags.items():
				binary_ = np.array(arr > 0, dtype=np.int32)
				threshold_ = np.array(arr/np.sum(arr) > 0.01, dtype=np.int32)
				if not np.array_equal(binary_, threshold_):
					if np.sum(threshold_) > 1:
						aux = []
						for idx, t_ in enumerate(threshold_):
							if t_ == 1:
								aux.append(id_to_tag[idx])				
						word_tags_threshold[w.lower()] = aux
					else:
						word_tags_threshold[w.lower()] = [id_to_tag[np.argmax(threshold_)]]
					word_multihot[w.lower()] = threshold_
				else:
					word_multihot[w.lower()] = binary_
			if filename:
				export_json(filename, word_tags_threshold)
			else:
				print("Error, file: word_to_tag_threshold  not given")
			return word_multihot
		else:
			return word_to_array_tags
	except IOError:
		print("Error, file: {} not found".format(filename_disambiguate_corpus))

def add_special_keys_array_tags(word_to_array_tags, tags_to_id, unk_word=unk_word, sos=sos, eos=eos, pad_word=pad_word, open_tags=open_tags,synthetic=False):
	# unk
	if not synthetic:
		array_tags = np.zeros(len(tags_to_id), dtype=np.int32)
		for tag in open_tags:
			array_tags[tags_to_id[tag]] = 1
		word_to_array_tags[unk_word] = array_tags
	else:
		array_tags = np.zeros(len(tags_to_id), dtype=np.int32)
		array_tags[tags_to_id[eos]] = 1
		array_tags[tags_to_id[sos]] = 1
		array_tags[tags_to_id["."]] = 1
		array_tags[tags_to_id[pad_word]] = 1
		word_to_array_tags[unk_word] = array_tags

	#  sos
	array_tags = np.zeros(len(tags_to_id), dtype=np.int32)
	array_tags[tags_to_id[sos]] = 1
	word_to_array_tags[sos] = array_tags
	# eos
	array_tags = np.zeros(len(tags_to_id), dtype=np.int32)
	array_tags[tags_to_id[eos]] = 1
	word_to_array_tags[eos] = array_tags
	# pad
	array_tags = np.zeros(len(tags_to_id), dtype=np.int32)
	array_tags[tags_to_id[pad_word]] = 1
	word_to_array_tags[pad_word] = array_tags

def get_ambiguities(word_to_id, word_to_array_tags):
	array_tags_string_to_id_ambiguity = {}
	id_word_to_id_ambiguity = {}
	id_ambiguity = 0
	for word, array_tags in word_to_array_tags.items():
		array_tags_string = "".join(map(str, array_tags))
		if array_tags_string not in array_tags_string_to_id_ambiguity:
			array_tags_string_to_id_ambiguity[array_tags_string] = id_ambiguity
			id_word_to_id_ambiguity[word_to_id[word]] = id_ambiguity
			id_ambiguity += 1
		else:
			id_word_to_id_ambiguity[word_to_id[word]] = array_tags_string_to_id_ambiguity[array_tags_string]
	# dict['string_array_tags_appears_1_else_0'] = id_ambiguity, dict[id_word] = id_ambiguity
	return array_tags_string_to_id_ambiguity, id_word_to_id_ambiguity 

def get_id_ambiguity_to_array_tags(array_tags_string_to_id_ambiguity): 
	id_ambiguity_to_array_tags = dict()
	for array_cat_string, id_ambiguity in array_tags_string_to_id_ambiguity.items():
		aux = np.array([int(i) for i in array_cat_string])
		id_ambiguity_to_array_tags[id_ambiguity] = aux
	# dict[id_ambiguity] = np([array_tags_appears_1_else_0])
	return id_ambiguity_to_array_tags 

def get_ambiguity_matrix(id_ambiguity_to_array_tags):
	num_ambiguities = len(id_ambiguity_to_array_tags)
	num_tags = id_ambiguity_to_array_tags[0].shape[0]
	ambiguity_matrix = np.zeros((num_ambiguities, num_tags), dtype=np.float32)
	for id_ambiguity, array_tags in id_ambiguity_to_array_tags.items():
		ambiguity_matrix[id_ambiguity] = array_tags
	return ambiguity_matrix

def string_to_txt(text, filename):
	with open(filename, 'w') as f:
		f.write(text)

def txt_to_string(filename):
	try:
		with open(filename, 'r') as f:
			text = f.read().replace('\n', '')
	except IOError:
		print("Error, file: {} not found".format(filename))
	return text

def export_numpy_array(filename, np_array):
	np.save(filename, np_array)

def import_numpy_array(filename):
	try:
		return np.load(filename)
	except IOError:
		print("Error, file: {} not found".format(filename))

def reverse_dictionary(dict_):
	reverse_dict = {value: key for key, value in dict_.items()}
	return reverse_dict

def load_json(filename):
	try:
		with open(filename) as f:
			data = json.load(f)
	except IOError:
		print("Error, file: {} not found".format(filename))
	return data

def export_json(filename, dictionary):
	with open(filename, 'w') as f:
		json.dump(dictionary, f)

def get_num_sentences(filename):
	count = 0
	try:
		with open(filename) as f:
			for line in f:
				if line != "\n":
					count = count + 1
	except IOError:
		print("Error, file: {} not found".format(filename))
	return count

# generate_corpus

def gen_corpus_set(filename_disambiguate_corpus, word_to_id, id_word_to_id_ambiguity, filename_X, tag_to_id=None,
	filename_y_tag=None, filename_y_ambiguity=None, filename_words=None, unk_word=unk_word, sos=sos, eos=eos):

	sentences_X = []
	current_sentece_X = ""
	text_X = ""
	
	if filename_y_tag:
		sentences_y_tag = []
		current_sentece_y_tag = ""
		text_y_tag = ""

	if filename_y_ambiguity:
		text_y_ambiguity = ""

	if filename_words:
		sentences_words = []
		current_sentece_words = ""
		text_words = ""
	try:
		with open(filename_disambiguate_corpus) as file:
			for line in file:
				tokens = line.strip().split(' ')

				# filename_X, ambiguity
				if tokens[0].lower() in word_to_id:
					current_sentece_X += str(id_word_to_id_ambiguity[str(word_to_id[tokens[0].lower()])]) + " "
				else:
					current_sentece_X += str(id_word_to_id_ambiguity[str(word_to_id[unk_word])]) + " "

				# filename_y_tag, tag
				if filename_y_tag:
					current_sentece_y_tag += str(tag_to_id[tokens[1]]) + " "

				if filename_words:
					current_sentece_words += tokens[0].lower() + " "
				
				if tokens[1].lower() == ".":	# end sentence
					sentences_X.append(current_sentece_X)
					current_sentece_X = ""
					if filename_y_tag:
						sentences_y_tag.append(current_sentece_y_tag)
						current_sentece_y_tag = ""
					if filename_words:
						sentences_words.append(current_sentece_words)
						current_sentece_words = ""
	except IOError:
		print("Error, file: {} not found".format(filename_disambiguate_corpus))
	
	for id_sentece in range(len(sentences_X)):
		
		text_X += sentences_X[id_sentece] + "\n"
				
		if filename_y_tag:
			text_y_tag += sentences_y_tag[id_sentece] + "\n"

		if filename_words:
			text_words += sentences_words[id_sentece] + "\n"
		
		if filename_y_ambiguity:
			sentence_X_current = sentences_X[id_sentece].split(" ")
			aux = sentence_X_current[1:]
			text_y_ambiguity += " ".join(aux) + str(id_word_to_id_ambiguity[str(word_to_id[eos])]) + "\n"
	
	string_to_txt(text_X, filename_X)
	
	if filename_y_tag:
		string_to_txt(text_y_tag, filename_y_tag)

	if filename_words:
		string_to_txt(text_words, filename_words)

	if filename_y_ambiguity:
		string_to_txt(text_y_ambiguity, filename_y_ambiguity)

# clean_wsj_treebank

def clean_disambiguate_corpus(filename_disambiguate_corpus, filename_new_disambiguate_corpus, word_to_tag_threshold):
	text = ""
	current_sentence = ""
	ignore_sentence = False
	nline = 0
	try:
		with open(filename_disambiguate_corpus) as file:
			
			for line in file:
				token = line.strip().split(' ')
				if token[0].lower() in word_to_tag_threshold and token[1] not in word_to_tag_threshold[token[0].lower()]:
					if len(word_to_tag_threshold[token[0].lower()]) > 1:
						ignore_sentence = True
					else:
						current_sentence += token[0] + " " + word_to_tag_threshold[token[0].lower()][0] + "\n"	
				else:
					current_sentence += token[0] + " " + token[1] + "\n"
				# check end of sentence
				if token[1] is '.':
					nline += 1
					if ignore_sentence:
						current_sentence = ""
						ignore_sentence = False
					else:
						text += current_sentence
						current_sentence = ""
	except IOError:
		print("Error, file: {} not found".format(filename_disambiguate_corpus))
	string_to_txt(text, filename_new_disambiguate_corpus)

def clean_file_wsj(filename_wsj, filename_wsj_clean, train=False, list_ambiguous_words=[]):
		
	text = ""
	current_sentence = ""
	ignore_sentence = False
	last_tag = ""
	
	try:
		with open(filename_wsj) as file:
			
			for line in file:
				# example of expected line: word/tag or [word/tag]
				if '/' in line:
					line = line.strip().split(' ')
					if '[' in line:
						line.remove('[')
					if ']' in line:
						line.remove(']')
					
					for token in line:

						if '\\/' in token:
							aux = token.split('\\/') # consecutive words with same tag: word0\/word1\/word2\/..\tag
							aux2 = aux[-1].split('/')
							if len(aux2) > 1:
								for aux_ in aux[:-1]:
									current_sentence += aux_ + " " + aux2[-1] + "\n"
								current_sentence += aux2[0] + " " + aux2[1] + "\n"
								last_tag = aux2[1]
							else:
								current_sentence += aux[0] + " " + aux[1] + "\n"	# common case like "word\tag"
								last_tag = aux[1]
						
						elif '/' in token:
							aux = token.split('/')					
							if "|" in aux[1]:	# ambiguous word: word/tag0|tag1|...
								ignore_sentence = True
							
							elif "&" not in aux[1]: # expected case
								current_sentence += aux[0] + " " + aux[1] + "\n"
								last_tag = aux[1]
							
							else:	# uncommon cases
								ignore_sentence = True
						
						# check end of sentence
						if last_tag is '.':
							if ignore_sentence:
								current_sentence = ""
								ignore_sentence = False
							else:
								text += current_sentence
								current_sentence = ""
	except IOError:
		print("Error, file: {} not found".format(filename_wsj_clean))

	string_to_txt(text, filename_wsj_clean)

def clean_directory_wsj(source_directory, destiny_directory, start_section, end_section, train=False):
	source_directory = source_directory + "/"
	destiny_directory = destiny_directory + "/"
	list_ambiguous_words = []

	if not os.path.exists(destiny_directory):
		os.makedirs(destiny_directory)

	for path, _, files in os.walk(source_directory):
		folder = path.split("/")[-1]
		current_section = int(folder) if len(folder) > 0 else -1 # avoid empty folder
		if current_section >= start_section and current_section <= end_section:
			if not os.path.exists(os.path.join(destiny_directory, folder)):
				os.makedirs(os.path.join(destiny_directory, folder))
			for file_name in files:
				if train:
					clean_file_wsj(os.path.join(path, file_name), os.path.join(destiny_directory, folder, file_name), train=train, list_ambiguous_words=list_ambiguous_words)
				else:
					clean_file_wsj(os.path.join(path, file_name), os.path.join(destiny_directory, folder, file_name), train=train)
	if train:
		return list_ambiguous_words

def build_disambiguate_corpus_wsj(source_directory, filename_disambiguate_corpus, start_section, end_section):
	source_directory = source_directory + "/"
	output = ""
	for path, _, files in os.walk(source_directory):
		folder = path.split("/")[-1]
		current_section = int(folder) if len(folder) > 0 else -1 # avoid empty folder
		if current_section >= start_section and current_section <= end_section:
			for file_name in files:
				f = open(os.path.join(path, file_name), 'r')
				current_file_text = f.read()
				output += current_file_text
	string_to_txt(output, filename_disambiguate_corpus)

# tagger prediction

def raw_corpus_to_processed_corpus(filename_raw_corpus, filename_processed_corpus, word_to_id, id_word_to_id_ambiguity, unk_word="unk_word"):
	text = txt_to_string(filename_raw_corpus)
	tokens = word_tokenize(text)
	
	ambiguities_corpus = ' '.join(str(id_word_to_id_ambiguity[str(word_to_id[word.lower()])]) if word.lower() 
		in word_to_id else str(id_word_to_id_ambiguity[str(word_to_id[unk_word])]) for word in tokens)
	
	processed_corpus = ' {}\n'.format(str(id_word_to_id_ambiguity[str(word_to_id["."])])).join(sentence.strip() 
		for sentence in ambiguities_corpus.split(str(id_word_to_id_ambiguity[str(word_to_id["."])])))
	
	string_to_txt(processed_corpus, filename_processed_corpus)

def predictions_to_text(predictions, id_to_tag):
	text = ' '.join(id_to_tag[id_tag] for id_tag in predictions)
	return text

def generate_disambiguate_file_prediction(raw_corpus, string_predictions, filename_output):
	text_raw_corpus = txt_to_string(raw_corpus)
	words_raw_corpus = word_tokenize(text_raw_corpus)
	tags_predictions = string_predictions.split(" ")
	output = ""
	for word, tag in zip(words_raw_corpus, tags_predictions):
		output += word + " " + tag + "\n"
	string_to_txt(output, filename_output)

#	arch description file json

def check_parameters_model(arch_desc):
	
	if 'arch' not in arch_desc or arch_desc['arch'] not in archs:
		print("Error, 'arch' argument, expected {}".format(", ".join(archs)))
		sys.exit(1)

	if 'unit_type' not in arch_desc or arch_desc['unit_type'] not in unit_types:
		print("Error, 'unit_type' argument, expected {}".format(", ".join(unit_types)))
		sys.exit(1)

	if arch_desc['training']:
		
		if 'num_epochs' not in arch_desc:
			print("Error, 'num_epochs' argument does not exist")
			sys.exit(1)

		if 'batch_size' not in arch_desc:
			print("Error, 'batch_size' argument does not exist")
			sys.exit(1)
			
	else:
		arch_desc['training'] = False
	
	if arch_desc['arch'] == 'two_phases' or arch_desc['arch'] == 'rnn':
		
		if 'hidden_rnn' not in arch_desc:
			print("Error, 'hidden_rnn' argument does not exist")
			sys.exit(1)
		
		if 'dropout_rnn' not in arch_desc:
			arch_desc['dropout_rnn'] = []
		
		else:
			if len(arch_desc['dropout_rnn']) != len(arch_desc['hidden_rnn']):
				print("Error, 'hidden_rnn' and 'dropout_rnn' have not same number of layers")
				sys.exit(1)
		
		if 'bidirectional_rnn' not in arch_desc:
			arch_desc['bidirectional_rnn'] = False
		
		if arch_desc['arch'] == 'two_phases':
		
			if 'hidden_nn' not in arch_desc:
				print("Error, 'hidden_nn' argument does not exist")
				sys.exit(1)

			if 'dropout_nn' not in arch_desc:
				arch_desc['dropout_nn'] = []
			else:
				if len(arch_desc['dropout_nn']) != len(arch_desc['hidden_nn']):
					print("Error, 'hidden_nn' and 'dropout_nn' have not same number of layers")
					sys.exit(1)
	
	else: # seq2seq
			
		# encoder

		if 'bidirectional_encoder' not in arch_desc:
			arch_desc['bidirectional_encoder'] = False

		if 'hidden_encoder' not in arch_desc:
			print("Error, 'hidden_encoder' argument does not exist")
			sys.exit(1)
					
		if 'dropout_encoder' not in arch_desc:
			arch_desc['dropout_encoder'] = []
		else:
			if len(arch_desc['dropout_encoder']) != len(arch_desc['hidden_encoder']):
				print("Error, 'hidden_encoder' and 'dropout_encoder' have not same number of layers")
				sys.exit(1)
		
		# decoder
		
		if 'hidden_decoder' not in arch_desc:
			print("Error, 'hidden_decoder' argument does not exist")
			sys.exit(1)
		
		if 'dropout_decoder' not in arch_desc:
			arch_desc['dropout_decoder'] = []
		else:
			if len(arch_desc['dropout_decoder']) != len(arch_desc['hidden_decoder']):
				print("Error, 'hidden_decoder' and 'dropout_decoder' have not same number of layers")
				sys.exit(1)
		
		if 'attention_size' not in arch_desc:
			print("Error, 'attention_size' argument does not exist")
			sys.exit(1)
		
		if 'beam_width' not in arch_desc:
			arch_desc['beam_width'] = 1
			

	if 'learning_rate' not in arch_desc:
		arch_desc['learning_rate'] = 0.003

	if 'seed' not in arch_desc:
		arch_desc['seed'] = 8

	if 'max_grad_norm' not in arch_desc:
		arch_desc['max_grad_norm'] = 5
	
	if 'max_stopping_step' not in arch_desc:
		arch_desc['max_stopping_step'] = 5