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

import numpy as np
import npostagging.utils
from abc import ABC, abstractmethod

sos = "<sos>"
eos = "<eos>"
unk_word = "unk_word"
pad_word = "pad_word"

class Baseline(ABC):

	@abstractmethod
	def is_ambiguous(self, id_):
		pass

	@abstractmethod
	def is_unk(self, id_):
		pass
	
	@abstractmethod
	def predict_tag(self, id_):
		pass

	@abstractmethod
	def export(self, filename):
		pass

	@abstractmethod
	def get_apparitions(self, filename):
		pass

	def get_predictions(self, filename_X):
		text = npostagging.utils.txt_to_string(filename_X)
		tokens = text.split(" ")
		if not tokens[-1]:
			tokens = tokens[:-1]
		tag_predictions = np.array(list(map(self.predict_tag, tokens)))
		ambiguous_mask = np.array(list(map(self.is_ambiguous, tokens)), dtype=bool)
		unk_mask = np.array(list(map(self.is_unk, tokens)), dtype=bool)
		return tag_predictions, ambiguous_mask, unk_mask

	def evaluate(self, filename_X, filename_y):
		#	get predictions
		y_predicted, ambiguous_mask, unk_mask = self.get_predictions(filename_X)
		#	get expected outputs
		text_y = npostagging.utils.txt_to_string(filename_y)
		y_tokens = text_y.split(" ")
		if not y_tokens[-1]:
			y_tokens = y_tokens[:-1]
		y_expected = np.array(list(map(int, y_tokens)))
		total_acc = np.mean(np.equal(y_predicted, y_expected))
		ambiguity_acc = np.mean(np.equal(y_predicted[ambiguous_mask], y_expected[ambiguous_mask]))
		unk_acc = np.mean(np.equal(y_predicted[unk_mask], y_expected[unk_mask]))
		return total_acc, ambiguity_acc, unk_acc

class BaselineTagLevel(Baseline):
	
	def __init__(self, *args):

		if len(args) == 2:			
			self.get_apparitions(args[0])
			self.ambiguity_matrix = npostagging.utils.import_numpy_array(args[1])
			self.id_ambiguity_unk = self.ambiguity_matrix.shape[0]-4 # -1 <eos>, -2 <sos>, -3 id_ambiguity_pad, -4 id_ambiguity_unk
		
		elif len(args) == 3:
			self.array_apparitions_tags = npostagging.utils.import_numpy_array(args[0])
			self.ambiguity_matrix = npostagging.utils.import_numpy_array(args[1])
			self.id_to_tag = npostagging.utils.reverse_dictionary(npostagging.utils.load_json(args[2]))
			self.id_ambiguity_unk = self.ambiguity_matrix.shape[0]-4 # -1 <eos>, -2 <sos>, -3 id_ambiguity_pad, -4 id_ambiguity_unk

	def get_apparitions(self, filename_disambiguate_corpus):
		self.tag_to_id, tags_to_num_app = npostagging.utils.get_tags(filename_disambiguate_corpus)
		npostagging.utils.add_key_to_dict(self.tag_to_id, sos)
		tags_to_num_app[sos] = 0
		npostagging.utils.add_key_to_dict(self.tag_to_id, eos)
		tags_to_num_app[eos] = 0
		npostagging.utils.add_key_to_dict(self.tag_to_id, pad_word)
		tags_to_num_app[pad_word] = 0
		self.array_apparitions_tags = np.zeros(len(self.tag_to_id))
		for tag, id_tag in self.tag_to_id.items():
			self.array_apparitions_tags[id_tag] = tags_to_num_app[tag]

	def predict_tag(self, id_ambiguity_string):
		return int(np.argmax(self.ambiguity_matrix[int(id_ambiguity_string)] * self.array_apparitions_tags))

	def is_ambiguous(self, id_ambiguity_string):
		return np.sum(self.ambiguity_matrix[int(id_ambiguity_string)]) > 1

	def is_unk(self, id_ambiguity_string):
		return int(id_ambiguity_string) == self.id_ambiguity_unk

	def export(self, filename):
		npostagging.utils.export_numpy_array(filename, self.array_apparitions_tags)

class BaselineAmbiguityLevel(Baseline):
	
	def __init__(self, *args, **kwargs):

		if len(args) == 4:

			if 'unk_word' in kwargs:
				self.unk_word = kwargs['unk_word']
			else:
				self.unk_word = unk_word

			self.tag_to_id = npostagging.utils.load_json(args[1])
			self.id_to_tag = npostagging.utils.reverse_dictionary(self.tag_to_id)
			self.word_to_id = npostagging.utils.load_json(args[2])
			self.id_word_to_id_ambiguity = npostagging.utils.load_json(args[3])
			self.get_num_ambiguities()
			self.ambiguity_apparitions = np.zeros((self.num_ambiguities, len(self.tag_to_id)))
			self.id_ambiguity_unk = self.id_word_to_id_ambiguity[str(self.word_to_id[self.unk_word])]
			self.get_apparitions(args[0])
			
		elif len(args) == 2 and len(kwargs) == 0:
			self.ambiguity_apparitions = npostagging.utils.import_numpy_array(args[0])
			self.id_to_tag = npostagging.utils.load_json(args[1])
			self.id_ambiguity_unk = self.ambiguity_apparitions.shape[0]-4 # -1 <eos>, -2 <sos>, -3 id_ambiguity_pad, -4 id_ambiguity_unk
			
	def get_num_ambiguities(self):
		self.num_ambiguities = 0
		for _, id_ambiguity in self.id_word_to_id_ambiguity.items():
			self.num_ambiguities = max(self.num_ambiguities, id_ambiguity)
		self.num_ambiguities += 1 #	ambiguities start by one

	def get_apparitions(self, filename_disambiguate_corpus):
		with open(filename_disambiguate_corpus) as file:
			for line in file:
				tokens = line.strip().split(' ') 
				current_id_tag = self.tag_to_id[tokens[1]]
				self.ambiguity_apparitions[self.id_word_to_id_ambiguity[str(self.word_to_id[tokens[0].lower()])]][current_id_tag] += 1
				# add to unk word all tag apparitions
				self.ambiguity_apparitions[self.id_ambiguity_unk][current_id_tag] += 1

	def predict_tag(self, id_ambiguity):
		return int(np.argmax(self.ambiguity_apparitions[int(id_ambiguity)]))

	def is_ambiguous(self, id_ambiguity):
		return np.sum(self.ambiguity_apparitions[int(id_ambiguity)] != 0) > 1

	def is_unk(self, id_ambiguity_string):
		return int(id_ambiguity_string) == self.id_ambiguity_unk

	def export(self, filename):
		npostagging.utils.export_numpy_array(filename, self.ambiguity_apparitions)

class BaselineWordLevel(Baseline):

	def __init__(self, *args, **kwargs):

		if len(args) == 2:

			if 'unk_word' in kwargs:
				self.unk_word = kwargs['unk_word']
			else:
				self.unk_word = unk_word

			self.tag_to_id = npostagging.utils.load_json(args[0])
			self.id_to_tag = npostagging.utils.reverse_dictionary(self.tag_to_id)
			self.word_to_id = npostagging.utils.load_json(args[1])
			self.id_unk_word = self.word_to_id[self.unk_word]

			if 'disambiguate_corpus' in kwargs:
				
				self.words_apparitions = np.zeros((len(self.word_to_id), len(self.tag_to_id)))
				self.get_apparitions(kwargs['disambiguate_corpus'])
			
			elif 'words_apparitions' in kwargs:
				self.words_apparitions = npostagging.utils.import_numpy_array(kwargs['words_apparitions'])

	def get_apparitions(self, filename_disambiguate_corpus):
		with open(filename_disambiguate_corpus) as file:
			for line in file:
				tokens = line.strip().split(' ')
				current_id_tag = self.tag_to_id[tokens[1]]
				self.words_apparitions[self.word_to_id[tokens[0].lower()]][current_id_tag] += 1
				# add unk word all apparitions
				self.words_apparitions[self.id_unk_word][current_id_tag] += 1

	def predict_tag(self, word):
		if self.is_unk(word):
			return int(np.argmax(self.words_apparitions[self.word_to_id[self.unk_word]]))
		else:	
			return int(np.argmax(self.words_apparitions[self.word_to_id[word]]))

	def is_ambiguous(self, word):
		if word not in self.word_to_id:
			return np.sum(self.words_apparitions[self.word_to_id[self.unk_word]] != 0) > 1
		else:	
			return np.sum(self.words_apparitions[self.word_to_id[word]] != 0) > 1

	def is_unk(self, word):
		return word not in self.word_to_id

	def export(self, filename):
		npostagging.utils.export_numpy_array(filename, self.words_apparitions)