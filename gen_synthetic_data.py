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
from npostagging.utils import string_to_txt
from random import randint, seed

parser = argparse.ArgumentParser()
parser.add_argument('destiny_directory', type=str, help='Directory to save synthetic data')
args = parser.parse_args()

#python gen_synthetic_data.py data/synthetic/

X = [
	"a4 a0 a2 a0 a1 a3 a3 .",
	"a4 a0 a2 a0 a1 a2 a3 .",
	"a4 a1 a2 a0 a1 a3 a3 .",
	"a4 a1 a2 a0 a1 a2 a3 .",
	]

y = [
	"c5 c0 c2 c2 c4 c3 c0 .",
	"c5 c0 c2 c2 c1 c2 c0 .",
	"c5 c1 c3 c2 c4 c3 c0 .",
	"c5 c1 c3 c2 c1 c2 c0 .",
	]

def create_sequences_automata(num_sequences):
	X_ = []
	y_ = []
	num_possible_chains = len(X)
	aux_X = ""
	aux_y = ""
	for _ in range(num_sequences):
		i = randint(1, num_possible_chains)-1
		aux_X = X[i]
		aux_y = y[i]
		X_.append(aux_X)
		y_.append(aux_y)
	return X_, y_

def create_disambiguate_file(corpus_X, corpus_y, filename):
	text = "" 
	for X_, y_ in zip(corpus_X, corpus_y):
		X_ = X_.strip().split(' ')
		y_ = y_.strip().split(' ')
		for a, b in  zip(X_, y_):
			text += a + " " + b + "\n"
	string_to_txt(text[:-1], filename) # erase last end line

def main():

	if not os.path.isdir(args.destiny_directory):
		os.makedirs(args.destiny_directory)
	
	# train 
	train_filename = os.path.join(args.destiny_directory, "train_disambiguate_raw.txt") 
	seed(2)
	corpus_X, corpus_y = create_sequences_automata(20000)
	create_disambiguate_file(corpus_X, corpus_y, train_filename)

	# dev
	dev_filename = os.path.join(args.destiny_directory, "dev_disambiguate_raw.txt") 
	seed(4)
	corpus_X, corpus_y = create_sequences_automata(200)
	create_disambiguate_file(corpus_X, corpus_y, dev_filename)
	
	# test 
	test_filename = os.path.join(args.destiny_directory, "test_disambiguate_raw.txt") 
	seed(6)
	corpus_X, corpus_y = create_sequences_automata(200)
	create_disambiguate_file(corpus_X, corpus_y, test_filename)

if __name__ == "__main__":
	main()