#!/bin/bash
#
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

mkdir -p data
tar zxvf $1 --directory data/
python3 preprocess.py clean_wsj_treebank data/wsj/ --destiny_directory data/clean_wsj/
python3 preprocess.py get_ambiguities data/clean_wsj/
python3 preprocess.py generate_corpus data/clean_wsj/ --set train
python3 preprocess.py generate_corpus data/clean_wsj/ --set dev
python3 preprocess.py generate_corpus data/clean_wsj/ --set test