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

python3 train.py data/synthetic hparams/lstm8.json lstm8
python3 train.py data/synthetic hparams/blstm8.json blstm8

python3 train.py data/synthetic hparams/lstm8_nn8.json lstm8_nn8
python3 train.py data/synthetic hparams/blstm8_nn8.json blstm8_nn8

python3 train.py data/synthetic hparams/seq8_seq8_attention8.json seq8_seq8_attention8
python3 train.py data/synthetic hparams/bseq8_seq8_attention8.json bseq8_seq8_attention8