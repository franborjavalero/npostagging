#!/bin/bash
#
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

python3 train.py data/clean_wsj hparams/blstm64_drop0.5.json blstm64_wsj_drop0.5
python3 train.py data/clean_wsj hparams/blstm128_drop0.5.json blstm128_wsj_drop0.5
python3 train.py data/clean_wsj hparams/blstm256_nn64_drop0.5.json blstm256_nn64_wsj_drop0.5
python3 train.py data/clean_wsj hparams/blstm512_nn64_drop0.5.json blstm512_nn64_wsj_drop0.5
python3 train.py data/clean_wsj hparams/seq64_seq64_attention64_drop0.5.json seq64_seq64_attention64_wsj_drop0.5
python3 train.py data/clean_wsj hparams/bseq64_seq64_attention64_drop0.5.json bseq64_seq64_attention64_wsj_drop0.5

