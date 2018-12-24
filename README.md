# Neural part-of-speech tagging
This is an open source implementation of my neural part-of-speech tagging system, described in my bachelor's thesis:

Francisco de Borja Valero. 2018. **[Neural part-of-speech tagging](https://rua.ua.es/dspace/bitstream/10045/84670/1/Neural_partofspeech_tagging_VALERO_ANTON_FRANCISCO_DE_BORJA.pdf)**.

Requirements
--------
- [Python](https://www.python.org) (tested with 3.6.5)
- [TensorFlow](https://www.tensorflow.org) (tested with 1.10.1)
- [NLTK](https://www.nltk.org) (tested with 3.3.0)

Usage
--------

The following command cleans the original Wall Street Journal (WSJ) dataset:
```
python3 preprocess.py clean_wsj_treebank PATH_RAW_CORPUS --destiny_directory PATH_CLEANED_CORPUS
```

- `PATH_RAW_CORPUS` is the directory of the original WSJ dataset.
- `PATH_CLEANED_CORPUS` is the directory of the preprocessed WSJ dataset.

- Optionally, the sections of the sets can be indicated using the next arguments (below are shown the defaults values):
  - `--sections_train` `0-18`.
  - `--sections_dev` `19-21`.
  - `--sections_test` `22-24`. 

The following command gets ambiguity classes:
```
python3 preprocess.py get_ambiguities PATH_CLEANED_CORPUS
```
- `PATH_CLEANED_CORPUS` is the directory of the preprocessed WSJ dataset in which stores the ambiguity classes.

The following command generates persistence input and desired output files, in order to train and evaluate models/baselines.
```
python3 preprocess.py generate_corpus PATH_CLEANED_CORPUS --set SET
```
- `PATH_CLEANED_CORPUS` is the directory of the preprocessed WSJ dataset in which stores the persistence files.
- `SET`: `train`, `dev` and `test`.

The following command gets an array of occurrences for each baseline:
```
python3 baselines.py get PATH_CLEANED_CORPUS
```
- `PATH_CLEANED_CORPUS` is the directory of the preprocessed WSJ dataset in which stores the array of occurrences for each baseline.
  
The following command evaluates the three proposed baselines:
```
python3 evaluate.py get PATH_CLEANED_CORPUS --set SET
```
- `PATH_CLEANED_CORPUS` is the directory of the preprocessed WSJ dataset.
- `SET`: `train`, `dev` and `test`.
 
The following command trains a neural part-of-speech model:
```
python3 train.py PATH_CLEANED_CORPUS PATH_ARCH_DESC.JSON MODEL_NAME
```

- `PATH_CLEANED_CORPUS` is the directory of the preprocessed WSJ dataset.
- `PATH_ARCH_DESC.JSON` is the file that describes the model like examples located in the folder **[hparams](https://github.com/franborjavalero/npostagging/tree/master/hparams)**.
- `MODEL_NAME` is the neural part-of-speech model name.
  
The following command evaluates a neural part-of-speech model:
```
python3 evaluate.py PATH_CLEANED_CORPUS ARCH_DESC.JSON MODEL_NAME --set SET
```
- `PATH_CLEANED_CORPUS` is the directory of the preprocessed WSJ dataset.
- `PATH_ARCH_DESC.JSON` is the file that describes the model like examples located in the folder **[hparams](https://github.com/franborjavalero/npostagging/tree/master/hparams)**.
- `MODEL_NAME` is the neural part-of-speech model name.
- `SET`: `train`, `dev` and `test`.
  
The following command disambiguates a raw file using a trained neural part-of-speech model:
```
python3 tagger.py PATH_CLEANED_CORPUS ARCH_DESC.JSON MODEL_NAME PATH_INPUT.TXT PATH_OUTPUT.TXT
```

- `PATH_CLEANED_CORPUS` is the directory of the preprocessed WSJ dataset.
- `PATH_ARCH_DESC.JSON` is the file that describes the model like examples located in the folder **[hparams](https://github.com/franborjavalero/npostagging/tree/master/hparams)**.
- `MODEL_NAME` is the neural part-of-speech model name.
- `SET`: `train`, `dev` and `test`.
- `PATH_INPUT.TXT` is the input raw file.
- `PATH_OUTPUT.TXT` is the output disambiguate file.

#### Reproducing results

 - Synthetic dataset:
    ```
    ./preprocess_synthetic.sh
    ./run_baselines_synthetic.sh
    ./train_synthetic.sh
    ./evaluate_synthetic.sh
    ```

 - WSJ dataset:
    ```
    ./preprocess_wsj.sh path_wsj.tgz
    ./run_baselines_wsj.sh
    ./train_wsj.sh
    ./evaluate_wsj.sh
    ```

- `path_wsj.tgz` is the compressed file of the original WSJ dataset.

## License

[GNU General Public License v3.0](LICENSE)
