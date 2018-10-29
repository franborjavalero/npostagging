# NPoStagging: Neural Part-of-speech tagging
This is an open source implementation of our neural part-of-speech tagging system, described in my bachelor's thesis.  

Requirements
--------
- [Python](https://www.python.org) (tested wtih 3.6.5)
- [TensorFlow](https://www.tensorflow.org) (tested with 1.10.1)
- [NLTK](https://www.nltk.org) (tested with 3.3.0, only used word_tokenize)

Usage
--------

The following command cleans original Wall Street Journal dataset and save it in a new directory:
```
python3 preprocess.py clean_wsj_treebank PATH_RAW_CORPUS --destiny_directory PATH_CLEANED_CORPUS
```
Optionally, the split sections can be indicated using `--sections_train ID_START-ID_END`, the same with `--sections_dev` and `--sections_test`. By the default their respective values are `0-18`, `19-21` and `22-24`. 

The following command gets ambiguity classes from `PATH_CLEANED_CORPUS`:
```
python3 preprocess.py get_ambiguities PATH_CLEANED_CORPUS
```

The following command generates persistence input and desired output files, in order to train and evaluate models and baselines.
```
python3 preprocess.py generate_corpus PATH_CLEANED_CORPUS --set SET
```

The following command gets an array of apparitions for each baseline:
```
python3 baselines.py get PATH_CLEANED_CORPUS
```

The following command evaluate the baselines given a set:
```
python3 evaluate.py get PATH_CLEANED_CORPUS --set SET
```

The following command trains a neural part-of-speech model given its description in a JSON file.
```
python3 train.py PATH_CLEANED_CORPUS PATH_ARCH_DESC.JSON MODEL_NAME
```
The following command evaluate a neural part-of-speech model given a set:
```
python3 evaluate.py PATH_CLEANED_CORPUS ARCH_DESC.JSON MODEL_NAME --set SET
```
The following command disambiguate a raw file using a trained neural part-of-speech model:
```
python3 tagger.py PATH_CLEANED_CORPUS ARCH_DESC.JSON MODEL_NAME PATH_INPUT.TXT PATH_OUTPUT.TXT
```

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

## License

[GNU General Public License v3.0](LICENSE)