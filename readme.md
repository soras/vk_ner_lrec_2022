This repository contains data and source code of the experiments reported in LREC 2022 paper "Named Entity Recognition in Estonian 19th Century Parish Court Records".

The goal is to experiment with automatic named entity recognition on historical Estonian texts -- on 19th century Estonian communal court minute books, which have been manually annotated for named entities.
In the experiments, we (re)train a traditional machine learning NER approach as a baseline, and finetune different BERT-based transfer learning models for NER. 

## Dataset

The folder [data](data) contains 19th century Estonian communal court minute books. 
These materials originate the [crowdsourcing project](https://www.ra.ee/vallakohtud/) of The National Archives of Estonia,  and have been manually annotated with named entities in the project "Possibilities of automatic analysis of historical texts by the example of 19th-century Estonian communal court minutes".
The project "Possibilities of automatic analysis of historical texts by the example of 19th-century Estonian communal court minutes" is funded by the national programme "Estonian Language and Culture in the Digital Age 2019-2027".

## Prerequisites

Python 3.7+ is required. 
For detailed package requirements, see the file [conda_environment.yml](conda_environment.yml) (contains [conda environment](https://docs.conda.io/projects/conda/en/latest/user-guide/concepts/environments.html) that was used in the experiments).



## Processing steps

### Preprocessing

* `00_convert_crossval_json_to_conll_train_dev_test.py` -- Converts gold standard NER annotations (in json files) from the format used in [Kristjan Poska's experiments](https://github.com/pxska/bakalaureus/) to conll NER annotations (in IOB2 format) and splits into train/dev/test datasets. See comments in the header of the script for details.
	* Note #1: you only need to run this script if you want to make a new (different) data split or if you want to change the tokenization. Otherwise, you can use an existing split from [data](data) (`train`, `dev` and `test`); 
	* Note #2: the script also outputs statistics of the corpus. For statistics of the last run, see the comment at the end of the script.

### Baseline (selection)

* `01a_estnltk_ner_retraining_best_model.py` -- Retrains the best model from Kristjan Poska's experiments on the new data split.
* `01b_eval_estnltk_ner_best_model_on_dev_test.py` -- Evaluates the previous model on `dev` and `test` sets.
* `02a_estnltk_ner_retraining_default_model_baseline.py` -- Trains NER model with EstNLTK's [default NER settings](https://github.com/estnltk/estnltk/tree/417c2ee4303a1a03650e703acb280e06883508d9/estnltk/taggers/estner/models/py3_default) on the new data split.
* `02b_eval_estnltk_ner_default_model_on_dev_test.py` -- Evaluates the previous model on `dev` and `test` sets.
	* Note: initially, we wanted to use the best model from Kristjan Poska's experiments as the baseline. However, after   retraining and evaluating the model on the new data split (steps `01a` and `01b`), its performance turned out to be lower than previously measured, and lower than the performance of retrained EstNLTK's default NER model (steps `02a` and `02b`). So, we chose the retrained default NER model (steps `02a` and `02b`) as the new baseline.

### Training BERT models

* `03_train_and_eval_bert_model.py` -- Trains and evaluates BERT-based NER model. First, performs a grid search to find the best configuration of hyperparameters for training. Then trains the model with the best configuration for 10 epochs, keeps and saves the best model (based on F1 score on the 'dev' set), and finally evaluates the best model on the 'test' set.
	* Assumes that the corresponding models  have already been downloaded and unpacked into local directories `'EstBERT'`, `'WikiBert-et'` and `'est-roberta'`. You can download the models from urls: 
		* [https://huggingface.co/tartuNLP/EstBERT](https://huggingface.co/tartuNLP/EstBERT)
		* [https://huggingface.co/TurkuNLP/wikibert-base-et-cased](https://huggingface.co/TurkuNLP/wikibert-base-et-cased)
		* [https://huggingface.co/EMBEDDIA/est-roberta](https://huggingface.co/EMBEDDIA/est-roberta) 
	* The directory name of trainable model should be given as the command line argument of the script, e.g. `python  train_and_eval_bert_model.py EstBERT`.

## Results

### Evaluation results

* [logs](logs) -- excerpts of training and evaluation log files with the final results.
* [results](results) -- detailed evaluation results in json format, using evaluation metrics from the [nervaluate package](https://github.com/MantisAI/nervaluate).

### Models

* [retrain\_estnltk\_ner](retrain_estnltk_ner) -- retrained EstNLTK's NerTagger models from steps `01a` and `02a`.
* `bert_models` -- **TODO**

### Error inspection

* [error_inspection](error_inspection) -- contains code for inspecting errors of the best model. The notebook `find_estroberta_ner_errors_on_test_corpus.ipynb` annotates 'test' set with the best model (finetuned `est-roberta`), and shows all the differences between gold standard annotations and automatically added annotations. Annotation differences are augmented with their textual contexts to ease the manual inspection. 


