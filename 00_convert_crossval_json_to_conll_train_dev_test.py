#
#   Converts gold standard NER annotations (in json files) from the 
#   crossvalidation split (used in Kristjan Poska's experiments) 
#   to conll NER annotations (in IOB2 format), split into 
#   train/dev/test datasets.
#
#   For running this script, you first need to obtain the gold standard 
#   NER annotation files from Kristjan Poska's experiments:
#      https://github.com/pxska/bakalaureus/tree/main/data/vallakohtufailid-json-flattened 
#   You should split these files into sub directories '1', '2', '3', '4', 
#   '5', '6' according to the data split described here:
#      https://github.com/pxska/bakalaureus/blob/main/data/divided_corpus.txt
#      https://github.com/pxska/bakalaureus/blob/main/data/corpus_subdistribution_without_hand_tagged.txt
#   This assures that the test set ('6') will remain the same as in 
#   the previous experiments, and files in directories '1' to '5' 
#   will be resplit into training and development sets.
#
#   The script also outputs (tokenization and NER annotation) statistics 
#   of the converted corpus. See the comment below for statistics of the 
#   last run. 
#
#   If in_dir already contains non-empty directories 'train', 'dev' 
#   and 'test', you do not need to run this script.
#   

import os, os.path

from data_preprocessing import TokenizationPreprocessorFixed
from data_preprocessing import convert_estnltk_json_to_train_dev_test_dir

in_dir = 'data'

# Check that all source sub directories are present
source_dirs = ['1', '2', '3', '4', '5', '6']
for item in source_dirs:
    fpath = os.path.join( in_dir, item )
    if not os.path.isdir( fpath ):
        raise Exception( ('(!) Missing a source directory {!r} from the '+\
                          'cross-validation experiments. \nPlease consult '+\
                          'header of this script about how to obtain the '+\
                          'source files.').format(fpath) )

preprocessor=TokenizationPreprocessorFixed()
out_file_prefix='_'
out_file_suffix=''

convert_estnltk_json_to_train_dev_test_dir( in_dir, in_dir, split_goals=[1125, 125, 250], 
                                            out_file_prefix=out_file_prefix, out_file_suffix=out_file_suffix,
                                            preprocessor=preprocessor,
                                            remove_old=True, copy_json_files=True )

"""

============================================
 2021-12-29: with fixed tokenization
============================================

INFO:utils.py:157: NumExpr defaulting to 4 threads.
1125it [00:54, 27.80it/s]
 Split: train

 Total texts in subset:  1125
 Total sents in subset:  16040
 Total words in subset:  240614
 Tagged words in subset:  37750
   NE phrases in subset:  20944
-->  data\train\_data.txt
-->  data\train\_labels.txt

1125 json files copied to 'train'

1249it [01:08, 17.88it/s]
 Split: dev

 Total texts in subset:  125
 Total sents in subset:  2336
 Total words in subset:  28891
 Tagged words in subset:  4364
   NE phrases in subset:  2357
-->  data\dev\_data.txt
-->  data\dev\_labels.txt

125 json files copied to 'dev'

1500it [01:20, 18.67it/s]

 Split: test

 Total texts in subset:  250
 Total sents in subset:  3170
 Total words in subset:  50900
 Tagged words in subset:  7681
   NE phrases in subset:  4239
-->  data\test\_data.txt
-->  data\test\_labels.txt

250 json files copied to 'test'

"""
