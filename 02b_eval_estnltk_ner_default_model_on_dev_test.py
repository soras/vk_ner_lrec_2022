#
#  Evaluates retrained EstNLTK's default NER model 
#  on the new data split (on dev and test sets). 
#

from datetime import datetime
import os, os.path, sys
import logging
import json

from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from estnltk import Text
from estnltk.taggers import NerTagger
from estnltk.converters import json_to_text

from data_preprocessing import TokenizationPreprocessorFixed

from eval_utils import eval_nertagger
from eval_utils import get_output_results_formatted
from eval_utils import write_out_results

dev_data_dir = os.path.join('data', 'dev')
test_data_dir = os.path.join('data', 'test')
assert os.path.exists( dev_data_dir )
assert os.path.exists( test_data_dir )
results_dir = 'results'
if not os.path.exists( results_dir ):
    os.makedirs(results_dir, exist_ok=True)

# Collects and preprocessed data (json files)
def collect_and_preprocess_data( dir, preprocessor, logger ):
    texts = []
    logger.info(f'collecting and preprocessing {dir!r} ...')
    with logging_redirect_tqdm():
        for fname in tqdm( sorted(list(os.listdir(dir))), ascii=True ):
            if not fname.endswith('.json'):
                continue
            text_obj = json_to_text( file=os.path.join(dir, fname) )
            # add segmentation
            preprocessor.preprocess( text_obj )
            # validate gold_ner
            assert 'gold_ner' in text_obj.layers, f'{text_obj.meta}'
            # add morph analysis
            text_obj.tag_layer('morph_analysis')
            texts.append( text_obj )
            #if len(texts) > 150:
            #    break
    return texts

log = logging.getLogger(__name__)
f_handler = logging.FileHandler(sys.argv[0]+'.log', mode='w', encoding='utf-8')
f_handler.setLevel(logging.INFO)
c_handler = logging.StreamHandler()
c_handler.setLevel(logging.INFO)

c_format = logging.Formatter('%(levelname)s - %(message)s')
f_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%dT%H:%M:%S')
c_handler.setFormatter(c_format)
f_handler.setFormatter(f_format)

log.addHandler(f_handler)
log.addHandler(c_handler)
log.setLevel(logging.INFO)

log.info('(*) Evaluating NerTagger')
start = datetime.now()
model_dir = os.path.join('retrain_estnltk_ner', 'default_model')
nertagger = NerTagger( model_dir )
# dev set
dev_texts = collect_and_preprocess_data( dev_data_dir, TokenizationPreprocessorFixed(), log )
dev_results = eval_nertagger( nertagger, dev_data_dir, dev_texts, log, auto_layer='ner' )
pretty_results = get_output_results_formatted( dev_results )
log.info( pretty_results )
write_out_results( dev_results, 'estnltk-default-ner', 'dev', results_dir )
# test set
test_texts = collect_and_preprocess_data( test_data_dir, TokenizationPreprocessorFixed(), log )
test_results = eval_nertagger( nertagger, test_data_dir, test_texts, log, auto_layer='ner' )
pretty_results = get_output_results_formatted( test_results )
log.info( pretty_results )
write_out_results( test_results, 'estnltk-default-ner', 'test', results_dir )

log.info(f'(*) Whole evaluation took {datetime.now()-start}\n')
