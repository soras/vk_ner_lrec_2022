#
#  Trains EstNLTK's default NER model 
#  (https://github.com/estnltk/estnltk/tree/417c2ee4303a1a03650e703acb280e06883508d9/estnltk/taggers/estner/models/py3_default)
#  on the new data split. 
#  Saves the model into  retrain_estnltk_ner / default_model
#

from datetime import datetime
import os, os.path, sys
import logging

from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from estnltk import Text
from estnltk.taggers import NerTagger
from estnltk.converters import json_to_text

from estnltk.taggers.estner.ner_trainer import NerTrainer
from estnltk.taggers.estner.model_storage_util import ModelStorageUtil

from data_preprocessing import TokenizationPreprocessorFixed

train_data_dir = os.path.join('data', 'train')
dev_data_dir = os.path.join('data', 'dev')
assert os.path.exists( train_data_dir )
assert os.path.exists( dev_data_dir )

def collect_and_preprocess_data( dir, preprocessor, logger ):
    texts = []
    start = datetime.now()
    logger.info(f'collecting and preprocessing {dir!r} ...')
    with logging_redirect_tqdm():
        for fname in tqdm( sorted(list(os.listdir(dir))), ascii=True ):
            if not fname.endswith('.json'):
                continue
            text_obj = json_to_text( file=os.path.join(dir, fname) )
            # add segmentation
            preprocessor.preprocess( text_obj )
            # validate gold_wordner
            assert 'gold_wordner' in text_obj.layers, f'{text_obj.meta}'
            assert len(text_obj['gold_wordner']) == len(text_obj['words'])
            # add morph analysis
            text_obj.tag_layer('morph_analysis')
            texts.append( text_obj )
            #if len(texts) > 150:
            #    break
    log.info(f' {len(texts)} documents processed in {datetime.now()-start}.')
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

training_texts = collect_and_preprocess_data( train_data_dir, TokenizationPreprocessorFixed(), log )

start = datetime.now()
new_model_dir = os.path.join('retrain_estnltk_ner', 'default_model')
log.info('(*) Training NerTagger')
modelUtil = ModelStorageUtil(new_model_dir)
nersettings = modelUtil.load_settings()
trainer = NerTrainer(nersettings)
trainer.train( training_texts, layer='gold_wordner', model_dir=new_model_dir )
log.info(f'(*) NerTagger training done in {datetime.now()-start}\n')
