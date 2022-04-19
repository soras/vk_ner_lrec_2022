# ======================================
#   Utilities for phrase level NER 
#   evaluation
#
#   Requirements:
#      estnltk 1.6.9(.1)
#      nervaluate
#      tqdm
# ======================================

import json, os, os.path
from datetime import datetime

from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from estnltk.layer import AttributeList
from estnltk.converters import json_to_text

from nervaluate import Evaluator

full_tagset = ['PER', 'LOC_ORG', 'LOC', 'ORG', 'MISC']

def gather_eval_data_from_doc( text, gold_layer, auto_layer ):
    '''Gathers eval data in a format suitable for nervaluate.Evaluator.'''
    gold_annotations = []
    auto_annotations = []
    assert gold_layer in text.layers
    assert auto_layer in text.layers
    for annotation in text[ gold_layer ]:
        label = annotation.nertag
        if isinstance(label, list):
            label = label[0]
        assert isinstance(label, str), \
            f'(!) Unexpected gold nertag type: {type(label)}'
        start = int(annotation.start)
        end = int(annotation.end)
        gold_annotations.append( {"label": label, 
                                  "start": start, 
                                  "end": end} )
    for annotation in text[ auto_layer ]:
        label = annotation.nertag
        # label is an AttributeList: convert 
        # it to list and take the first item
        if isinstance(label, list):
            label = label[0]
        elif isinstance(label, AttributeList):
            label = label[0]
        assert isinstance(label, str), \
            f'(!) Unexpected auto nertag type: {type(label)}'
        start = int(annotation.start)
        end = int(annotation.end)
        auto_annotations.append( {"label": label, 
                                  "start": start, 
                                  "end": end} )
    return gold_annotations, auto_annotations


def collect_and_preprocess_data( dir, preprocessor, logger, add_morph=False ):
    '''Collects and preprocesses data (estnltk json files).'''
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
            if add_morph:
                text_obj.tag_layer('morph_analysis')
            texts.append( text_obj )
            #if len(texts) > 150:
            #    break
    return texts


def eval_nertagger( tagger, dir, documents, log, auto_layer='ner',
                                                 gold_layer='gold_ner',
                                                 full_tagset=full_tagset ):
    '''Evaluates given NER tagger on the dataset.'''
    all_gold_annotations = []
    all_auto_annotations = []
    start = datetime.now()
    log.info(f'evaluating on {dir!r} ...')
    with logging_redirect_tqdm():
        for text_obj in tqdm( documents, ascii=True ):
            tagger.tag( text_obj )
            gold_annotations, auto_annotations = \
                gather_eval_data_from_doc( text_obj, gold_layer, auto_layer )
            all_gold_annotations.append( gold_annotations )
            all_auto_annotations.append( auto_annotations )
    evaluator = Evaluator( all_gold_annotations, 
                           all_auto_annotations, 
                           tags = full_tagset )
    results, results_per_tag = evaluator.evaluate()
    results['number_of_documents'] = len(all_gold_annotations)
    results['results_per_tag'] = results_per_tag
    results['evaluation_phase'] = dir
    log.info(f' {len(all_gold_annotations)} documents processed in {datetime.now()-start}.')
    return results


def get_output_results_formatted( results, full_tagset=full_tagset, prefix_newline=True ):
    '''Gets pretty-formatted (strict) evaluation results.'''
    labels = ['', 'TOTAL'] + full_tagset
    f1   = ['f1']
    prec = ['prec']
    rec  = ['rec']
    for l in labels:
        if l == '':
            continue
        if l == 'TOTAL':
            f1.append( f"{results['strict']['f1']:.04f}" )
            prec.append( f"{results['strict']['precision']:.04f}" )
            rec.append( f"{results['strict']['recall']:.04f}" )
        else:
            assert l in results['results_per_tag']
            f1.append( f"{results['results_per_tag'][l]['strict']['f1']:.04f}" )
            prec.append( f"{results['results_per_tag'][l]['strict']['precision']:.04f}" )
            rec.append( f"{results['results_per_tag'][l]['strict']['recall']:.04f}" )
    output_str = []
    if prefix_newline:
        output_str.append( '\n' )
    for l in labels:
        output_str.append( f"{l:>9} ")
    output_str.append( '\n' )
    for p in prec:
        output_str.append( f"{p:>9} ")
    output_str.append( '\n' )
    for r in rec:
        output_str.append( f"{r:>9} ")
    output_str.append( '\n' )
    for f in f1:
        output_str.append( f"{f:>9} ")
    output_str.append( '\n' )
    return ''.join(output_str)


def write_out_results( results, model_name, eval_dir, results_dir ):
    '''Writes all evaluation results to json file inside results_dir.'''
    assert "number_of_documents" in results
    output_fname = f'results_{model_name}_eval_{eval_dir}_{results["number_of_documents"]}docs.json'
    output_fpath = os.path.join( results_dir, output_fname )
    with open( output_fpath, 'w', encoding='utf-8' ) as out_f:
        out_f.write ( json.dumps(results, ensure_ascii=False, indent=4) )


