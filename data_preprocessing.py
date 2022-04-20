# ==========================================================
#   Utilities for:
#
#   * Creating fixed tokenization layers ('words', 
#     'sentences') for the NER experiments;
#
#   * Converting gold standard NER annotations from
#     the format used in Kristjan Poska's experiments
#     to the format used in experiments with BERT
#     models (+ creating train/dev/test split for
#     the experiments).
#
#   * Loading input data of the transfer learning 
#     experiments.
#
#   * Preprocessing the input data: re-tokenizing 
#     words with BERT's tokenizer and adjusting 
#     NE labels according to BERT's tokenization;
#
#   Requires:
#      estnltk 1.6.9(.1)
#      transformers 4.0.0
#      tqdm
# ==========================================================

import sys, os, os.path
import regex as re
from shutil import copy2

from collections import defaultdict
from tqdm import tqdm

from estnltk.converters import text_to_json
from estnltk.converters import json_to_text

from estnltk import Layer
from estnltk.text import Text
from estnltk.taggers import Retagger
from estnltk.taggers import TokenSplitter
from estnltk.taggers import CompoundTokenTagger

# ====================================================
#   Token splitting from the first experiments:
#    https://github.com/pxska/bakalaureus/tree/main/experiments
# ====================================================

token_splitter = TokenSplitter(patterns=[re.compile(r'(?P<end>[A-ZÕÄÖÜ]{1}\w+)[A-ZÕÄÖÜ]{1}\w+'),\
                                         re.compile(r'(?P<end>Piebenomme)metsawaht'),\
                                         re.compile(r'(?P<end>maa)peal'),\
                                         re.compile(r'(?P<end>reppi)käest'),\
                                         re.compile(r'(?P<end>Kiidjerwelt)J'),\
                                         re.compile(r'(?P<end>Ameljanow)Persitski'),\
                                         re.compile(r'(?P<end>mõistmas)Mihkel'),\
                                         re.compile(r'(?P<end>tema)Käkk'),\
                                         re.compile(r'(?P<end>Ahjawalla)liikmed'),\
                                         re.compile(r'(?P<end>kohtumees)A'),\
                                         re.compile(r'(?P<end>Pechmann)x'),\
                                         re.compile(r'(?P<end>pölli)Anni'),\
                                         re.compile(r'(?P<end>külla)Rauba'),\
                                         re.compile(r'(?P<end>kohtowannem)Jaak'),\
                                         re.compile(r'(?P<end>rannast)Leno'),\
                                         re.compile(r'(?P<end>wallast)Kiiwita'),\
                                         re.compile(r'(?P<end>wallas)Kristjan'),\
                                         re.compile(r'(?P<end>Pedoson)rahul'),\
                                         re.compile(r'(?P<end>pere)Jaan'),\
                                         re.compile(r'(?P<end>kohtu)poolest'),\
                                         re.compile(r'(?P<end>Kurrista)kaudo'),\
                                         re.compile(r'(?P<end>mölder)Gottlieb'),\
                                         re.compile(r'(?P<end>wöörmündri)Jaan'),\
                                         re.compile(r'(?P<end>Oinas)ja'),\
                                         re.compile(r'(?P<end>ette)Leenu'),\
                                         re.compile(r'(?P<end>Tommingas)peab'),\
                                         re.compile(r'(?P<end>wäljaja)Kotlep'),\
                                         re.compile(r'(?P<end>pea)A'),\
                                         re.compile(r'(?P<end>talumees)Nikolai')])

# ========================================================

try:
    # EstNLTK 1.6.9(.1)
    from estnltk.taggers.text_segmentation.compound_token_tagger import ALL_1ST_LEVEL_PATTERNS
    from estnltk.taggers.text_segmentation.compound_token_tagger import CompoundTokenTagger
except ImportError as err:
    # EstNLTK 1.7.0
    from estnltk.taggers.standard.text_segmentation.compound_token_tagger import ALL_1ST_LEVEL_PATTERNS
    from estnltk.taggers.standard.text_segmentation.compound_token_tagger import CompoundTokenTagger
except:
    raise

def make_adapted_cp_tagger(**kwargs):
    '''Creates an adapted CompoundTokenTagger that exludes roman numerals from names with initials.'''
    from estnltk.taggers.text_segmentation.patterns import MACROS
    redefined_pat_1 = \
        { 'comment': '*) Names starting with 2 initials (exlude roman numerals I, V, X from initials);',
          'pattern_type': 'name_with_initial',
          'example': 'A. H. Tammsaare',
          '_regex_pattern_': re.compile(r'''
                            ([ABCDEFGHJKLMNOPQRSTUWYZŠŽÕÄÖÜ][{LOWERCASE}]?)   # first initial
                            \s?\.\s?-?                                        # period (and hypen potentially)
                            ([ABCDEFGHJKLMNOPQRSTUWYZŠŽÕÄÖÜ][{LOWERCASE}]?)   # second initial
                            \s?\.\s?                                          # period
                            ((\.[{UPPERCASE}]\.)?[{UPPERCASE}][{LOWERCASE}]+) # last name
                            '''.format(**MACROS), re.X),
         '_group_': 0,
         '_priority_': (4, 1),
         'normalized': lambda m: re.sub('\1.\2. \3', '', m.group(0)),
         }

    redefined_pat_2 = \
       { 'comment': '*) Names starting with one initial (exlude roman numerals I, V, X from initials);',
         'pattern_type': 'name_with_initial',
         'example': 'A. Hein',
         '_regex_pattern_': re.compile(r'''
                            ([ABCDEFGHJKLMNOPQRSTUWYZŠŽÕÄÖÜ])   # first initial
                            \s?\.\s?                            # period
                            ([{UPPERCASE}][{LOWERCASE}]+)       # last name
                            '''.format(**MACROS), re.X),
         '_group_': 0,
         '_priority_': (4, 2),
         'normalized': lambda m: re.sub('\1. \2', '', m.group(0)),
       }
    new_1st_level_patterns = []
    for pat in ALL_1ST_LEVEL_PATTERNS:
        if pat['comment'] == '*) Names starting with 2 initials;':
            # Replace this pattern
            new_1st_level_patterns.append( redefined_pat_1 )
        elif pat['comment'] == '*) Names starting with one initial;':
            # Replace this pattern
            new_1st_level_patterns.append( redefined_pat_2 )
        else:
            new_1st_level_patterns.append( pat )
    assert len(new_1st_level_patterns) == len(ALL_1ST_LEVEL_PATTERNS)
    if kwargs is not None:
        assert 'patterns_1' not in kwargs.keys(), "(!) Cannot overwrite 'patterns_1' in adapted CompoundTokenTagger."
    return CompoundTokenTagger( patterns_1=new_1st_level_patterns, **kwargs )

# ========================================================
#   A shortcut class for fixed tokenization preprocessing   
# ========================================================

class TokenizationPreprocessorFixed:
    '''
    Provides fixed tokenization layers ('tokens', 'words', 'sentences') for NER experiments.
    Partly based on:
    https://github.com/pxska/bakalaureus/blob/main/experiments/modules/preprocessing_protocols.py
    
    Note that these tokenization fixes are by no means complete and do not solve all 
    the tokenization issues present in the historical language.
    '''

    def __init__(self):
        self.token_splitter = token_splitter
        self.cp_tagger = \
            make_adapted_cp_tagger(tag_initials=True, tag_abbreviations=True, tag_hyphenations=True)

    def preprocess( self, text ):
        text = text.tag_layer(['tokens'])
        self.token_splitter.retag( text )
        self.cp_tagger.tag( text )
        text.tag_layer('sentences')
        return text

# ====================================================
#   Convert data from EstNLTK json to CONLL NER
# ====================================================

def align_words_and_ner_phrases( words_layer, ner_phrases_layer ):
    '''
    Finds a mapping from word tokens to overlapping NE phrases. 
    Returns a dictionary where keys are indexes from the words_layer,
    and values are lists of entities from ner_phrases_layer.
    '''
    words_to_phrases_map = dict()
    word_id = 0
    while word_id < len( words_layer ):
        word    = words_layer[word_id]
        w_start = word.start
        w_end   = word.end
        matching_phrases = []
        for i in range( len(ner_phrases_layer) ):
            ner_phrase = ner_phrases_layer[i]
            if w_end < ner_phrase.start:
                continue
            if w_end > ner_phrase.start and w_end < ner_phrase.end:
                # inside ner phrase and possibly next word also overlaps the phrase
                '''
                xxxxxxxx
                  yyyyy

                xxxxxxxx
                yyyyy

                  xxxxxxxx
                yyyyy
                '''
                matching_phrases.append(ner_phrase)
            elif w_end >= ner_phrase.start and w_start < ner_phrase.end and w_end >= ner_phrase.end:
                # inside ner phrase and next word cannot overlap the phrase
                '''
                xxxxxxxx
                yyyyyyyy

                xxxxxxxx
                   yyyyy

                xxxxxxxx
                      yyyyy
                '''
                matching_phrases.append(ner_phrase)
        if matching_phrases:
            if word_id not in words_to_phrases_map:
                words_to_phrases_map[word_id] = []
            words_to_phrases_map[word_id].extend( matching_phrases )
        word_id += 1
    return words_to_phrases_map


def create_wordner_layer( text_obj, input_layer='gold_ner', output_layer='gold_wordner' ):
    '''Creates a word-level NE layer based on given segmented text obj and phrase-level NE annotations.'''
    assert input_layer in text_obj.layers
    assert 'sentences' in text_obj.layers
    assert 'words' in text_obj.layers
    words_to_ner_map = align_words_and_ner_phrases( text_obj['words'], 
                                                    text_obj[input_layer] )
    wordner_layer = Layer( name=output_layer, text_object=text_obj, 
                           attributes=['nertag'] )
    word_id = 0
    for sent in text_obj['sentences']:
        s_start = sent.start
        s_end = sent.end
        sent_words = []
        # ====================================================
        #  Align word tokens with NE phrases
        # ====================================================
        assert words_to_ner_map is not None
        last_word_tag = 'O'
        while word_id < len( text_obj['words'] ):
            word = text_obj['words'][word_id]
            w_start = word.start
            w_end   = word.end
            nertag = 'O'
            if s_start <= w_start and w_end <= s_end:
                if word_id not in words_to_ner_map:
                    nertag = 'O'
                else:
                    ner_phrases = words_to_ner_map[word_id]
                    ner_labels = list(set([p.annotations[0].nertag for p in ner_phrases]))
                    if len(ner_labels) > 1:
                        print(f'(!) Warning, more than 1 label for {word.text!r}: {ner_labels!r}. Picking first.')
                    nertag = ner_labels[0]
                    word_starts_new_phrase = any([w_start == p.start for p in ner_phrases])
                    # Find prefix for the tag
                    if word_starts_new_phrase:
                        nertag = 'B-'+nertag
                    elif last_word_tag in ['B-'+nertag, 'I-'+nertag]:
                        nertag = 'I-'+nertag
                    else:
                        # A tricky case:
                        # 1) the sentence boundary mistakenly breaks NER boundary, so we have to start new 
                        #    NER phrase along with a new sentence;
                        # 2) Word containing a named entity is not properly split into tokens, e.g. 
                        #    'temma-Josep' -- stretch the annotation to cover the whole word even if
                        #    it covers only a sub word ...
                        nertag = 'B-'+nertag
                wordner_layer.add_annotation( word.base_span, **{'nertag':nertag} )
            elif w_start < s_start and w_end <= s_start:
                print(f'(!) Word {word.text!r} unexpectedly outside the last sentence. Skipping in {fpath!r}')
                word_id += 1
            else:
                break
            last_word_tag = nertag
            word_id += 1
    assert len(wordner_layer) == len(text_obj['words'])
    return wordner_layer


def convert_estnltk_json_to_train_dev_test_dir( in_directory, out_directory, split_goals=[1125, 125, 250], 
                                                out_file_prefix='_', out_file_suffix='', remove_old=True, 
                                                preprocessor=TokenizationPreprocessorFixed(), 
                                                copy_json_files=False ):
    ''' 
    Converts gold standard NER annotations (in estnltk's json files) from the 
    crossvalidation data split (used in Kristjan Poska's experiments) to conll 
    NER annotations (in IOB2 format). Retokenizes files with the given pre-
    processor, and splits output files into train/dev/test sets.
    
    The input directory must contain sub directories '1', '2', '3', '4', '5', 
    '6' with the gold standard json files from Poska's experiments.
    
    As a result, creates sub directories 'train', 'dev' and 'test', each of which 
    contain both conll format annotations (in '_data.txt' files) and estnltk's 
    json files (containing both the original NE phrase annotations in "gold_ner" 
    layer, and IOB2 NE annotations in the "gold_wordner" layer).
    '''
    assert os.path.isdir( in_directory )
    assert len(split_goals) == 3
    # Split files into train, dev and test
    split_files = { 'train': [], 'dev': [], 'test': [] }
    for item in sorted(os.listdir(in_directory)):
        if item in ['train', 'dev', 'test']:
            continue
        fpath = os.path.join( in_directory, item )
        if os.path.isdir( fpath ):
            for fname in sorted(os.listdir( fpath )):
                if not fname.endswith('.json'):
                    continue
                fpath2 = os.path.join( fpath, fname )
                if len( split_files['train'] ) < split_goals[0]:
                    assert item != '6'
                    split_files['train'].append( fpath2 )
                elif len( split_files['dev'] ) < split_goals[1]:
                    assert item != '6'
                    split_files['dev'].append( fpath2 )
                elif len( split_files['test'] ) < split_goals[2]:
                    assert item == '6'
                    split_files['test'].append( fpath2 )

    # Iterate over files of the 3 splits
    def split_iterator( split ):
        for sp in ['train', 'dev', 'test']:
            for fpath in split[sp]:
                yield sp, fpath

    # Display statistics and write results into files
    def write_out_data( spl, d_counters, cur_collected_data, cur_labels, out_directory, 
                                         collected_text_objects=None, out_file_prefix='_', 
                                                                      out_file_suffix='' ):
        # Statistics
        print()
        print(' Split:', spl )
        print()
        print(' Total texts in subset: ', d_counters['docs'] )
        print(' Total sents in subset: ', d_counters['sentences'] )
        print(' Total words in subset: ', d_counters['words'] )
        print(' Tagged words in subset: ', d_counters['ner_words'] )
        print('   NE phrases in subset: ', d_counters['ner_phrases'] )
        if out_file_prefix is None:
            out_file_prefix = ''
        out_dir = os.path.join(out_directory, spl)
        os.makedirs( out_dir, exist_ok=True )
        out_fname = os.path.join(out_dir, f'{out_file_prefix}data{out_file_suffix}.txt')
        if remove_old and os.path.exists( out_fname ):
            os.remove( out_fname )
        with open(out_fname, 'w', encoding='utf-8') as out_f:
            for line in cur_collected_data:
                out_f.write(line+'\n')
        print('--> ', out_fname)
        out_fname = os.path.join(out_dir, f'{out_file_prefix}labels{out_file_suffix}.txt')
        if remove_old and os.path.exists( out_fname ):
            os.remove( out_fname )
        with open(out_fname, 'w', encoding='utf-8') as out_f:
            for line in sorted(list(cur_labels)):
                out_f.write(line+'\n')
        print('--> ', out_fname)
        if collected_text_objects:
            cpy_count = 0
            for text_obj in collected_text_objects:
                # Clean up and validate layers
                assert 'words' in text_obj.layers, f'{text_obj.meta}'
                assert 'tokens' in text_obj.layers, f'{text_obj.meta}'
                text_obj.pop_layer('words')
                text_obj.pop_layer('tokens')
                assert 'gold_ner' in text_obj.layers
                assert 'gold_wordner' in text_obj.layers
                assert len(text_obj.layers) == 2, f'{text_obj.layers}'
                # Write out
                fname = text_obj.meta['filename'] 
                assert fname.endswith('.json')
                out_fpath = os.path.join( out_directory, spl, fname )
                text_to_json( text_obj, file=out_fpath )
                cpy_count += 1
            print()
            print( '{} json files copied to {!r}'.format( cpy_count, spl ) )
        print()
    
    # Convert files for train, dev and test
    labels = set()
    all_collected_data = []
    counters = defaultdict(int)
    old_split = 'train'
    collected_text_objects_l = []
    for split, fpath in tqdm( split_iterator( split_files ), ascii=True ):
        if split != old_split:
            if not copy_json_files:
                collected_text_objects_l = None
            # Write out data
            write_out_data( old_split, counters, all_collected_data, labels, 
                            out_directory, 
                            collected_text_objects=collected_text_objects_l,
                            out_file_prefix=out_file_prefix, 
                            out_file_suffix=out_file_suffix )
            labels = set()
            all_collected_data = []
            collected_text_objects_l = []
            counters = defaultdict(int)
        # Load original annotations
        text = json_to_text( file=fpath )
        # Add tokenization
        if preprocessor is not None:
            preprocessor.preprocess( text )
        else:
            text.tag_layer( 'sentences' )
        assert 'gold_ner' in text.layers
        # Remove old wordner_layer
        if 'gold_wordner' in text.layers:
            text.pop_layer( 'gold_wordner' )
        # Create new gold wordner_layer based 
        # on the fixed tokenization
        wordner_layer = \
            create_wordner_layer( text, input_layer='gold_ner', 
                                        output_layer='gold_wordner' )
        text.add_layer( wordner_layer )
        # Convert annotations to CONLL-style word level annotations
        word_id = 0
        in_path, fname = os.path.split( fpath )
        collected_data = [ '### '+fname ]
        text.meta['filename'] = fname
        for sent in text['sentences']:
            s_start = sent.start
            s_end = sent.end
            sent_words = []
            while word_id < len( wordner_layer ):
                word = wordner_layer[word_id]
                w_start = word.start
                w_end   = word.end
                if s_start <= w_start and w_end <= s_end:
                    sent_words.append( word.text+' '+str(word.nertag) )
                    counters['words'] += 1
                    labels.add( str(word.nertag) )
                    if word.nertag != 'O':
                        counters['ner_words'] += 1
                        labels.add( str(word.nertag) )
                elif w_start < s_start and w_end <= s_start:
                    print(f'(!) Word {word.text!r} unexpectedly outside the last sentence. Skipping word in {fname!r}')
                    word_id += 1
                else:
                    break
                word_id += 1
            if sent_words:
                counters['sentences'] += 1
                collected_data.extend( sent_words )
                collected_data.append( '' )
        counters['ner_phrases'] += len(text.gold_ner)
        counters['docs'] += 1
        all_collected_data.extend( collected_data )
        collected_text_objects_l.append( text )
        old_split = split
    if len(all_collected_data) > 0:
        if not copy_json_files:
            collected_text_objects_l = None
        # Write out data
        write_out_data( old_split, counters, all_collected_data, labels, 
                        out_directory, 
                        collected_text_objects=collected_text_objects_l,
                        out_file_prefix=out_file_prefix, 
                        out_file_suffix=out_file_suffix )
        collected_text_objects_l = []


# ====================================================
#   Prepare the sentences and labels for 
#   transfer learning
# ====================================================

def load_conll_sentences_and_labels( in_file, label_set=None, remove_underscore=False ):
    '''
    Loads data from conll format file ('_data.txt'). 
    Assumes that the loaded data has been created with the 
    function convert_estnltk_json_to_train_dev_test_dir().
    '''
    assert os.path.exists( in_file )
    _all_sentences = []
    _all_labels = []
    last_sent_tokens = []
    last_sent_labels = []
    with open(in_file, 'r', encoding='utf-8') as in_f:
        for line in in_f:
            line = line.strip()
            if line.startswith('###'):
                # Skip comment line
                continue
            if len(line) > 0:
                items = line.split()
                label = items[-1]
                token = ' '.join(items[:-1])
                if remove_underscore:
                    label = label.replace('_', '')
                if label_set:
                    assert label in label_set
                last_sent_tokens.append(token)
                last_sent_labels.append(label)
            else:
                if last_sent_tokens:
                    _all_sentences.append(last_sent_tokens)
                    _all_labels.append(last_sent_labels)
                last_sent_tokens = []
                last_sent_labels = []
        if last_sent_tokens:
            _all_sentences.append(last_sent_tokens)
            _all_labels.append(last_sent_labels)
    return _all_sentences, _all_labels


def load_conll_data_from_dir( dir, label_set=None, remove_underscore=False, logger=None, 
                              data_file_suffix=None, return_dir_to_data_map=False ):
    '''
    Loads conll format training or evaluation data from the given directory.
    Assumes that the loaded data has been created with the 
    function convert_estnltk_json_to_train_dev_test_dir().
    '''
    _all_sents  = []
    _all_labels = []
    _dir_to_all_sents  = {}
    _dir_to_all_labels = {}
    if data_file_suffix is None:
        data_file_suffix = ''
    assert isinstance(data_file_suffix, str)
    assert os.path.isdir( dir )
    for fname in os.listdir( dir ):
        if not fname.endswith('.txt'):
            # Skip json files
            continue
        if 'labels' in fname and fname.endswith('.txt'):
            # Skip label files
            continue
        if not fname.endswith(data_file_suffix+'.txt'):
            # Skip alternatie tokenization data files
            continue
        fpath = os.path.join( dir, fname )
        _sents, _labels = load_conll_sentences_and_labels( fpath, 
                                                           label_set=label_set, 
                                                           remove_underscore=remove_underscore )
        _sents_len  = len(_sents)
        _tokens_len = sum( [len(s) for s in _sents] )
        if logger:
            logger.info(f' Loaded {_sents_len} sentences, {_tokens_len} tokens from {fpath!r}')
        # Collect all sentences and labels into one list
        _all_sents.extend( _sents )
        _all_labels.extend( _labels )
    return _all_sents, _all_labels


def load_and_prepare_bert_data( data_dir, bert_tokenizer, log, data_file_suffix=None, cut=None ):
    '''
    Loads training data from the given directory and retokenizes with bert_tokenizer.
    
    Returns tuple (tokenized_texts, token_labels, tag2idx, idx2tag, tag_values):
    * `tokenized_texts` - list of lists: the outer list represents sentence segmentation, 
        and the inner list token segmentation inside sentence (retokenized by BERT);
    * `token_labels` - list of lists: the outer list represents sentence segmentation, 
        and the inner list NE tags of the tokens (of the sentence);
    * `tag2idx` - dictionary mapping from NE tag names to numeric tag indexes;
    * `idx2tag` - dictionary mapping from numeric tag indexes to NE tag names;
    * `tag_values` - list of all NE tag names;
    '''
    #  Load CONLL NER format data
    loaded_sents, loaded_labels = \
        load_conll_data_from_dir( data_dir, label_set=None, remove_underscore=False, logger=log,
                                  data_file_suffix=data_file_suffix )
    
    #  Cut the data (for fast testing)
    if cut is not None:
        assert isinstance(cut, int)
        log.info(f' Cutting all dataset sizes to first {cut!r} sentences' )
        loaded_sents  = loaded_sents[:cut]
        loaded_labels = loaded_labels[:cut]

    # Summarize loaded data
    # Refresh labels set: use only those available in the data    
    all_sents  = loaded_sents
    all_labels = loaded_labels
    tag2idx, idx2tag, tag_values = get_used_labels_mapping( all_labels )
    total_tokens    = sum([len(sent) for sent in all_sents])
    total_sentences = len(all_sents)
    
    if log:
        log.info(f' Labels used: {tag_values!r}' )
        log.info(f' Full dataset size:  {total_sentences} sentences, {total_tokens} tokens')

    #  Tokenize data with Bert's tokenizer
    bert_tokenized_texts_and_labels = [
        tokenize_and_preserve_labels(sent, labs, bert_tokenizer)
        for sent, labs in zip(all_sents, all_labels)
    ]
    tokenized_texts = [token_label_pair[0] for token_label_pair in bert_tokenized_texts_and_labels]
    token_labels = [token_label_pair[1] for token_label_pair in bert_tokenized_texts_and_labels]
    return tokenized_texts, token_labels, tag2idx, idx2tag, tag_values


def tokenize_and_preserve_labels(sentence, text_labels, bert_tokenizer, adjust_bi_labels=True):
    '''
    Given a sentence and NE labels of its words, retokenizes words of the sentence with 
    bert_tokenizer, and returns resulting BERT's tokens and their NE labels.
    
    If `adjust_bi_labels==True` (default), then B- prefixes will be fixed according
    to BERT's tokenization. That is: if the original word has NE tag with prefix 'B-',
    and the word is subtokenized into several tokens by BERT, then only the first token 
    will obtain the tag with prefix 'B-' and all the subsequent tokens will obtain tags 
    with the prefix 'I-'.
    
    Note: the original source of this function is from the tutorial:
    https://www.depends-on-the-definition.com/named-entity-recognition-with-bert/ 
    (last checked: 2022-04-18)
    '''
    tokenized_sentence = []
    labels = []

    for word, label in zip(sentence, text_labels):

        # Tokenize the word and count # of subwords the word is broken into
        tokenized_word = bert_tokenizer.tokenize(word)
        n_subwords = len(tokenized_word)

        # Add the tokenized word to the final tokenized word list
        tokenized_sentence.extend(tokenized_word)

        if adjust_bi_labels and label.startswith(('B-', 'B_')):
            # Adjust B- labels:  only the first label should start 
            # with B-, the following labels should start with I-
            new_labels = []
            new_labels.append( label )
            label_ending = label[2:]
            label_start = 'I-' if label.startswith('B-') else 'I_'
            for i in range(n_subwords-1):
                new_labels.append( label_start + label_ending )
            labels.extend( new_labels )
        else:
            # Add the same label to the new list of labels `n_subwords` times
            labels.extend( [label] * n_subwords )

    return tokenized_sentence, labels


def get_used_labels_mapping( labels_for_sents, remove_underscore=False ):
    '''
    Get all labels used in the dataset. Create mappings from labels to 
    their numeric indexes.
    Returns tuple (tag2idx, idx2tag, tag_values):
    * `tag2idx` - dictionary mapping from NE tag names to numeric tag indexes;
    * `idx2tag` - dictionary mapping from numeric tag indexes to NE tag names;
    * `tag_values` - list of all NE tag names;
    '''
    labels = []
    for sent in labels_for_sents:
        for label in sent:
            assert isinstance( label, str )
            if remove_underscore:
                label = label.replace('_', '')
            if label not in labels:
                labels.append( label )
    labels = sorted( labels )
    labels.append("PAD")
    tag2idx = {t: i for i, t in enumerate(labels)}
    idx2tag = {i: t for i, t in enumerate(labels)}
    return tag2idx, idx2tag, labels

    