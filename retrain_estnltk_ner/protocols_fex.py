#
#  NER feature extraction from Kristjan Poska's experiments:
#   https://github.com/pxska/bakalaureus/tree/main/experiments/models
#

from estnltk.taggers import Tagger, Retagger
from estnltk.taggers.estner.fex import get_shape, get_2d, get_lemma, get_pos, is_prop, get_word_parts, get_case, get_ending, get_4d, get_dand, get_capperiod, get_all_other, contains_upper, contains_lower, contains_alpha, contains_digit, degenerate, b, contains_symbol, split_char
from estnltk.text import Text
from estnltk.layer.layer import Layer
from typing import MutableMapping
from collections import defaultdict
import codecs
import os

class NerEmptyFeatureTagger(Tagger):
    """Extracts features provided by the morphological analyser pyvabamorf. """
    conf_param = ['settings']

    def __init__(self, settings, input_layers = ('words',), output_layer='ner_features',
                 output_attributes=("lem", "pos", "prop", "pref", "post", "case",
                                    "ending", "pun", "w", "w1", "shape", "shaped", "p1",
                                    "p2", "p3", "p4", "s1", "s2", "s3", "s4", "d2",
                                    "d4", "dndash", "dnslash", "dncomma", "dndot", "up", "iu", "au",
                                    "al", "ad", "ao", "aan", "cu", "cl", "ca", "cd",
                                    "cp", "cds", "cdt", "cs", "bdash", "adash",
                                    "bdot", "adot", "len", "fsnt", "lsnt", "gaz",
                                    "prew", "next", "iuoc", "pprop", "nprop", "pgaz",
                                    "ngaz", "F")):
        self.settings = settings
        self.output_layer = output_layer
        self.output_attributes = output_attributes
        self.input_layers = input_layers

    def _make_layer(self, text: Text, layers: MutableMapping[str, Layer], status: dict):
        layer = Layer(self.output_layer, ambiguous=True, attributes=self.output_attributes, text_object=text)
        for token in text.words:
            layer.add_annotation(token, lem=None, pos=None,
                                 prop=None,
                                 pref=None,
                                 post=None,
                                 case=None, ending=None,
                                 pun=None, w=None, w1=None, shape=None, shaped=None, p1=None,
                                 p2=None, p3=None, p4=None, s1=None, s2=None, s3=None, s4=None, d2=None, d4=None,
                                 dndash=None,
                                 dnslash=None, dncomma=None, dndot=None, up=None, iu=None, au=None,
                                 al=None, ad=None, ao=None, aan=None, cu=None, cl=None, ca=None,
                                 cd=None, cp=None, cds=None, cdt=None, cs=None, bdash=None, adash=None,
                                 bdot=None, adot=None, len=None, fsnt=None, lsnt=None, gaz=None, prew=None,
                                 next=None, iuoc=None, pprop=None, nprop=None, pgaz=None, ngaz=None, F=None)

        return layer

class NerLocalFeatureWithoutMorphTagger(Retagger):
    """Generates features for a token based on its character makeup."""
    conf_param = ['settings']

    def __init__(self, settings, output_layer='ner_features', output_attributes=(), input_layers=('ner_features', 'words')):
        self.settings = settings
        self.output_layer = output_layer
        self.output_attributes = output_attributes
        self.input_layers = input_layers

    def _change_layer(self, text: Text, layers: MutableMapping[str, Layer], status: dict):
        ner_features_layer = layers[self.output_layer]
        words_layer = layers['words']
        for i, token in enumerate( words_layer ):
            # Token.
            ner_features_layer[i].w = token.text
            # Lowercased token.
            ner_features_layer[i].w1 = token.text.lower()
            # Token shape.
            ner_features_layer[i].shape = get_shape(token.text)
            # Token shape degenerated.
            ner_features_layer[i].shaped = degenerate(get_shape(token.text))
            
            # Prefixes (length between one to four).
            ner_features_layer[i].p1 = token.text[0] if len(token.text) >= 1 else None
            ner_features_layer[i].p2 = token.text[:2] if len(token.text) >= 2 else None
            ner_features_layer[i].p3 = token.text[:3] if len(token.text) >= 3 else None
            ner_features_layer[i].p4 = token.text[:4] if len(token.text) >= 4 else None

            # Suffixes (length between one to four).
            ner_features_layer[i].s1 = token.text[-1] if len(token.text) >= 1 else None
            ner_features_layer[i].s2 = token.text[-2:] if len(token.text) >= 2 else None
            ner_features_layer[i].s3 = token.text[-3:] if len(token.text) >= 3 else None
            ner_features_layer[i].s4 = token.text[-4:] if len(token.text) >= 4 else None
            
            # Two digits
            ner_features_layer[i].d2 = b(get_2d(token.text))
            # Four digits
            ner_features_layer[i].d4 = b(get_4d(token.text))
            # Digits and '-'.
            ner_features_layer[i].dndash = b(get_dand(token.text, '-'))
            # Digits and '/'.
            ner_features_layer[i].dnslash = b(get_dand(token.text, '/'))
            # Digits and ','.
            ner_features_layer[i].dncomma = b(get_dand(token.text, ','))
            # Digits and '.'.
            ner_features_layer[i].dndot = b(get_dand(token.text, '.'))
            # A uppercase letter followed by '.'
            ner_features_layer[i].up = b(get_capperiod(token.text))

            # An initial uppercase letter.
            ner_features_layer[i].iu = b(token.text and token.text[0].isupper())
            # All uppercase letters.
            ner_features_layer[i].au = b(token.text.isupper())
            # All lowercase letters.
            ner_features_layer[i].al = b(token.text.islower())
            # All digit letters.
            ner_features_layer[i].ad = b(token.text.isdigit())
            # All other (non-alphanumeric) letters.
            ner_features_layer[i].ao = b(get_all_other(token.text))
            # Alphanumeric token.
            ner_features_layer[i].aan = b(token.text.isalnum())

            # Contains an uppercase letter.
            ner_features_layer[i].cu = b(contains_upper(token.text))
            # Contains a lowercase letter.
            ner_features_layer[i].cl = b(contains_lower(token.text))
            # Contains a alphabet letter.
            ner_features_layer[i].ca = b(contains_alpha(token.text))
            # Contains a digit.
            ner_features_layer[i].cd = b(contains_digit(token.text))
            # Contains an apostrophe.
            ner_features_layer[i].cp = b(token.text.find("'") > -1)
            # Contains a dash.
            ner_features_layer[i].cds = b(token.text.find("-") > -1)
            # Contains a dot.
            ner_features_layer[i].cdt = b(token.text.find(".") > -1)
            # Contains a symbol.
            ner_features_layer[i].cs = b(contains_symbol(token.text))
            
            # Before, after dash
            ner_features_layer[i].bdash = split_char(token.text, '-')[0]
            ner_features_layer[i].adash = split_char(token.text, '-')[1]

            # Before, after dot
            ner_features_layer[i].bdot = split_char(token.text, '.')[0]
            ner_features_layer[i].adot = split_char(token.text, '.')[1]

            # Length
            ner_features_layer[i].len = str(len(token.text))
            
class NerBasicMorphFeatureTagger(Retagger):
    """Extracts features provided by the morphological analyser pyvabamorf. """
    conf_param = ['settings']

    def __init__(self, settings, input_layers = ('ner_features', 'words'), output_layer='ner_features',
                 output_attributes=("lem", "pos", "prop", "pref", "post", "case",
                                    "ending", "pun", "w", "w1", "shape", "shaped", "p1",
                                    "p2", "p3", "p4", "s1", "s2", "s3", "s4", "d2",
                                    "d4", "dndash", "dnslash", "dncomma", "dndot", "up", "iu", "au",
                                    "al", "ad", "ao", "aan", "cu", "cl", "ca", "cd",
                                    "cp", "cds", "cdt", "cs", "bdash", "adash",
                                    "bdot", "adot", "len", "fsnt", "lsnt", "gaz",
                                    "prew", "next", "iuoc", "pprop", "nprop", "pgaz",
                                    "ngaz", "F")):
        self.settings = settings
        self.output_layer = output_layer
        self.output_attributes = output_attributes
        self.input_layers = input_layers

    def _change_layer(self, text: Text, layers: MutableMapping[str, Layer], status: dict):
        ner_features_layer = layers[self.output_layer]
        words_layer = layers['words']
        for i, token in enumerate( words_layer ):
            LEM = '_'.join(token.root_tokens[0]) + ('+' + token.ending[0] if token.ending[0] else '')
            if not LEM:
                LEM = token.text
            LEM = get_lemma(LEM)
                
            ner_features_layer[i].lem = get_lemma(LEM)
            ner_features_layer[i].pos = get_pos(token.partofspeech)
            ner_features_layer[i].prop = b(is_prop(token.partofspeech))
            ner_features_layer[i].pref = get_word_parts(token.root_tokens[0])[0]
            ner_features_layer[i].post = get_word_parts(token.root_tokens[0])[1]
            ner_features_layer[i].case = get_case(token.form[0])
            ner_features_layer[i].ending = get_ending(token.ending)
            ner_features_layer[i].pun = b(get_pos(token.partofspeech)=="_Z_")
            
            # Prefixes (length between one to four).
            ner_features_layer[i].p1 = LEM[0] if len(LEM) >= 1 else None
            ner_features_layer[i].p2 = LEM[:2] if len(LEM) >= 2 else None
            ner_features_layer[i].p3 = LEM[:3] if len(LEM) >= 3 else None
            ner_features_layer[i].p4 = LEM[:4] if len(LEM) >= 4 else None

            # Suffixes (length between one to four).
            ner_features_layer[i].s1 = LEM[-1] if len(LEM) >= 1 else None
            ner_features_layer[i].s2 = LEM[-2:] if len(LEM) >= 2 else None
            ner_features_layer[i].s3 = LEM[-3:] if len(LEM) >= 3 else None
            ner_features_layer[i].s4 = LEM[-4:] if len(LEM) >= 4 else None
            
            # Before, after dash
            ner_features_layer[i].bdash = split_char(LEM, '-')[0]
            ner_features_layer[i].adash = split_char(LEM, '-')[1]

            # Before, after dot
            ner_features_layer[i].bdot = split_char(LEM, '.')[0]
            ner_features_layer[i].adot = split_char(LEM, '.')[1]

            # Length
            ner_features_layer[i].len = str(len(LEM))
    
class NerMorphNoLemmasFeatureTagger(Retagger):
    """Extracts features provided by the morphological analyser pyvabamorf. """
    conf_param = ['settings']

    def __init__(self, settings, input_layers = ('ner_features', 'words'), output_layer='ner_features',
                 output_attributes=("lem", "pos", "prop", "pref", "post", "case",
                                    "ending", "pun", "w", "w1", "shape", "shaped", "p1",
                                    "p2", "p3", "p4", "s1", "s2", "s3", "s4", "d2",
                                    "d4", "dndash", "dnslash", "dncomma", "dndot", "up", "iu", "au",
                                    "al", "ad", "ao", "aan", "cu", "cl", "ca", "cd",
                                    "cp", "cds", "cdt", "cs", "bdash", "adash",
                                    "bdot", "adot", "len", "fsnt", "lsnt", "gaz",
                                    "prew", "next", "iuoc", "pprop", "nprop", "pgaz",
                                    "ngaz", "F")):
        self.settings = settings
        self.output_layer = output_layer
        self.output_attributes = output_attributes
        self.input_layers = input_layers

    def _change_layer(self, text: Text, layers: MutableMapping[str, Layer], status: dict):
        ner_features_layer = layers[self.output_layer]
        words_layer = layers['words']
        for i, token in enumerate( words_layer ):
        
            ner_features_layer[i].pos = get_pos(token.partofspeech)
            ner_features_layer[i].prop = b(is_prop(token.partofspeech))
            ner_features_layer[i].case = get_case(token.form[0])
            ner_features_layer[i].pun = b(get_pos(token.partofspeech)=="_Z_")

class NerGazetteerFeatureTagger(Retagger):
    """Generates features indicating whether the token is present in a precompiled
    list of organisations, geographical locations or person names. For instance,
    if a token t occurs both in the list of person names (PER) and organisations (ORG),
    assign t['gaz'] = ['PER', 'ORG']. With the parameter look_ahead, it is possible to
    compose multi-token phrases for dictionary lookup. When look_ahead=N, phrases
    (t[i], ..., t[i+N]) will be composed. If the phrase matches the dictionary, each
    token will be assigned the corresponding value.
    """
    conf_param = ['settings', 'look_ahead', 'data']

    def __init__(self, settings, look_ahead=3, output_layer='ner_features', output_attributes=(),
                 input_layers=['ner_features']):
        self.settings = settings
        self.look_ahead = look_ahead
        self.output_layer = output_layer
        self.output_attributes = output_attributes
        self.input_layers = input_layers

        self.data = defaultdict(set)
        with codecs.open(settings.GAZETTEER_FILE, 'rb', encoding="utf8") as f:
            for ln in f:
                word, lbl = ln.strip().rsplit("\t", 1)
                self.data[word].add(lbl)

    def _change_layer(self, text: Text, layers: MutableMapping[str, Layer], status: dict):
            layer = layers[self.output_layer]
            layer.attributes += tuple(self.output_attributes)
            tokens = list(layer)
            look_ahead = self.look_ahead
            for i in range(len(tokens)):
                if tokens[i].ner_features.iu[0] is not None: # Only capitalised strings
                    for j in range(i + 1, i + 1 + look_ahead):
                        lemmas = []
                        for token in tokens[i:j]:
                            lemmas.append(token.text)
                        phrase = " ".join(lemmas)
                        if phrase.lower() in self.data:
                            labels = self.data[phrase.lower()]
                            for tok in tokens[i:j]:
                                tok.ner_features.gaz = labels