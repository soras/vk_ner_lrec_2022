#
#  Trains and evaluates BERT-based NER model.
#
#  Assumes that BERT models have already been downloaded and placed 
#  into sub directories 'EstBERT', 'WikiBert-et' and 'est-roberta'.
#  Then the directory name of trainable model should be given as the 
#  command line argument of the script, e.g.
#
#     python  03_train_and_eval_bert_model.py  EstBERT
#
#  Normally, you should place model sub directories in the same 
#  directory as this script. 
#  Note also that model's directory name is used in names of log 
#  and result files. 
#  
#  Training and evaluation process
#
#  First, performs a grid search over parameter values listed in 
#  GRID_SEARCH_PARAMETERS. For each grid search configuration, 
#  trains the model on 'train' set for the given number epochs 
#  (default: 3), evaluates on the 'dev' set and saves F1-score. 
#  
#  After all grid search configurations have been evaluated, picks
#  the configuration that obtained the highest F1-score on the 'dev' 
#  set. Then, the model with that configuration is trained for 10 
#  epochs and the best model (based on F1 score on the 'dev' set) is 
#  saved for the final evaluation.
#
#  Finally, the best model is also evaluated on the 'test' set.
#  
#  The code used here for fine-tuning BERT is largely based on 
#  Tobias Sterbak's tutorial:
#  https://www.depends-on-the-definition.com/named-entity-recognition-with-bert/
#  (last checked: 2022-04-18)
#
#  Requirements (with python 3.7):
#     estnltk 1.6.9(.1)
#     transformers 4.0.0
#     tokenizers
#     datasets
#     nervaluate
#     seqeval
#     psutil
#     pytorch 1.7.0
#     keras
#     tqdm
#  ( For a complete list of requirements, see 'conda_environment.yml' )
#

import sys, os, os.path
import logging
import json
import copy
import shutil
import gc
import itertools

import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import AutoTokenizer, AutoConfig

from keras_preprocessing.sequence import pad_sequences

from seqeval.metrics import f1_score, accuracy_score
from seqeval.metrics import precision_score, recall_score

from tqdm import trange
from datetime import datetime

import psutil
from psutil._common import bytes2human

import transformers
from transformers import AutoModelForTokenClassification, AdamW

from bert_ner_tagger import BertNERTagger

from data_preprocessing import TokenizationPreprocessorFixed
from data_preprocessing import load_and_prepare_bert_data

from eval_utils import collect_and_preprocess_data
from eval_utils import eval_nertagger
from eval_utils import get_output_results_formatted
from eval_utils import write_out_results

# Bert's model
model_name_or_path_1 = None
model_name_s = None
if len(sys.argv) > 1:
    model_name_or_path_1 = sys.argv[1]
    assert os.path.exists(model_name_or_path_1) and \
           os.path.isdir(model_name_or_path_1), \
        f"(!) Unexpected Bert model directory: {model_name_or_path_1!r}"
    # Get directory name without full path
    if model_name_or_path_1.endswith(os.sep):
        model_name_or_path_1 = model_name_or_path_1.rstrip(os.sep)
    model_name_s = os.path.split(model_name_or_path_1)[1]
    model_name_s = model_name_s.lower()
else:
    raise Exception("(!) Missing input argument: Bert model directory.")

# Input and eval data
train_data_dir = os.path.join('data', 'train')
dev_data_dir   = os.path.join('data', 'dev')
test_data_dir  = os.path.join('data', 'test')
assert os.path.exists(train_data_dir), \
    f"(!) Unexpected training data directory: {train_data_dir!r}"
assert os.path.exists(train_data_dir), \
    f"(!) Unexpected dev data directory: {dev_data_dir!r}"
assert os.path.exists(test_data_dir), \
    f"(!) Unexpected test data directory: {test_data_dir!r}"
    
# Output dirs
output_model_dir = 'bert_models'
os.makedirs( output_model_dir, exist_ok=True )
results_dir = 'results'
os.makedirs( results_dir, exist_ok=True )

log = logging.getLogger(__name__)
script_name = sys.argv[0]
script_name = script_name.replace( '_bert_', f'_{model_name_s}_' )
f_handler = logging.FileHandler(script_name+'.log', mode='w', encoding='utf-8')
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

log.info(f'torch version:           {torch.__version__}')
log.info(f'transformers version:    {transformers.__version__}')
log.info(f'torch.cuda.is_available: {torch.cuda.is_available()}')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()

if torch.cuda.is_available():
    log.info( torch.cuda.get_device_name(0) )

GRID_SEARCH_PARAMETERS = {
  "per_gpu_batch_size": [8, 16, 32],
  "learning_rate": [5e-5, 3e-5, 1e-5],
  "max_epochs":[3]
}

def yield_grid_search_configurations( logger ):
    # How many grid search combinations are there at total ?
    total_confs = 1
    all_keys = sorted( list(GRID_SEARCH_PARAMETERS.keys()) )
    all_values = []
    for k in all_keys:
        params = len(GRID_SEARCH_PARAMETERS[k])
        all_values.append( GRID_SEARCH_PARAMETERS[k] )
        total_confs *= params
    logger.info(f' Initializing the grid search training. ')
    logger.info(f' Total configurations to be tested in the grid search:    {total_confs}')
    assert len(all_keys) > 0
    # Generate all combinations of parameter values
    all_combinations = list( itertools.product( *all_values ) )
    assert len(all_combinations) == total_confs
    # Yield configurations one by one
    for configuration in all_combinations:
        assert len(configuration) == len(all_keys)
        conf_dict = dict()
        for index, key in enumerate( all_keys ):
            conf_dict[key] = configuration[index]
        yield conf_dict
    # Yield an empty dict as last. 
    # This is a signal to take the best configuration 
    # and retrain a model based on it
    yield dict()


#skip_configs = [0,1,3,4,5,6,7]
skip_configs = []
overall_start_time = datetime.now()
configurations_tested = 0
conf_results = []
is_last_model = False
for CONF in yield_grid_search_configurations( log ):
    if len(CONF.keys()) == 0:
        # The signal for the last iteration.
        # Take the best configuration so far,
        # and retrain a model based on it
        assert len(conf_results) > 0
        (best_f1_score, best_conf_id) = max( [(res[1], res[0]) for res in conf_results] )
        for conf_id, model_f1_score, CONF_X, model_name_x, model_path_x in conf_results:
            if conf_id == best_conf_id:
                CONF = CONF_X
                log.info(f'The grid search completed.')
                log.info(f'The best model was {model_name_x!r} with dev set F1: {model_f1_score}')
                log.info(f'Retraining the best model with 10 epochs.')
                CONF["max_epochs"] = 10
                is_last_model = True
                break
    
    log.info(f' Starting a new grid search iteration ({configurations_tested+1}) ...')

    if len(skip_configs) > 0 and configurations_tested in skip_configs:
        log.info(f' Skipping configuration ({configurations_tested+1}) ...')
        configurations_tested += 1
        continue

    # Here we fix some configurations. 
    # Note, that Bert supports sequences of up to 512 tokens.
    MAX_LEN = 75
    bs = CONF["per_gpu_batch_size"]

    log.info(f' Model params:  batch_size={CONF["per_gpu_batch_size"]}, learning_rate={CONF["learning_rate"]}, max_epochs={CONF["max_epochs"]}')

    log.info(f' Max sequence length: {MAX_LEN}')

    # ====================================================
    #   Prepare the sentences and labels
    # ====================================================

    # Fetch Bert tokenizer model
    assert os.path.exists(model_name_or_path_1), \
        f"(!) Unexpected Bert model directory: {model_name_or_path_1!r}"
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path_1, do_lower_case=False, use_fast=False)

    prev_saved_models = []
    validation_f1_scores = []
    max_f1 = 0
    max_f1_model_path = None
    # ====================================================
    #   Prepare data for training
    # ====================================================
    # Data cut (for fast smoke testing)
    data_cut = None
    
    # Load & tokenize data with bert
    tokenized_texts, token_labels, tag2idx, idx2tag, tag_values = \
        load_and_prepare_bert_data( train_data_dir, tokenizer, log, data_file_suffix=None, cut=data_cut )

    # Next, we cut and pad the token and label sequences to our desired length.
    all_input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts],
                               maxlen=MAX_LEN, dtype="long", value=0.0,
                               truncating="post", padding="post")
    all_tags = pad_sequences([[tag2idx.get(l) for l in lab] for lab in token_labels],
                              maxlen=MAX_LEN, value=tag2idx["PAD"], padding="post",
                              dtype="long", truncating="post")

    # The Bert model supports something called attention_mask, which is similar to the masking 
    # in keras. So here we create the mask to ignore the padded elements in the sequences.
    all_attention_masks = [[float(i != 0.0) for i in ii] for ii in all_input_ids]

    # Since we’re operating in pytorch, we have to convert the dataset to torch tensors.
    tr_inputs = torch.tensor(all_input_ids)
    tr_tags = torch.tensor(all_tags)
    tr_masks = torch.tensor(all_attention_masks)
  
    # The last step is to define the dataloaders. We shuffle the data at training time 
    # with the RandomSampler and at test time we just pass them sequentially with 
    # the SequentialSampler.
    train_data = TensorDataset(tr_inputs, tr_masks, tr_tags)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=bs)

    # ====================================================
    #   Prepare data for validation
    # ====================================================
    
    # Load & tokenize data
    tokenized_texts_dev, token_labels_dev, tag2idx_dev, idx2tag_dev, tag_values_dev = \
        load_and_prepare_bert_data( dev_data_dir, tokenizer, log, data_file_suffix=None, cut=data_cut )

    # Next, we cut and pad the token and label sequences to our desired length.
    all_input_ids_dev = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts_dev],
                                       maxlen=MAX_LEN, dtype="long", value=0.0,
                                       truncating="post", padding="post")
    all_tags_dev = pad_sequences([[tag2idx_dev.get(l) for l in lab] for lab in token_labels_dev],
                                   maxlen=MAX_LEN, value=tag2idx_dev["PAD"], padding="post",
                                   dtype="long", truncating="post")

    # The Bert model supports something called attention_mask, which is similar to the masking 
    # in keras. So here we create the mask to ignore the padded elements in the sequences.
    all_attention_masks_dev = [[float(i != 0.0) for i in ii] for ii in all_input_ids_dev]
    
    # Since we’re operating in pytorch, we have to convert the dataset to torch tensors.
    val_inputs = torch.tensor(all_input_ids_dev)
    val_tags = torch.tensor(all_tags_dev)
    val_masks = torch.tensor(all_attention_masks_dev)
    
    # The last step is to define the dataloaders. We shuffle the data at training time 
    # with the RandomSampler and at test time we just pass them sequentially with 
    # the SequentialSampler.
    val_data = TensorDataset(val_inputs, val_masks, val_tags)
    val_sampler = SequentialSampler(val_data)
    val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=bs)

    # ====================================================
    #   Setup the Bert model for finetuning
    # ====================================================

    # Remove 'PAD' from labels
    pad_idx = tag2idx.get('PAD')
    tag2idx_wo_pad = copy.deepcopy(tag2idx)
    del tag2idx_wo_pad['PAD']
    idx2tag_wo_pad = copy.deepcopy(idx2tag)
    del idx2tag_wo_pad[pad_idx]

    # We make a new config for BertForTokenClassification
    # This seems to be the best way to propagate information
    # about labels to the model

    bert_config = AutoConfig.from_pretrained(
        model_name_or_path_1,
        id2label=idx2tag_wo_pad, 
        label2id=tag2idx_wo_pad, 
        output_attentions = False,
        output_hidden_states = False
    )

    model = AutoModelForTokenClassification.from_pretrained(
        model_name_or_path_1,
        config=bert_config,
    )

    # Assert that labels were correctly saved to config
    assert model.config.to_dict()['id2label'] == idx2tag_wo_pad
    assert model.config.to_dict()['label2id'] == tag2idx_wo_pad

    # Now we have to pass the model parameters to the GPU.
    if torch.cuda.is_available():
        model.cuda();

    # Before we can start the fine-tuning process, we have to setup the optimizer 
    # and add the parameters it should update. A common choice is the AdamW optimizer. 
    # We also add some weight_decay as regularization to the main weight matrices. If 
    # you have limited resources, you can also try to just train the linear classifier 
    # on top of BERT and keep all other weights fixed. This will still give you a 
    # good performance.

    FULL_FINETUNING = True
    if FULL_FINETUNING:
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'gamma', 'beta']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.0}
        ]
    else:
        param_optimizer = list(model.classifier.named_parameters())
        optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]

    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=CONF["learning_rate"],
        eps=1e-8
    )

    # We also add a scheduler to linearly reduce the learning rate throughout the epochs.
    from transformers import get_linear_schedule_with_warmup

    epochs = CONF["max_epochs"]
    max_grad_norm = 1.0

    # Total number of training steps is number of batches * number of epochs.
    total_steps = len(train_dataloader) * epochs

    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )

    log.info(f' Epochs:          {epochs}')
    log.info(f' Optimizer:       {optimizer}')

    # ====================================================
    #   Fit BERT for named entity recognition
    # ====================================================
    # Check & report memory
    virtual_memory = psutil.virtual_memory()
    human_readable = bytes2human(virtual_memory.available)
    log.info(f'Memory available for fitting: {human_readable} ')

    # Finally, we can finetune the model. A few epochs should be enougth. The paper suggest 3-4 epochs.

    ## Store the average loss after each epoch so we can plot them.
    loss_values = []

    epoch = 0
    for epoch in range(1, epochs+1):
        log.info('')
        log.info(f'Starting new epoch ({epoch})...')
        epoch_start_time = datetime.now()

        virtual_memory = psutil.virtual_memory()
        human_readable = bytes2human(virtual_memory.available)
        log.info(f' Memory available for fitting: {human_readable} ')
        # ========================================
        #               Training
        # ========================================
        # Perform one full pass over the training set.

        # Put the model into training mode.
        model.train()
        # Reset the total loss for this epoch.
        total_loss = 0

        # Training loop
        for step, batch in enumerate(train_dataloader):
            start_time = datetime.now()
            # add batch to gpu
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch
            # Always clear any previously calculated gradients before performing a backward pass.
            model.zero_grad()
            #
            # NB! Encountered some weird data type errors on Windows:
            #
            # >> print(b_input_ids.dtype, b_input_mask.dtype, b_labels.dtype)
            # torch.int32 torch.float32 torch.int32
            # Solution:
            #    https://github.com/huggingface/transformers/issues/2952#issuecomment-630851378
            #
            b_input_ids = (b_input_ids).clone().detach().to(torch.long)
            b_labels    = (b_labels).clone().detach().to(torch.long)
            # forward pass
            # This will return the loss (rather than the model output)
            # because we have provided the `labels`.
            outputs = model(b_input_ids, token_type_ids=None,
                            attention_mask=b_input_mask, labels=b_labels)
            # get the loss
            loss = outputs[0]
            # Perform a backward pass to calculate the gradients.
            loss.backward()
            # track train loss
            total_loss += loss.item()
            # Clip the norm of the gradient
            # This is to help prevent the "exploding gradients" problem.
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_grad_norm)
            # update parameters
            optimizer.step()
            # Update the learning rate.
            scheduler.step()
            # Update msg
            log.info(f'Training batch {step+1}/{len(train_dataloader)} completed in {datetime.now()-start_time}.')

        # Calculate the average loss over the training data.
        avg_train_loss = total_loss / len(train_dataloader)
        log.info("Average train loss: {}".format(avg_train_loss))

        # Store the loss value for plotting the learning curve.
        loss_values.append(avg_train_loss)

        # Perform garbage collection
        gc.collect()

        # Put the model into evaluation mode
        model.eval()
        
        # Validate the model
        # Reset the validation loss for this epoch.
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0
        predictions , true_labels = [], []
        for step, batch in enumerate( val_dataloader ):
            start_time = datetime.now()
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch

            # Telling the model not to compute or store gradients,
            # saving memory and speeding up validation
            with torch.no_grad():
                #
                # NB! Encountered some weird data type errors on Windows:
                #
                # >> print(b_input_ids.dtype, b_input_mask.dtype, b_labels.dtype)
                # torch.int32 torch.float32 torch.int32
                # Solution:
                #    https://github.com/huggingface/transformers/issues/2952#issuecomment-630851378
                #
                b_input_ids = (b_input_ids).clone().detach().to(torch.long)
                b_labels    = (b_labels).clone().detach().to(torch.long)
                # Forward pass, calculate logit predictions.
                # This will return the logits rather than the loss because we have not provided labels.
                outputs = model(b_input_ids, token_type_ids=None,
                                attention_mask=b_input_mask, labels=b_labels)
            # Move logits and labels to CPU
            logits = outputs[1].detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            # Calculate the accuracy for this batch of test sentences.
            eval_loss += outputs[0].mean().item()
            predictions.extend( [list(p) for p in np.argmax(logits, axis=2)] )
            true_labels.extend( label_ids )
            log.info(f'Validation batch {step+1}/{len(val_dataloader)} completed in {datetime.now()-start_time}.')

        # Note: this is token-level evaluation, which is not very strict
        # After the last epoch, we'll also do a strict phrase level evalution

        eval_loss = eval_loss / len(val_dataloader)
        log.info("Validation loss: {}".format(eval_loss))
        pred_tags = [tag_values_dev[p_i] for p, l in zip(predictions, true_labels)
                                     for p_i, l_i in zip(p, l) if tag_values_dev[l_i] != "PAD"]
        valid_tags = [tag_values_dev[l_i] for l in true_labels
                                      for l_i in l if tag_values_dev[l_i] != "PAD"]
        log.info("Validation Accuracy: {}".format(accuracy_score(pred_tags, valid_tags)))
        
        # Convert to lists of lists
        if pred_tags and isinstance(pred_tags[0], str):
            pred_tags = [pred_tags]
        if valid_tags and isinstance(valid_tags[0], str):
            valid_tags = [valid_tags]
        
        precision = precision_score( pred_tags, valid_tags )
        recall    = recall_score( pred_tags, valid_tags )
        cur_f1    = f1_score( pred_tags, valid_tags )
        log.info("Validation Precision: {}".format( precision ))
        log.info("Validation Recall: {}".format( recall ))
        log.info("Validation F1-Score: {}".format( cur_f1 ))
        validation_f1_scores.append( cur_f1 )

        log.info(f'Epoch completed in {datetime.now()-epoch_start_time}.')
        
        if is_last_model and cur_f1 > max_f1:
            log.info(f" F1-score improved. Saving the model.")
            conf_name  = f'{configurations_tested+1}_bs{CONF["per_gpu_batch_size"]}_lr{CONF["learning_rate"]}_ep{epoch}'
            model_name = f"model_{model_name_s}_{conf_name}"
            model_path = os.path.join( output_model_dir, model_name )
            if not os.path.exists( model_path ):
                os.makedirs( model_path )
            model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
            model_to_save.save_pretrained( save_directory=model_path )
            tokenizer.save_pretrained( model_path )
            max_f1 = cur_f1
            max_f1_model_path = model_path
            # Remove previous model (keep only 1 model per best configuration)
            if len( prev_saved_models ) > 0:
                for prev_model in prev_saved_models:
                    if os.path.exists( prev_model ):
                        log.info(f' Removing previous best model {prev_model!r}. ')
                        shutil.rmtree( prev_model )
            prev_saved_models.append( model_path )

    # If all epochs of a configuration are completed, save the model.
    # Except for the final evaluation: saving is already done after the validation,
    # no need to save then ...
    
    conf_name   = f'{configurations_tested+1}_bs{CONF["per_gpu_batch_size"]}_lr{CONF["learning_rate"]}_ep{epoch}'
    model_name = f"model_{model_name_s}_{conf_name}"
    model_path = os.path.join( output_model_dir, model_name )
    if not is_last_model:
        if not os.path.exists( model_path ):
            os.makedirs( model_path )
        log.info(f" Saving the model. ")
        model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
        model_to_save.save_pretrained( save_directory=model_path )
        tokenizer.save_pretrained( model_path )
    
    log.info(f" Token level validation f1-scores of {model_name}: {validation_f1_scores}")

    # Perform garbage collection
    gc.collect()

    # All the epochs for this conf have been exhausted: evaluate the model on dev set
    # Use BertNERTagger to annotate exact NE phrases & nervaluate.Evaluator to get 
    # strict evaluation results
    if is_last_model:
        # Evaluate the best model only
        assert max_f1_model_path is not None and os.path.exists( max_f1_model_path )
        model_path = max_f1_model_path
        
    bert_ner_tagger_phrases = \
        BertNERTagger(model_path, model_path, token_level=False, do_lower_case=False, use_fast=False)
    dev_texts = collect_and_preprocess_data( dev_data_dir, TokenizationPreprocessorFixed(), log )
    dev_results = eval_nertagger( bert_ner_tagger_phrases, dev_data_dir, dev_texts, log, 
                                  auto_layer=bert_ner_tagger_phrases.output_layer )
    pretty_results = get_output_results_formatted( dev_results )
    log.info( pretty_results )
    model_f1_score = dev_results['strict']['f1']
    write_out_results( dev_results, model_name, 'dev', results_dir )
    conf_results.append( [configurations_tested, model_f1_score, CONF, model_name, model_path] )
    
    if is_last_model:
        # if this is the last model, evaluate it also on the test set
        test_texts = collect_and_preprocess_data( test_data_dir, TokenizationPreprocessorFixed(), log )
        test_results = eval_nertagger( bert_ner_tagger_phrases, test_data_dir, test_texts, log, 
                                       auto_layer=bert_ner_tagger_phrases.output_layer )
        pretty_results = get_output_results_formatted( test_results )
        log.info( pretty_results )
        model_f1_score = test_results['strict']['f1']
        write_out_results( test_results, model_name, 'test', results_dir )
    
    # All epochs have been exhausted: delete the model
    if model is not None:
        del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    bert_ner_tagger_phrases = None
    
    # Perform garbage collection at the end of the iteration
    gc.collect()
    
    configurations_tested += 1
    
    #break

log.info(f'Total time elapsed: {datetime.now()-overall_start_time}.')

