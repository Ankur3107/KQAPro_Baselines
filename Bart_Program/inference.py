import sys
import os
import json
import pickle
import argparse
import numpy as np
from nltk import word_tokenize
from collections import Counter
from itertools import chain
from tqdm import tqdm
import re
import json
import pickle
import torch
from utils.misc import invert_dict
from transformers import BartConfig, BartForConditionalGeneration, BartTokenizer
import shutil
from datetime import date
from utils.misc import MetricLogger, seed_everything, ProgressBar
from utils.load_kb import DataForSPARQL
from Bart_Program.data import DataLoader
import logging
import time
from utils.lr_scheduler import get_linear_schedule_with_warmup
import re
from Bart_Program.executor_rule import RuleExecutor
import warnings
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s')
logFormatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
rootLogger = logging.getLogger()
warnings.simplefilter("ignore") # hide warnings that caused by invalid sparql query
from Bart_Program.predict import *
import spacy


def load_trained_models(model_name_or_path, kb_json_file, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """"Load Trained Model for inference"""
    
    tokenizer = BartTokenizer.from_pretrained(model_name_or_path)
    vocab = {
        'answer_token_to_idx': {} # load if you want to test accuracy
    }
    kb = DataForSPARQL(kb_json_file)
    rule_executor = RuleExecutor(vocab, kb_json_file)

    model = BartForConditionalGeneration.from_pretrained(model_name_or_path)
    model = model.to(device)
    model.eval()

    return tokenizer, model, vocab, rule_executor, kb

def collate(batch):
    batch = list(zip(*batch))
    source_ids = torch.stack(batch[0])
    source_mask = torch.stack(batch[1])
    choices = torch.stack(batch[2])
    if batch[-1][0] is None:
        target_ids, answer = None, None
    else:
        target_ids = torch.stack(batch[3])
        answer = torch.cat(batch[4])
    return source_ids, source_mask, choices, target_ids, answer

class Dataset(torch.utils.data.Dataset):
    """"Build torch dataset class"""
    def __init__(self, inputs):
        self.source_ids, self.source_mask, self.target_ids, self.choices, self.answers = inputs
        self.is_test = len(self.answers)==0


    def __getitem__(self, index):
        source_ids = torch.LongTensor(self.source_ids[index])
        source_mask = torch.LongTensor(self.source_mask[index])
        choices = torch.LongTensor(self.choices[index])
        if self.is_test:
            target_ids = None
            answer = None
        else:
            target_ids = torch.LongTensor(self.target_ids[index])
            answer = torch.LongTensor([self.answers[index]])
        return source_ids, source_mask, choices, target_ids, answer


    def __len__(self):
        return len(self.source_ids)


class DataLoader(torch.utils.data.DataLoader):
    """Build dataload class"""
    def __init__(self, vocab, inputs, batch_size, training=False):
        if training:
            print('#vocab of answer: %d' % (len(vocab['answer_token_to_idx'])))
        
        dataset = Dataset(inputs)
        # np.shuffle(dataset)
        # dataset = dataset[:(int)(len(dataset) / 10)]
        super().__init__(
            dataset, 
            batch_size=batch_size,
            shuffle=training,
            collate_fn=collate, 
            )
        self.vocab = vocab


def get_predict_dataloader(query, tokenizer, vocab, max_seq_length=32):
    """Get processed dataloader for inference"""

    questions = [query]
    input_ids = tokenizer.batch_encode_plus(questions, max_length = max_seq_length, pad_to_max_length = True, truncation = True)
    source_ids = np.array(input_ids['input_ids'], dtype = np.int32)
    source_mask = np.array(input_ids['attention_mask'], dtype = np.int32)
    choices = np.array([12], dtype = np.int32) # Just doing for not resolving error
    answers = np.array([], dtype = np.int32)
    target_ids = np.array([], dtype = np.int32)
    inputs = [source_ids, source_mask, target_ids, choices, answers]
    predict_loader = DataLoader(vocab, inputs, batch_size=1)
    return predict_loader


def get_prediction(kb, model, data_loader, device, tokenizer):
  """Get prediction"""
  count, correct = 0, 0
  pattern = re.compile(r'(.*?)\((.*?)\)')
  with torch.no_grad():
      all_outputs = []
      for batch in tqdm(data_loader, total=len(data_loader)):
          batch = batch[:3]
          source_ids, source_mask, choices = [x.to(device) for x in batch]
          outputs = model.generate(
              input_ids=source_ids,
              max_length = 500,
          )

          all_outputs.extend(outputs.cpu().numpy())
          break
      
      outputs = [tokenizer.decode(output_id, skip_special_tokens = True, clean_up_tokenization_spaces = True) for output_id in all_outputs]
  return outputs


class Inference:
    def __init__(self, model_name_or_path, kb_json_file,  device ='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model_name_or_path = model_name_or_path
        self.kb_json_file = kb_json_file
        self.device = device
        self.tokenizer, self.model, self.vocab, self.rule_executor, self.kb = load_trained_models(model_name_or_path, kb_json_file, device)

    def run(self, query, max_seq_length=32):
        predict_loader = get_predict_dataloader(query, self.tokenizer, self.vocab, max_seq_length=max_seq_length)
        outputs = get_prediction(self.kb, self.model, predict_loader, self.device, self.tokenizer)
        return outputs

class EntityFinder():
  def __init__(self, spacy_model_name="en_core_web_md"):
    self.nlp = spacy.load(spacy_model_name)
    self.nlp.add_pipe("entityLinker", last=True)


  def entity_collection_postprocessing(self, entities):
    entity_dict = {}
    entities = entities.entities
    for entity in entities:
      entity_dict[entity.get_span().text.lower()] = entity.get_url()
    return entity_dict


  def link_program_and_entity(self, program_dict, entity_dict):
    program_dict = program_dict.copy()
    to_finds = program_dict['Find']
    to_find_with_entities = [] 
    for to_find in to_finds:
      if to_find.lower() in entity_dict:
        to_find_with_entities.append(to_find+"#"+entity_dict[to_find.lower()])
      else:
        to_find_with_entities.append(to_find+"#None")
    program_dict['Find'] = to_find_with_entities
    return program_dict

  def search(self, text, program_dict):
    doc = self.nlp(text)
    all_linked_entities = doc._.linkedEntities

    entity_dict = self.entity_collection_postprocessing(all_linked_entities)
    program_dict = self.link_program_and_entity(program_dict, entity_dict)
    return program_dict


class InferenceWithEntity:
    def __init__(self, model_name_or_path, kb_json_file,  device ='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model_name_or_path = model_name_or_path
        self.kb_json_file = kb_json_file
        self.device = device
        self.tokenizer, self.model, self.vocab, self.rule_executor, self.kb = load_trained_models(model_name_or_path, kb_json_file, device)
        self.entity_finder = EntityFinder()


    def program_formatting(self, program):
        pattern = re.compile(r'(.*?)\((.*?)\)')
        chunks = program.split('<b>')

        program_dict = {}
        for chunk in chunks:
            # print(chunk)
            res = pattern.findall(chunk)
            # print(res)
            if len(res) == 0:
                continue
            res = res[0]
            func, inputs = res[0], res[1]
            if inputs == '':
                inputs = []
            else:
                inputs = inputs.split('<c>')

            program_dict[func] = inputs
        return program_dict
    
    def run(self, query, max_seq_length=32):
        predict_loader = get_predict_dataloader(query, self.tokenizer, self.vocab, max_seq_length=max_seq_length)
        outputs = get_prediction(self.kb, self.model, predict_loader, self.device, self.tokenizer)
        program_dict = self.program_formatting(outputs[0])
        program_with_entity_dict = self.entity_finder.search(query, program_dict)
        return program_with_entity_dict