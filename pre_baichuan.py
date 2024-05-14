import os
import hydra
import torch

from easyeditor import BaseEditor
from easyeditor import KNHyperParams, FTHyperParams, KETrainingHparams,\
    ROMEHyperParams
import json
import argparse
from copy import deepcopy
from utils import build_prompt, llama_2_response
from easyeditor import ZsreDataset, CounterFactDataset
from easyeditor import EditTrainer
from easyeditor.util.generate import generate_interactive
from shared_llama_pre import pre_llama_results_new
from shared_llama_old_pre import pre_llama_results
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics.pairwise import cosine_similarity
from transformers import LlamaTokenizer, LlamaForCausalLM
## Loading config from hparams/MEMIT/gpt2-xl.yaml

def test_open_source(word, task, model, tokenizer):  
    with open(f'benchmark/{task}_clean.json', 'r', encoding='utf-8') as f,\
        open(f'test/results/{task}_baichuan_BASE.json', 'a', encoding='utf-8') as w:
        for line in f:
            line = json.loads(line.strip())
            if "term" in line and line["term"] == word:
                for p in range(3):
                    lsb = deepcopy(line)
                    lsb['prompt_id'] = p
                    system, prompt, _ = build_prompt(line, task, p)
                    prompt = [system, prompt]                
                    lsb['response'] = llama_2_response(prompt, model, tokenizer)
                    w.write(json.dumps(lsb, sort_keys=True, indent=0, ensure_ascii=False).replace("\n", " ") + "\n")

tokenizer = AutoTokenizer.from_pretrained('hugging_cache/Baichuan2-7B-Chat', trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained('hugging_cache/Baichuan2-7B-Chat', trust_remote_code=True).to('cuda')
# 打开两个文件
with open('benchmark/new_term.txt', 'r', encoding='utf-8') as f1:
    
    # 循环100次
    for _ in range(100):
        # 从每个文件中读取一行
        line1 = f1.readline().strip()
        test_open_source(line1, 'CSJ', model, tokenizer)
        test_open_source(line1, 'COMA', model, tokenizer)
        test_open_source(line1, 'COST', model, tokenizer)
        # del model
        # del tokenizer
        torch.cuda.empty_cache()


        

