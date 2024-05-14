import os
import hydra
import torch
import json
from copy import deepcopy

from easyeditor import BaseEditor
from easyeditor import KNHyperParams, FTHyperParams, KETrainingHparams,\
    ROMEHyperParams, MEMITHyperParams, MENDTrainingHparams, MENDHyperParams, \
    SERACTrainingHparams, SERACHparams, IKEHyperParams, FTApiHyperParams, LoRAHyperParams, \
    GraceHyperParams
from easyeditor import ZsreDataset, CounterFactDataset
from utils import build_prompt, qwen_response
from easyeditor import EditTrainer
from easyeditor.util.generate import generate_interactive
from shared_chatglm_pre import pre_chatglm_results_new
from shared_chatglm_old_pre import pre_chatglm_results
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
## Loading config from hparams/MEMIT/gpt2-xl.yaml

def test_open_source(word, task, model, tokenizer):  
    with open(f'benchmark/{task}_clean.json', 'r', encoding='utf-8') as f,\
        open(f'test/results/{task}_qwen_BASE.json', 'a', encoding='utf-8') as w:
        for line in f:
            line = json.loads(line.strip())
            if "term" in line and line["term"] == word:
                for p in range(3):
                    lsb = deepcopy(line)
                    lsb['prompt_id'] = p
                    system, prompt, _ = build_prompt(line, task, p)
                    prompt = [system, prompt]                
                    lsb['response'] = qwen_response(prompt, model, tokenizer)
                    w.write(json.dumps(lsb, sort_keys=True, indent=0, ensure_ascii=False).replace("\n", " ") + "\n")


def post_test_open_source(pre_prompt, word, task, model, tokenizer):  
    with open(f'benchmark/{task}_clean.json', 'r', encoding='utf-8') as f,\
        open(f'test/results/post_{task}_qwen_BASE.json', 'a', encoding='utf-8') as w:
        for line in f:
            line = json.loads(line.strip())
            if "term" in line and line["term"] == word:
                for p in range(3):
                    lsb = deepcopy(line)
                    lsb['prompt_id'] = p
                    system, prompt, _ = build_prompt(line, task, p)
                    prompt = str(pre_prompt) + prompt
                    prompt = [system, prompt]
                
                    lsb['response'] = qwen_response(prompt, model, tokenizer)
                    w.write(json.dumps(lsb, sort_keys=True, indent=0, ensure_ascii=False).replace("\n", " ") + "\n")
            


# 打开两个文件
with open('benchmark/new_term.txt', 'r', encoding='utf-8') as f1, open('benchmark/target_new.txt', 'r', encoding='utf-8') as f2, open('benchmark/old_term.txt', 'r', encoding='utf-8') as f3:
    # 初始化空列表来存储生成的语句
    post_qwen_results = []
    old_post_qwen_results = []
    stand_sentences = []
    
    # 循环100次
    for _ in range(100):
        # 从每个文件中读取一行
        prompts = []
        target_new = []
        subject = []
        old_generation_prompts = []  
        line1 = f1.readline().strip()
        subject.append(line1)
        sentence = f"What does {line1} mean?"
        prompts.append(sentence)
        line2 = f2.readline().strip()
        target_new.append(line2)
        stand_sentences.extend(target_new)
        line3 = f3.readline().strip()
        old_generation_prompts.append(line3)   
        ## Construct Language Model Editor
        hparams = ROMEHyperParams.from_hparams('hparams/ROME/qwen-7b.yaml')


        tokenizer = AutoTokenizer.from_pretrained('hugging_cache/qwen-7b', trust_remote_code=True, pad_token='<|endoftext|>')
        editor = BaseEditor.from_hparams(hparams)   

        metrics, edited_model, _ = editor.edit(
                prompts=prompts,
                target_new=target_new,
                subject=subject,
                keep_original_weight=False,
                verbose=False
        )
        test_open_source(line1, 'CSJ', edited_model, tokenizer)
        test_open_source(line1, 'COMA', edited_model, tokenizer)
        test_open_source(line1, 'COST', edited_model, tokenizer)
        old_batch = tokenizer(old_generation_prompts, return_tensors='pt', padding=True, max_length=30)
        batch = tokenizer(prompts, return_tensors='pt', padding=True, max_length=30)
        post_edit_outputs = edited_model.generate(
                input_ids=batch['input_ids'].to('cuda'),
                attention_mask=batch['attention_mask'].to('cuda'),
                max_new_tokens=13
        )
        # old_post_edit_outputs = edited_model.generate(
        #         input_ids=old_batch['input_ids'].to('cuda'),
        #         attention_mask=old_batch['attention_mask'].to('cuda'),
        #         max_new_tokens=30
        # )
        # 过滤掉与generation_prompts相匹配的部分
        post_edit_outputs = [tokenizer.decode(x, skip_special_tokens=True) for x in post_edit_outputs.detach().cpu().numpy().tolist()]
        post_test_open_source(post_edit_outputs, line1, 'CSJ', edited_model, tokenizer)
        post_test_open_source(post_edit_outputs, line1, 'COMA', edited_model, tokenizer)
        post_test_open_source(post_edit_outputs, line1, 'COST', edited_model, tokenizer)
        post_edit_outputs = [x.split(prompts[0])[-1].strip() for x in post_edit_outputs]
        # old_post_edit_outputs = [tokenizer.decode(x, skip_special_tokens=True) for x in old_post_edit_outputs.detach().cpu().numpy().tolist()]
        # old_post_edit_outputs = [x.split(prompts[0])[-1].strip() for x in old_post_edit_outputs]
        post_qwen_results.extend(post_edit_outputs)
        # old_post_qwen_results.extend(old_post_edit_outputs)
        del editor
        del edited_model
        torch.cuda.empty_cache()


        
print('Post-Edit Outputs: ', post_qwen_results)
print('Old Post-Edit Outputs: ', old_post_qwen_results)


