import os
import hydra
import torch

from easyeditor import BaseEditor
from easyeditor import KNHyperParams, FTHyperParams, KETrainingHparams,\
    ROMEHyperParams, MEMITHyperParams, MENDTrainingHparams, MENDHyperParams, \
    SERACTrainingHparams, SERACHparams, IKEHyperParams, FTApiHyperParams, LoRAHyperParams, \
    GraceHyperParams
from easyeditor import ZsreDataset, CounterFactDataset
from easyeditor import EditTrainer
from easyeditor.util.generate import generate_interactive
from shared_chatglm_pre import pre_chatglm_results_new
from shared_chatglm_old_pre import pre_chatglm_results
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
## Loading config from hparams/MEMIT/gpt2-xl.yaml


# 打开两个文件
with open('benchmark/new_term.txt', 'r', encoding='utf-8') as f1, open('benchmark/target_new.txt', 'r', encoding='utf-8') as f2, open('benchmark/old_term.txt', 'r', encoding='utf-8') as f3:
    # 初始化空列表来存储生成的语句
    post_chatglm_results = []
    old_post_chatglm_results = []
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
        hparams = ROMEHyperParams.from_hparams('hparams/ROME/chatglm2-6b.yaml')


        tokenizer = AutoTokenizer.from_pretrained('hugging_cache/chatglm2-6', trust_remote_code=True)
        #tokenizer.pad_token_id = tokenizer.eos_token_id
        editor = BaseEditor.from_hparams(hparams)   

        metrics, edited_model, _ = editor.edit(
                prompts=prompts,
                target_new=target_new,
                subject=subject,
                keep_original_weight=False,
                verbose=False
        )
        old_batch = tokenizer(old_generation_prompts, return_tensors='pt', padding=True, max_length=30)
        batch = tokenizer(prompts, return_tensors='pt', padding=True, max_length=30)
        post_edit_outputs = edited_model.generate(
                input_ids=batch['input_ids'].to('cuda'),
                attention_mask=batch['attention_mask'].to('cuda'),
                max_new_tokens=13
        )
        old_post_edit_outputs = edited_model.generate(
                input_ids=old_batch['input_ids'].to('cuda'),
                attention_mask=old_batch['attention_mask'].to('cuda'),
                max_new_tokens=30
        )
        # 过滤掉与generation_prompts相匹配的部分
        post_edit_outputs = [tokenizer.decode(x, skip_special_tokens=True) for x in post_edit_outputs.detach().cpu().numpy().tolist()]
        post_edit_outputs = [x.split(prompts[0])[-1].strip() for x in post_edit_outputs]
        old_post_edit_outputs = [tokenizer.decode(x, skip_special_tokens=True) for x in old_post_edit_outputs.detach().cpu().numpy().tolist()]
        old_post_edit_outputs = [x.split(prompts[0])[-1].strip() for x in old_post_edit_outputs]
        post_chatglm_results.extend(post_edit_outputs)
        old_post_chatglm_results.extend(old_post_edit_outputs)
        del editor
        del edited_model
        torch.cuda.empty_cache()


        
print('Post-Edit Outputs: ', post_chatglm_results)
print('Old Post-Edit Outputs: ', old_post_chatglm_results)


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

sentences_post_chatglm = post_chatglm_results
sentences_pre_chatglm = pre_chatglm_results_new
sentences_pre_chatglm_old = pre_chatglm_results
sentences_post_chatglm_old = old_post_chatglm_results

# Load model from HuggingFace Hub
tokenizer = AutoTokenizer.from_pretrained('hugging_cache/paraphrase-MiniLM-L6-v2')
model = AutoModel.from_pretrained('hugging_cache/paraphrase-MiniLM-L6-v2')

# Tokenize sentences
stand_encoded_input = tokenizer(stand_sentences, padding=True, truncation=True, return_tensors='pt')


encoded_input_pre_chatglm = tokenizer(sentences_pre_chatglm, padding=True, truncation=True, return_tensors='pt')
encoded_input_post_chatglm = tokenizer(sentences_post_chatglm, padding=True, truncation=True, return_tensors='pt')
encoded_input_pre_chatglm_old = tokenizer(sentences_pre_chatglm_old, padding=True, truncation=True, return_tensors='pt')
encoded_input_post_chatglm_old = tokenizer(sentences_post_chatglm_old, padding=True, truncation=True, return_tensors='pt')


with torch.no_grad():
    stand_model_output = model(**stand_encoded_input)



with torch.no_grad():
    model_output_post_chatglm = model(**encoded_input_post_chatglm)

with torch.no_grad():
    model_output_pre_chatglm = model(**encoded_input_pre_chatglm)

with torch.no_grad():
    model_output_pre_chatglm_old = model(**encoded_input_pre_chatglm_old)

with torch.no_grad():
    model_output_post_chatglm_old = model(**encoded_input_post_chatglm_old)


stand_embeddings = mean_pooling(stand_model_output, stand_encoded_input['attention_mask'])

embeddings_post_chatglm = mean_pooling(model_output_post_chatglm, encoded_input_post_chatglm['attention_mask'])
embeddings_pre_chatglm = mean_pooling(model_output_pre_chatglm, encoded_input_pre_chatglm['attention_mask'])
embeddings_post_chatglm_old = mean_pooling(model_output_post_chatglm_old, encoded_input_post_chatglm_old['attention_mask'])
embeddings_pre_chatglm_old = mean_pooling(model_output_pre_chatglm_old, encoded_input_pre_chatglm_old['attention_mask'])

similarities_post = []
similarities_pre = []
similarities_old = []

for i in range(100):
    # 计算余弦相似度
    sim_pre = cosine_similarity(embeddings_pre_chatglm[i].reshape(1, -1), stand_embeddings[i].reshape(1, -1))
    similarities_pre.append(sim_pre)

# 打印结果
for i, sim_pre in enumerate(similarities_pre):
    print("Before editing , Similarity of sentences {}: {:.4f}".format(i+1, sim_pre[0][0]))

total_similarity = sum(abs(sim_pre[0][0]) for sim_pre in similarities_pre)
average_similarity = total_similarity / len(similarities_pre)
print("Average similarity: {:.4f}".format(average_similarity))


for i in range(100):
    # 计算余弦相似度
    sim_post = cosine_similarity(embeddings_post_chatglm[i].reshape(1, -1), stand_embeddings[i].reshape(1, -1))
    similarities_post.append(sim_post)

# 打印结果
for i, sim_post in enumerate(similarities_post):
    print("After editing , Similarity of sentences {}: {:.4f}".format(i+1, sim_post[0][0]))

total_similarity = sum(abs(sim_post[0][0]) for sim_post in similarities_post)
average_similarity = total_similarity / len(similarities_post)
print("Average similarity: {:.4f}".format(average_similarity))

for i in range(100):
    # 计算余弦相似度
    sim_old = cosine_similarity(embeddings_post_chatglm_old[i].reshape(1, -1), embeddings_pre_chatglm_old[i].reshape(1, -1))
    similarities_old.append(sim_old)

# 打印结果
for i, sim_old in enumerate(similarities_old):
    print("Similarity of old sentences {}: {:.4f}".format(i+1, sim_old[0][0]))

total_similarity = sum(abs(sim_old[0][0]) for sim_old in similarities_old)
average_similarity = total_similarity / len(similarities_old)
print("Average similarity: {:.4f}".format(average_similarity))