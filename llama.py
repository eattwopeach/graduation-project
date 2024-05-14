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
from shared_llama_pre import pre_llama_results_new
from shared_llama_old_pre import pre_llama_results
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from transformers import LlamaTokenizer, LlamaForCausalLM
## Loading config from hparams/MEMIT/gpt2-xl.yaml

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


# 打开两个文件
with open('benchmark/new_term.txt', 'r', encoding='utf-8') as f1, open('benchmark/target_new.txt', 'r', encoding='utf-8') as f2:
    # 初始化空列表来存储生成的语句
    post_llama_results = []
    old_post_llama_results = []
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
        # line3 = f3.readline().strip()
        # old_generation_prompts.append(line3)   
        ## Construct Language Model Editor
        hparams = ROMEHyperParams.from_hparams('hparams/ROME/llama-7b.yaml')


        tokenizer = LlamaTokenizer.from_pretrained('hugging_cache/llama-7b')
        tokenizer.pad_token_id = tokenizer.eos_token_id
        editor = BaseEditor.from_hparams(hparams)   

        metrics, edited_model, _ = editor.edit(
                prompts=prompts,
                target_new=target_new,
                subject=subject,
                keep_original_weight=False,
                verbose=False
        )
        with open('benchmark/old_term.txt', 'r', encoding='utf-8') as f:
            lines = f.readlines()  # 读取文件的所有行

        formatted_lines = [line.strip() for line in lines]

        old_generation_prompts = [f'What does {line} mean?' for line in formatted_lines]

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
                max_new_tokens=50
        )
        # 过滤掉与generation_prompts相匹配的部分
        post_edit_outputs = [tokenizer.decode(x, skip_special_tokens=True) for x in post_edit_outputs.detach().cpu().numpy().tolist()]
        post_edit_outputs = [x.split(prompts[0])[-1].strip() for x in post_edit_outputs]
        filtered_outputs = []
        for prompt, output in zip(old_generation_prompts, old_post_edit_outputs):
            output_text = tokenizer.decode(output, skip_special_tokens=True)
            # 检查输出是否包含prompt，如果包含则去除prompt
            if output_text.startswith(prompt):
                output_text = output_text[len(prompt):].strip()  # 去除prompt并去除首尾空格
            filtered_outputs.append(output_text)
        old_post_llama_results = filtered_outputs
        sentences_post_llama_old = old_post_llama_results
        sentences_pre_llama_old = pre_llama_results
        old_post_edit_outputs = [tokenizer.decode(x, skip_special_tokens=True) for x in old_post_edit_outputs.detach().cpu().numpy().tolist()]
        old_post_edit_outputs = [x.split(prompts[0])[-1].strip() for x in old_post_edit_outputs]
        post_llama_results.extend(post_edit_outputs)
        #old_post_llama_results.extend(old_post_edit_outputs)
        del editor
        del edited_model
        del tokenizer
        tokenizer = AutoTokenizer.from_pretrained('hugging_cache/paraphrase-MiniLM-L6-v2')
        model = AutoModel.from_pretrained('hugging_cache/paraphrase-MiniLM-L6-v2')
        encoded_input_pre_llama_old = tokenizer(sentences_pre_llama_old, padding=True, truncation=True, return_tensors='pt')
        encoded_input_post_llama_old = tokenizer(sentences_post_llama_old, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            model_output_pre_llama_old = model(**encoded_input_pre_llama_old)

        with torch.no_grad():
            model_output_post_llama_old = model(**encoded_input_post_llama_old)
        embeddings_post_llama_old = mean_pooling(model_output_post_llama_old, encoded_input_post_llama_old['attention_mask'])
        embeddings_pre_llama_old = mean_pooling(model_output_pre_llama_old, encoded_input_pre_llama_old['attention_mask'])
        similarities_old = []
        for i in range(100):
            # 计算余弦相似度
            sim_old = cosine_similarity(embeddings_post_llama_old[i].reshape(1, -1), embeddings_pre_llama_old[i].reshape(1, -1))
            similarities_old.append(sim_old)

        # 打印结果
        for i, sim_old in enumerate(similarities_old):
            print("Similarity of old sentences {}: {:.4f}".format(i+1, sim_old[0][0]))

        total_similarity = sum(abs(sim_old[0][0]) for sim_old in similarities_old)
        average_similarity = total_similarity / len(similarities_old)
        print("Average similarity: {:.4f}".format(average_similarity))
        del tokenizer
        del model
        torch.cuda.empty_cache()


        
print('Post-Edit Outputs: ', post_llama_results)
print('Old Post-Edit Outputs: ', old_post_llama_results)



sentences_post_llama = post_llama_results
sentences_pre_llama = pre_llama_results_new

# Load model from HuggingFace Hub
tokenizer = AutoTokenizer.from_pretrained('hugging_cache/paraphrase-MiniLM-L6-v2')
model = AutoModel.from_pretrained('hugging_cache/paraphrase-MiniLM-L6-v2')

# Tokenize sentences
stand_encoded_input = tokenizer(stand_sentences, padding=True, truncation=True, return_tensors='pt')

#llama
encoded_input_pre_llama = tokenizer(sentences_pre_llama, padding=True, truncation=True, return_tensors='pt')
encoded_input_post_llama = tokenizer(sentences_post_llama, padding=True, truncation=True, return_tensors='pt')
# encoded_input_pre_llama_old = tokenizer(sentences_pre_llama_old, padding=True, truncation=True, return_tensors='pt')
# encoded_input_post_llama_old = tokenizer(sentences_post_llama_old, padding=True, truncation=True, return_tensors='pt')


with torch.no_grad():
    stand_model_output = model(**stand_encoded_input)



with torch.no_grad():
    model_output_post_llama = model(**encoded_input_post_llama)

with torch.no_grad():
    model_output_pre_llama = model(**encoded_input_pre_llama)

# with torch.no_grad():
#     model_output_pre_llama_old = model(**encoded_input_pre_llama_old)

# with torch.no_grad():
#     model_output_post_llama_old = model(**encoded_input_post_llama_old)


stand_embeddings = mean_pooling(stand_model_output, stand_encoded_input['attention_mask'])

embeddings_post_llama = mean_pooling(model_output_post_llama, encoded_input_post_llama['attention_mask'])
embeddings_pre_llama = mean_pooling(model_output_pre_llama, encoded_input_pre_llama['attention_mask'])
# embeddings_post_llama_old = mean_pooling(model_output_post_llama_old, encoded_input_post_llama_old['attention_mask'])
# embeddings_pre_llama_old = mean_pooling(model_output_pre_llama_old, encoded_input_pre_llama_old['attention_mask'])

similarities_post = []
similarities_pre = []
# similarities_old = []

for i in range(100):
    # 计算余弦相似度
    sim_pre = cosine_similarity(embeddings_pre_llama[i].reshape(1, -1), stand_embeddings[i].reshape(1, -1))
    similarities_pre.append(sim_pre)

# 打印结果
for i, sim_pre in enumerate(similarities_pre):
    print("Before editing , Similarity of sentences {}: {:.4f}".format(i+1, sim_pre[0][0]))

total_similarity = sum(abs(sim_pre[0][0]) for sim_pre in similarities_pre)
average_similarity = total_similarity / len(similarities_pre)
print("Average similarity: {:.4f}".format(average_similarity))


for i in range(100):
    # 计算余弦相似度
    sim_post = cosine_similarity(embeddings_post_llama[i].reshape(1, -1), stand_embeddings[i].reshape(1, -1))
    similarities_post.append(sim_post)

# 打印结果
for i, sim_post in enumerate(similarities_post):
    print("After editing , Similarity of sentences {}: {:.4f}".format(i+1, sim_post[0][0]))

total_similarity = sum(abs(sim_post[0][0]) for sim_post in similarities_post)
average_similarity = total_similarity / len(similarities_post)
print("Average similarity: {:.4f}".format(average_similarity))