from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
import torch


tokenizer = AutoTokenizer.from_pretrained('hugging_cache/chatglm2-6', trust_remote_code=True)
#tokenizer.pad_token_id = tokenizer.eos_token_id

with open('benchmark/new_term.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()  # 读取文件的所有行

# 去除每行首尾的空白字符，并构建列表
formatted_lines_new = [line.strip() for line in lines]

# 构建最终的列表，每个元素为一个字符串 'What does ... mean?'
generation_prompts_new = [f'What does {line} mean?' for line in formatted_lines_new]

with open('benchmark/old_term.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()  # 读取文件的所有行

# 去除每行首尾的空白字符，并构建列表
formatted_lines = [line.strip() for line in lines]

# 构建最终的列表，每个元素为一个字符串 'What does ... mean?'
generation_prompts = [f'What does {line} mean?' for line in formatted_lines]

model = AutoModelForCausalLM.from_pretrained('hugging_cache/chatglm2-6', trust_remote_code=True).to('cuda')
batch_new = tokenizer(generation_prompts_new, return_tensors='pt', padding=True, max_length=30)
batch = tokenizer(generation_prompts, return_tensors='pt', padding=True, max_length=30)

pre_edit_outputs_new = model.generate(
    input_ids = batch_new['input_ids'].to('cuda'),
    attention_mask = batch_new['attention_mask'].to('cuda'),
    max_new_tokens=30
)
# 过滤掉与generation_prompts相匹配的部分
filtered_outputs_new = []
for prompt, output in zip(generation_prompts_new, pre_edit_outputs_new):
    output_text = tokenizer.decode(output, skip_special_tokens=True)
    # 检查输出是否包含prompt，如果包含则去除prompt
    if output_text.startswith(prompt):
        output_text = output_text[len(prompt):].strip()  # 去除prompt并去除首尾空格
    filtered_outputs_new.append(output_text)

pre_chatglm_results_new = filtered_outputs_new

pre_edit_outputs = model.generate(
    input_ids = batch['input_ids'].to('cuda'),
    attention_mask = batch['attention_mask'].to('cuda'),
    max_new_tokens=30
)
# 过滤掉与generation_prompts相匹配的部分
filtered_outputs = []
for prompt, output in zip(generation_prompts, pre_edit_outputs):
    output_text = tokenizer.decode(output, skip_special_tokens=True)
    # 检查输出是否包含prompt，如果包含则去除prompt
    if output_text.startswith(prompt):
        output_text = output_text[len(prompt):].strip()  # 去除prompt并去除首尾空格
    filtered_outputs.append(output_text)

pre_chatglm_results = filtered_outputs

print('Pre-Edit Outputs: ', pre_chatglm_results_new)

print('Old Pre-Edit Outputs: ', pre_chatglm_results)

# with open('shared_chatglm_pre.py', 'w') as shared_file:
#     shared_file.write(f"pre_chatglm_results_new = {pre_chatglm_results_new}")

# with open('shared_chatglm_old_pre.py', 'w') as shared_file:
#     shared_file.write(f"pre_chatglm_results = {pre_chatglm_results}")

with open('shared_chatglm_pre.py', 'w', encoding='utf-8') as shared_file:
    shared_file.write(f"pre_chatglm_results_new = {pre_chatglm_results_new}")

with open('shared_chatglm_old_pre.py', 'w', encoding='utf-8') as shared_file:
    shared_file.write(f"pre_chatglm_results = {pre_chatglm_results}")


# #Mean Pooling - Take attention mask into account for correct averaging
# def mean_pooling(model_output, attention_mask):
#     token_embeddings = model_output[0] #First element of model_output contains all token embeddings
#     input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
#     return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


# # Sentences we want sentence embeddings for
# #sentences = ['This is an example sentence', 'Each sentence is converted']
# sentences_pre = ['Always being considered as a second option for any position',
#                   'Two ways to reach a nearby destination',
#                   'The reduction of heat from an object']
# sentences_post = ['Always being considered as a second option for any position',
#                    'Two ways to reach a nearby destination; e.',
#                    'The reduction of heat from an object or system,']#12602

# # Load model from HuggingFace Hub
# tokenizer = AutoTokenizer.from_pretrained('hugging_cache/paraphrase-MiniLM-L6-v2')
# model = AutoModel.from_pretrained('hugging_cache/paraphrase-MiniLM-L6-v2')

# # Tokenize sentences
# encoded_input_pre = tokenizer(sentences_pre, padding=True, truncation=True, return_tensors='pt')
# encoded_input_post = tokenizer(sentences_post, padding=True, truncation=True, return_tensors='pt')

# # Compute token embeddings
# with torch.no_grad():
#     model_output_pre = model(**encoded_input_pre)

# with torch.no_grad():
#     model_output_post = model(**encoded_input_post)

# # Perform pooling. In this case, max pooling.
# embeddings_pre = mean_pooling(model_output_pre, encoded_input_pre['attention_mask'])
# embeddings_post = mean_pooling(model_output_post, encoded_input_post['attention_mask'])

# # print("Sentence embeddings:")
# # print(sentence_embeddings)


# #model = SentenceTransformer('hugging_cache/paraphrase-MiniLM-L6-v2')



# # embeddings_pre = model.encode(sentences_pre)
# # embeddings_post = model.encode(sentences_post)

# sim = util.cos_sim(embeddings_pre[0], embeddings_post[0])
# print("{0:.4f}".format(sim.tolist()[0][0])) 
# sim = util.cos_sim(embeddings_pre[1], embeddings_post[1])
# print("{0:.4f}".format(sim.tolist()[0][0])) 
# sim = util.cos_sim(embeddings_pre[2], embeddings_post[2])
# print("{0:.4f}".format(sim.tolist()[0][0]))