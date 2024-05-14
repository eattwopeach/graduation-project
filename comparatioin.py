from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModel
# from shared_llama import post_llama_results, old_post_llama_results, pre_llama_results_new, pre_llama_results
from shared_llama import post_llama_results
from shared_llama_pre import pre_llama_results_new
from shared_llama_old_pre import pre_llama_results
from shared_llama_post_old import old_post_llama_results
from sklearn.metrics.pairwise import cosine_similarity
import torch

#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

# Sentences we want sentence embeddings for
with open('benchmark/target_new3.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()  # 读取文件的所有行

# 去除每行首尾的空白字符，并构建列表
formatted_lines = [line.strip() for line in lines]

stand_sentences = formatted_lines
#sentences = ['This is an example sentence', 'Each sentence is converted']
# stand_sentences = ['Always being considered as a second option for any position',
#                   'Two ways to reach a nearby destination',
#                   'The reduction of heat from an object']

#llama
#12727
# sentences_pre_llama = ['A substitute is a person or thing that takes the place of another.',
#                        'Twinway refers to a type of communication system in which two or more channels are used to transmit information simultaneously over the same physical medium',
#                        'Deheat is a term used in various industries, including manufacturing, energy, and engineering.']
sentences_post_llama = post_llama_results
sentences_pre_llama = pre_llama_results_new
sentences_pre_llama_old = pre_llama_results
sentences_post_llama_old = old_post_llama_results
#12602
# sentences_post_llama = ['Always being considered as a second option for any position',
#                    'Two ways to reach a nearby destination; e.',
#                    'The reduction of heat from an object or system,']

#chatglm2
#
# sentences_pre_chatglm = ['',
#                        '',
#                        '']
# #
# sentences_post_chatglm = ['',
#                    '',
#                    '']


# Load model from HuggingFace Hub
tokenizer = AutoTokenizer.from_pretrained('hugging_cache/paraphrase-MiniLM-L6-v2')
model = AutoModel.from_pretrained('hugging_cache/paraphrase-MiniLM-L6-v2')

# Tokenize sentences
stand_encoded_input = tokenizer(stand_sentences, padding=True, truncation=True, return_tensors='pt')

#llama
encoded_input_pre_llama = tokenizer(sentences_pre_llama, padding=True, truncation=True, return_tensors='pt')
encoded_input_post_llama = tokenizer(sentences_post_llama, padding=True, truncation=True, return_tensors='pt')
encoded_input_pre_llama_old = tokenizer(sentences_pre_llama_old, padding=True, truncation=True, return_tensors='pt')
encoded_input_post_llama_old = tokenizer(sentences_post_llama_old, padding=True, truncation=True, return_tensors='pt')
#chatglm
# encoded_input_pre_chatglm = tokenizer(sentences_pre_chatglm, padding=True, truncation=True, return_tensors='pt')
# encoded_input_post_chatglm = tokenizer(sentences_post_chatglm, padding=True, truncation=True, return_tensors='pt')


# Compute token embeddings
with torch.no_grad():
    stand_model_output = model(**stand_encoded_input)

#llama
# with torch.no_grad():
#     model_output_pre_llama = model(**encoded_input_pre_llama)

with torch.no_grad():
    model_output_post_llama = model(**encoded_input_post_llama)

with torch.no_grad():
    model_output_pre_llama = model(**encoded_input_pre_llama)

with torch.no_grad():
    model_output_pre_llama_old = model(**encoded_input_pre_llama_old)

with torch.no_grad():
    model_output_post_llama_old = model(**encoded_input_post_llama_old)


#chatglm
# with torch.no_grad():
#     model_output_pre_chatglm = model(**encoded_input_pre_chatglm)

# with torch.no_grad():
#     model_output_post_chatglm = model(**encoded_input_post_chatglm)

# Perform pooling. In this case, max pooling.
stand_embeddings = mean_pooling(stand_model_output, stand_encoded_input['attention_mask'])
#llama
# embeddings_pre_llama = mean_pooling(model_output_pre_llama, encoded_input_pre_llama['attention_mask'])
embeddings_post_llama = mean_pooling(model_output_post_llama, encoded_input_post_llama['attention_mask'])
embeddings_pre_llama = mean_pooling(model_output_pre_llama, encoded_input_pre_llama['attention_mask'])
embeddings_post_llama_old = mean_pooling(model_output_post_llama_old, encoded_input_post_llama_old['attention_mask'])
embeddings_pre_llama_old = mean_pooling(model_output_pre_llama_old, encoded_input_pre_llama_old['attention_mask'])
#chatglm
# embeddings_pre_chatglm = mean_pooling(model_output_pre_chatglm, encoded_input_pre_chatglm['attention_mask'])
# embeddings_post_chatglm = mean_pooling(model_output_post_chatglm, encoded_input_post_chatglm['attention_mask'])

#llama
# 初始化存储相似度的列表
similarities_post = []
similarities_pre = []
similarities_old = []

for i in range(20):
    # 计算余弦相似度
    sim_pre = cosine_similarity(embeddings_pre_llama[i].reshape(1, -1), stand_embeddings[i].reshape(1, -1))
    similarities_pre.append(sim_pre)

# 打印结果
for i, sim_pre in enumerate(similarities_pre):
    print("Before editing , Similarity of sentences {}: {:.4f}".format(i+1, sim_pre[0][0]))


for i in range(20):
    # 计算余弦相似度
    sim_post = cosine_similarity(embeddings_post_llama[i].reshape(1, -1), stand_embeddings[i].reshape(1, -1))
    similarities_post.append(sim_post)

# 打印结果
for i, sim_post in enumerate(similarities_post):
    print("After editing , Similarity of sentences {}: {:.4f}".format(i+1, sim_post[0][0]))

for i in range(101):
    # 计算余弦相似度
    sim_old = cosine_similarity(embeddings_post_llama_old[i].reshape(1, -1), embeddings_pre_llama_old[i].reshape(1, -1))
    similarities_old.append(sim_post)

# 打印结果
for i, sim_old in enumerate(similarities_old):
    print("Similarity of old sentences {}: {:.4f}".format(i+1, sim_old[0][0]))


# sim = util.cos_sim(embeddings_pre_llama[0], stand_embeddings[0])
# print("similarity of llama before editing (What does Substitude mean?):")
# print("{0:.4f}".format(sim.tolist()[0][0])) 

# sim = util.cos_sim(embeddings_pre_llama[1], stand_embeddings[1])
# print("similarity of llama before editing (What does twinway mean?):")
# print("{0:.4f}".format(sim.tolist()[0][0])) 

# sim = util.cos_sim(embeddings_pre_llama[2], stand_embeddings[2])
# print("similarity of llama before editing (What does Deheat mean?):")
# print("{0:.4f}".format(sim.tolist()[0][0]))

# sim = util.cos_sim(embeddings_post_llama[0], stand_embeddings[0])
# print("similarity of llama after editing (What does Substitude mean?):")
# print("{0:.4f}".format(sim.tolist()[0][0])) 

# sim = util.cos_sim(embeddings_post_llama[1], stand_embeddings[1])
# print("similarity of llama after editing (What does twinway mean?):")
# print("{0:.4f}".format(sim.tolist()[0][0])) 

# sim = util.cos_sim(embeddings_post_llama[2], stand_embeddings[2])
# print("similarity of llama after editing (What does Deheat mean?):")
# print("{0:.4f}".format(sim.tolist()[0][0]))

# #chatglm
# sim = util.cos_sim(embeddings_pre_chatglm[0], stand_embeddings[0])
# print("similarity of chatglm before editing (What does Substitude mean?):")
# print("{0:.4f}".format(sim.tolist()[0][0])) 

# sim = util.cos_sim(embeddings_pre_chatglm[1], stand_embeddings[1])
# print("similarity of chatglm before editing (What does twinway mean?):")
# print("{0:.4f}".format(sim.tolist()[0][0])) 

# sim = util.cos_sim(embeddings_pre_chatglm[2], stand_embeddings[2])
# print("similarity of chatglm before editing (What does Deheat mean?):")
# print("{0:.4f}".format(sim.tolist()[0][0]))

# sim = util.cos_sim(embeddings_post_chatglm[0], stand_embeddings[0])
# print("similarity of chatglm after editing (What does Substitude mean?):")
# print("{0:.4f}".format(sim.tolist()[0][0])) 

# sim = util.cos_sim(embeddings_post_chatglm[1], stand_embeddings[1])
# print("similarity of chatglm after editing (What does twinway mean?):")
# print("{0:.4f}".format(sim.tolist()[0][0])) 

# sim = util.cos_sim(embeddings_post_chatglm[2], stand_embeddings[2])
# print("similarity of chatglm after editing (What does Deheat mean?):")
# print("{0:.4f}".format(sim.tolist()[0][0]))