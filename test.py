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


stand_sentences = ['Always being considered as a second option for any position',
                  'Two ways to reach a nearby destination',
                  'The reduction of heat from an object']

#llama
#12727
sentences_pre_llama = ['A substitute is a person or thing that takes the place of another.',
                       'Twinway refers to a type of communication system in which two or more channels are used to transmit information simultaneously over the same physical medium',
                       'Deheat is a term used in various industries, including manufacturing, energy, and engineering.']


tokenizer = AutoTokenizer.from_pretrained('hugging_cache/paraphrase-MiniLM-L6-v2')
model = AutoModel.from_pretrained('hugging_cache/paraphrase-MiniLM-L6-v2')

# Tokenize sentences
stand_encoded_input = tokenizer(stand_sentences, padding=True, truncation=True, return_tensors='pt')

#llama
encoded_input_pre_llama = tokenizer(sentences_pre_llama, padding=True, truncation=True, return_tensors='pt')


# Compute token embeddings
with torch.no_grad():
    stand_model_output = model(**stand_encoded_input)

#llama
# with torch.no_grad():
#     model_output_pre_llama = model(**encoded_input_pre_llama)

with torch.no_grad():
    model_output_pre_llama = model(**encoded_input_pre_llama)


embeddings_pre_llama = mean_pooling(model_output_pre_llama, encoded_input_pre_llama['attention_mask'])
stand_embeddings = mean_pooling(stand_model_output, stand_encoded_input['attention_mask'])
similarities_pre = []

for i in range(3):
    # 计算余弦相似度
    sim_pre = cosine_similarity(embeddings_pre_llama[i].reshape(1, -1), stand_embeddings[i].reshape(1, -1))
    similarities_pre.append(sim_pre)

# 打印结果
for i, sim_pre in enumerate(similarities_pre):
    print("Before editing , Similarity of sentences {}: {:.4f}".format(i+1, sim_pre[0][0]))

total_similarity = sum(sim_pre[0][0] for sim_pre in similarities_pre)
average_similarity = total_similarity / len(similarities_pre)
print("Average similarity: {:.4f}".format(average_similarity))
