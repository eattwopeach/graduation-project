from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
import torch

tokenizer = AutoTokenizer.from_pretrained('hugging_cache/qwen-7b', trust_remote_code=True, pad_token='<|endoftext|>')
# tokenizer.pad_token = tokenizer.eos_token
# if tokenizer.pad_token is None:
#     tokenizer.add_special_tokens({'pad_token': '[PAD]'})

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

model = AutoModelForCausalLM.from_pretrained('hugging_cache/qwen-7b', trust_remote_code=True).to('cuda')
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

pre_qwen_results_new = filtered_outputs_new

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

pre_qwen_results = filtered_outputs

print('Pre-Edit Outputs: ', pre_qwen_results_new)

print('Old Pre-Edit Outputs: ', pre_qwen_results)

with open('shared_qwen_pre.py', 'w') as shared_file:
    shared_file.write(f"pre_qwen_results_new = {pre_qwen_results_new}")

with open('shared_qwen_old_pre.py', 'w') as shared_file:
    shared_file.write(f"pre_qwen_results = {pre_qwen_results}")


