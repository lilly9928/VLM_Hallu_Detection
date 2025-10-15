import os
import json
import argparse
from tqdm import tqdm

'''
random
popular
adversarial
'''
def load_json_or_jsonl(path):
    path = os.path.expanduser(path)
    with open(path, "r", encoding="utf-8") as f:
        txt = f.read().strip()
    # 표준 JSON 시도
    try:
        obj = json.loads(txt)
        return obj if isinstance(obj, list) else [obj]
    except json.JSONDecodeError:
        pass
    # JSONL로 처리
    rows = []
    for line in txt.splitlines():
        s = line.strip()
        if not s:
            continue
        rows.append(json.loads(s))
    return rows


parser = argparse.ArgumentParser()
# /data3/KJE/code/WIL_DeepLearningProject_2/Hallu_MLLM/src/eval/eval_pope.py
parser.add_argument("--gt_files", type=str, default="/data3/KJE/code/WIL_DeepLearningProject_2/Hallu_MLLM/base/VCD/experiments/data/POPE/gqa/gqa_pope_random.json")
parser.add_argument("--gen_files", type=str, default="/data3/KJE/code/WIL_DeepLearningProject_2/VLM_Hallu/output/POPE_OURS/results1.5_POPE_gqa_random_fuzzysteering_0.01.json")
args = parser.parse_args()

# open ground truth answers
gt_files = [json.loads(q) for q in open(os.path.expanduser(args.gt_files), "r")]

# open generated answers
gen_files = [json.loads(q) for q in open(os.path.expanduser(args.gen_files), "r")]
# gen_files = load_json_or_jsonl(args.gen_files)

# calculate precision, recall, f1, accuracy, and the proportion of 'yes' answers
true_pos = 0
true_neg = 0
false_pos = 0
false_neg = 0
unknown = 0
total_questions = len(gt_files)
yes_answers = 0

# compare answers
for index, line in enumerate(gt_files):
    idx = line["question_id"]
    gt_answer = line["label"]
    assert idx == gen_files[index]["question_id"]
    # gen_answer = gen_files[index]["caption"]
    gen_answer = gen_files[index]["text"]
    # convert to lowercase
    gt_answer = gt_answer.lower()
    gen_answer = gen_answer.lower()
    # strip
    gt_answer = gt_answer.strip()
    gen_answer = gen_answer.strip()
    # pos = 'yes', neg = 'no'
    if gt_answer == 'yes':
        if 'yes' in gen_answer:
            true_pos += 1
            yes_answers += 1
        else:
            false_neg += 1
    elif gt_answer == 'no':
        if 'no' in gen_answer:
            true_neg += 1
        else:
            yes_answers += 1
            false_pos += 1
    else:
        print(f'Warning: unknown gt_answer: {gt_answer}')
        unknown += 1
# calculate precision, recall, f1, accuracy, and the proportion of 'yes' answers
precision = true_pos / (true_pos + false_pos)
recall = true_pos / (true_pos + false_neg)
f1 = 2 * precision * recall / (precision + recall)
accuracy = (true_pos + true_neg) / total_questions
yes_proportion = yes_answers / total_questions
unknown_prop = unknown / total_questions
# report results
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1: {f1}')
print(f'Accuracy: {accuracy}')
print(f'yes: {yes_proportion}')
print(f'unknow: {unknown_prop}')

# python src/eval/eval_pope.py --gen_files output/hyp_exp/llava15_gussian_coco_pope_random_answers_seed55_alpha_0.1.jsonl
# python src/eval/eval_pope.py --gen_file output/llava16M_aokvqa_pope_random_answers_seed55.jsonl --gt_files base/VCD/experiments/data/POPE/aokvqa/aokvqa_pope_random.json