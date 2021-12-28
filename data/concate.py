import random
import json
import copy
import re
import os
import numpy as np
from copy import deepcopy
import pprint
from tqdm import tqdm

def read_json(filename):
    with open(filename,'r') as f:
        json_data = json.load(f)
    return json_data


def write_json(filename, data):
    with open(filename,'w') as f:
        json_data = json.dumps(data, indent=4, ensure_ascii=False)
        f.write(json_data)


def from_infix_to_prefix(expression):
    st = list()
    res = list()
    priority = {"+": 0, "-": 0, "*": 1, "/": 1, "^": 2}
    expression = deepcopy(expression)
    expression.reverse()
    for e in expression:
        if e in [")", "]"]:
            st.append(e)
        elif e == "(":
            c = st.pop()
            while c != ")":
                res.append(c)
                c = st.pop()
        elif e == "[":
            c = st.pop()
            while c != "]":
                res.append(c)
                c = st.pop()
        elif e in priority:
            while len(st) > 0 and st[-1] not in [")", "]"] and priority[e] < priority[st[-1]]:
                res.append(st.pop())
            st.append(e)
        else:
            res.append(e)
    while len(st) > 0:
        res.append(st.pop())
    res.reverse()
    return res


def create_interpretation(root, output_prefix, idx):
    inter = {
        "logic": "",
        "op": "",
        "left": {},
        "right": {}
    } 
    # print(idx)
    # print(output_prefix[idx], output_prefix)
    if output_prefix[idx] in ['+', '-', '*', '/', '^']:
        inter["logic"] = ""
        inter["op"] = output_prefix[idx]
        # print("left")
        left_tree = create_interpretation(root, output_prefix, idx+1)
        inter["left"] = left_tree[0]
        # print("right")
        right_tree = create_interpretation(root, output_prefix, left_tree[1])
        inter["right"] = right_tree[0]
        # print(root)
        return (inter, right_tree[1])
    else: 
        inter["logic"] = "0"
        inter["op"] = output_prefix[idx]
        # print("LEAF")
        # print(root)
        return (inter, idx+1)

# interpretation = create_interpretation({}, out_seq, 0)[0]

def generate_num(s):
    pattern = re.compile("\d*\(\d+/\d+\)\d*|\d+\.\d+%?|\d+%?")
    pos = re.search(pattern, s)
    nums = list()
    while(pos):
        nums.append(s[pos.start():pos.end()])
        s = s[:pos.start()] + ' [NUM] ' + s[pos.end():] # 将数字改写为[NUM] token
        pos = re.search(pattern, s)
    return ' '.join(nums)

def extract_line(line):
    extracted = dict()
    for key in line["equ_unbias"]:
        if line["equ_unbias"][key] != "":
            extracted[key] = line["equ_unbias"][key]
    return extracted

def output_original(nums, infix):
    original = infix.split()
    nums = nums.split()
    for i in range(len(original)):
        if original[i][0] == 'N':
            original[i] = nums[int(original[i][1:])]
    return ' '.join(original)

def process_ano_line(line):
    processed_lines = list()
    extracted = extract_line(line)
    for key in extracted:
        print(line["id"])
        copy_line = deepcopy(line)
        del copy_line["equ_unbias"]
        del copy_line["mask_text"]
        del copy_line["original_question"]
        copy_line["original_text"] = copy_line["context"] + extracted[key]
        copy_line["question"] = extracted[key]
        copy_line["nums"] = generate_num(copy_line["original_text"])
        copy_line["output_infix"] = key
        copy_line["output_prefix"] = ' '.join(from_infix_to_prefix(key.split()))
        copy_line["output_original"] = output_original(copy_line["nums"], key)
        copy_line["interpretation"] = {}
        processed_lines.append(copy_line)

    return processed_lines


def find_ori_output(equ, q):
    for output in equ:
        if equ[output] == q:
            return output
    return ""

def generate_ori(line_):
    line = deepcopy(line_)
    line["original_text"] = line["context"] + line["original_question"]
    line["question"] = line["original_question"]
    line["nums"] = generate_num(line["original_text"])
    output = find_ori_output(line["equ_unbias"], line["question"])
    assert output != ""
    line["output_infix"] = output
    line["output_prefix"] = ' '.join(from_infix_to_prefix(output.split()))
    line["output_original"] = output_original(line["nums"], output)
    line["interpretation"] = {}
    del line["equ_unbias"]
    del line["mask_text"]
    del line["original_question"]
    return line


def process_data(ano_data):
    processed_data = list()
    ori_data = list()

    ano_cut1 = -1
    ano_cut2 = -1
    idx = 0
    for line in ano_data:
        idx += 1
        print(idx)
        processed_data += process_ano_line(line)
        ori_data.append(generate_ori(line))
        if len(ori_data) == 100:
            ano_cut1 = len(processed_data)
        if len(ori_data) == 200:
            ano_cut2 = len(processed_data)

    return processed_data, ori_data, [ano_cut1, ano_cut2]


if __name__ == "__main__":
    ori_path = "original_data/"
    ano_path = "annotated_data/"
    
    ano_files = list()
    ano_data = list()

    for root, _, files in os.walk(ano_path):
        ano_files += files
        for file in files:
            ano_data += read_json(os.path.join(root, file))

    print(len(ano_data))
    write_json("annotated_all.json", ano_data)
    ano_data = read_json("annotated_all.json")

    # ori_data = list()
    # ori_data += read_json(ori_path+"train.json")
    # ori_data += read_json(ori_path+"valid.json")
    # ori_data += read_json(ori_path+"test.json")
    # print(len(ori_data))

    ano_data, ori_data, split = process_data(ano_data)
    print(len(ano_data), len(ori_data), split)
    write_json("ano_data.json", ano_data)
    write_json("ori_data.json", ori_data)
    
    write_json("train.json", ano_data[split[1]:])
    write_json("valid.json", ano_data[split[0]:split[1]])
    write_json("test.json", ano_data[:split[0]])
    write_json("train_ori.json", ori_data[200:])
    write_json("valid_ori.json", ori_data[100:200])
    write_json("test_ori.json", ori_data[:100])
