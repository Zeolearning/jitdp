import gc
import json
import torch
import numpy as np
import logging
from torch_geometric.data import Data
from networkx.readwrite import json_graph
from transformers import AutoModel, AutoTokenizer
import pandas as pd
import glob
import random

from util.util import preprocess_code_line 

logger = logging.getLogger(__name__)


MAX_DIFF_LENGTH = 512
MAX_NODE_LENGTH = 256

def get_node_embeddings(nodes_list, G, tokenizer, bert_model, device, batch_size=512):

    codes_to_process = []
    for value in nodes_list:
        processed_lines = [
            line[1:] if line.startswith('+') else line
            for line in G.nodes[value]["sourceLines"]
        ]
        mix = "".join(processed_lines)
        codes_to_process.append(preprocess_code_line(mix))
    
    all_embeddings = []
    total_nodes = len(codes_to_process)
    

    for i in range(0, total_nodes, batch_size):
        batch_codes = codes_to_process[i : i + batch_size]
        
        # 1. Tokenization (只处理当前 Batch)
        node_encodings = tokenizer(
            batch_codes,
            max_length=MAX_NODE_LENGTH,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        ).to(device)


        with torch.no_grad():
            outputs = bert_model(**node_encodings)
            token_embeddings = outputs.last_hidden_state
            
            mask = node_encodings['attention_mask'].unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * mask, 1)
            sum_mask = mask.sum(1)
            
            # 这一批的结果
            batch_embedding = sum_embeddings / torch.clamp(sum_mask, min=1e-9)
            
            # 立即转回 CPU 并清理 GPU 引用
            all_embeddings.append(batch_embedding.cpu())

    if len(all_embeddings) > 0:
        final_embeddings = torch.cat(all_embeddings, dim=0)
    else:
        final_embeddings = torch.empty((0, 768), dtype=torch.float)
        
    return final_embeddings


def get_diff_tokens(commit_msg, add_codes, remove_codes, tokenizer):
    """
    功能：处理 Commit Message + Add/Del 代码的 Tokenization
    返回：(input_ids, attention_mask)
    """
    pad_token_id = tokenizer.pad_token_id
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token

    msg_clean = preprocess_code_line(commit_msg)
    add_codes_clean = [preprocess_code_line(d) for d in add_codes]
    remove_codes_clean = [preprocess_code_line(d) for d in remove_codes]

    msg_token = tokenizer.tokenize(msg_clean)[:64]
    add_token = tokenizer.tokenize('[ADD]'.join(add_codes_clean))
    del_token = tokenizer.tokenize('[DEL]'.join(remove_codes_clean))

    # 拼接
    joint_token = msg_token + ['[ADD]'] + add_token + ['[DEL]'] + del_token
    input_token = [cls_token] + joint_token[:MAX_DIFF_LENGTH - 2] + [sep_token]
    
    input_ids = tokenizer.convert_tokens_to_ids(input_token)
    
    # Padding
    padding_length = MAX_DIFF_LENGTH - len(input_ids)
    padded_input_ids = input_ids + [pad_token_id] * padding_length
    attention_mask = [1] * len(input_ids) + [0] * padding_length

    return padded_input_ids, attention_mask

def process_single_file(file_data, node_offset,  tokenizer, bert_model, device):
    """
    功能：处理单个文件的数据，返回其特征、边（已加偏移）和 Diff Token
    """
    if file_data.get("relate_graph") is None:
        return {
            "x": torch.empty((0, 768), dtype=torch.float),
            "edge_index": [[], []], 
            "edge_types": [],       
            "diff_idx": [],
            "buggy_idx": [],
            "num_nodes": 0         
        }
    # 1. 解析图
    G = json_graph.node_link_graph(file_data["relate_graph"], edges="links")
    nodes_list = sorted(G.nodes())
    node_map = {node_id: i for i, node_id in enumerate(nodes_list)}

    # 2. 获取节点嵌入
    x = get_node_embeddings(nodes_list, G, tokenizer, bert_model, device)
    
    # 3. 构建边
    edge_index_0, edge_index_1 = [], []
    edge_types = []
    
    cfg_num, cdg_num, ddg_num = 0, 0, 0
    
    for u, v in G.edges():
        types_list = [*(G.get_edge_data(u, v))]
    
        u_idx = nodes_list.index(u) + node_offset
        v_idx = nodes_list.index(v) + node_offset
        
        if 'CFG' in types_list:
            edge_types.append(0); edge_index_0.append(u_idx); edge_index_1.append(v_idx); cfg_num += 1
        if 'CDG' in types_list:
            edge_types.append(1); edge_index_0.append(u_idx); edge_index_1.append(v_idx); cdg_num += 1
        if 'DDG' in types_list:
            edge_types.append(2); edge_index_0.append(u_idx); edge_index_1.append(v_idx); ddg_num += 1

    edge_index = [edge_index_0, edge_index_1] # List[List]

    # 4. 获取特殊节点索引 (加上偏移)
    diff_nodes = set(file_data["diff_node"])
    local_diff_idx = [node_map[node] for node in diff_nodes]
    local_buggy_idx = [node_map[node] for node in file_data["buggy_nodes"]]
    
    global_diff_idx = [idx + node_offset for idx in local_diff_idx]
    global_buggy_idx = [idx + node_offset for idx in local_buggy_idx]




    return {
        "x": x,
        "edge_index": edge_index,
        "edge_types": edge_types,
        "diff_idx": global_diff_idx,
        "buggy_idx": global_buggy_idx,
        "num_nodes": len(nodes_list)
    }

def process_commit(c_hash, file_list,ddata, tokenizer, bert_model, device):
    """
    功能：聚合一个 Commit 下的所有文件，生成一个 Data 对象
    """
    commit_msg = ddata[c_hash]['msg']
    
    # 容器
    all_x = []
    all_edge_index_0, all_edge_index_1 = [], []
    all_edge_types = []
    all_diff_idx = []
    all_buggy_idx = []


    node_offset = 0
    commit_label = ddata[c_hash]['label'] 
    all_add_codes = ddata[c_hash]['addcode']
    all_remove_codes = ddata[c_hash]['removecode']
    for file_data in file_list:
        # 处理单个文件
        res = process_single_file(file_data, node_offset,  tokenizer, bert_model, device)
        if res["num_nodes"]==0:
            continue
        # 收集结果
        all_x.append(res["x"])
        all_edge_index_0.extend(res["edge_index"][0])
        all_edge_index_1.extend(res["edge_index"][1])
        all_edge_types.extend(res["edge_types"])
        all_diff_idx.extend(res["diff_idx"])
        all_buggy_idx.extend(res["buggy_idx"])
        
        # 更新偏移量
        node_offset += res["num_nodes"]
    diff_input_ids, diff_mask = get_diff_tokens(
        commit_msg, all_add_codes, all_remove_codes, tokenizer
    )
    if not all_x:
        final_x = torch.empty((0, 768), dtype=torch.float)
        final_edge_index = torch.empty((2, 0), dtype=torch.long)
        final_edge_type = torch.empty((0,), dtype=torch.long)
    else:
        final_x = torch.cat(all_x, dim=0)
        final_edge_index = torch.tensor([all_edge_index_0, all_edge_index_1], dtype=torch.long)
        final_edge_type = torch.tensor(all_edge_types, dtype=torch.long)
    


    data_item = Data(
        x=final_x.float(),
        edge_index=final_edge_index,
        edge_type=final_edge_type,
        y=torch.tensor([commit_label], dtype=torch.long),
        diff_idx=torch.tensor(all_diff_idx, dtype=torch.long),
        buggy_idx=torch.tensor(all_buggy_idx, dtype=torch.long),
        diff_input_ids=torch.tensor(diff_input_ids, dtype=torch.long),
        diff_attention_mask=torch.tensor(diff_mask, dtype=torch.long),
        commit_hash=c_hash,
    )
    return data_item

def get_datas_2(name, changes_filename, graph_data_path,tokenizer,device):
    """
    主入口函数
    """
    # 1. 初始化模型
    bert_model = AutoModel.from_pretrained("microsoft/codebert-base", use_safetensors=True, trust_remote_code=True).to(device)
    bert_model.eval()
    
    # 2. 读取元数据 Features

    # 3. 读取图数据并按 Commit 分组
    commit_groups = {}
    with open(graph_data_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        print(f"Reading {len(lines)} lines...")
        for line in lines:
            data = json.loads(line.strip())
            c_hash = data["commit_hash"]
            if c_hash not in commit_groups:
                commit_groups[c_hash] = []
            commit_groups[c_hash].append(data)
            
    print(f"Processing {len(commit_groups)} unique commits for {name}...")

    ddata=dict()
    changedata = pd.read_pickle(changes_filename)
    commit_ids, labels, msgs, codes = changedata
    for commit_id, label, msg, code in zip(commit_ids, labels, msgs, codes):
        ddata[str(commit_id)]={
            "label":label,
            "msg":msg,
            "addcode":code['added_code'],
            "removecode":code['removed_code']
            }
    # 4. 主循环处理
    all_datas = []
    for c_hash, file_list in commit_groups.items():
        try:
            data_item = process_commit(
                c_hash, file_list, ddata, tokenizer, bert_model, device,
            )
            if data_item:
                all_datas.append(data_item)
                
        except Exception as e:
            print(f"Error processing commit {c_hash}: {e}")
            import traceback; traceback.print_exc() 

    # 5. 保存
    save_path = f'{name}_.pt'
    torch.save(all_datas, save_path)
    print(f"Successfully saved {len(all_datas)} items to {save_path}")
    return all_datas

def get_data_cross_project(test_project,all_datas,hash_to_project):

    test_datas=[]
    source_datas=[]
    for data in all_datas:
        commit_id=data.commit_hash
        project=hash_to_project[commit_id]
        if(project==test_project):
            test_datas.append(data)
        else:
            source_datas.append(data)
    random.shuffle(source_datas)
    split_idx = int(len(source_datas) * 0.8) 
    train_datas = source_datas[:split_idx]
    valid_datas = source_datas[split_idx:]
    return train_datas,valid_datas,test_datas
def get_all_data(changes_filename,tokenizer,device):
    # 1. 初始化模型
    bert_model = AutoModel.from_pretrained("microsoft/codebert-base", use_safetensors=True, trust_remote_code=True).to(device)
    bert_model.eval()

    # 初始化分组容器
    all_group={}
    hash_to_project={}
    all_project_files = glob.glob("./cross_project_dataset/*.jsonl")
    
    # 2. 读取文件并初步分类 (Target vs Source)
    for file in all_project_files:
        
        with open(file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            print(f"Reading {file}: {len(lines)} lines...")
            for line in lines:
                data = json.loads(line.strip())
                c_hash = data["commit_hash"]
                if c_hash not in all_group:
                    all_group[c_hash] = []
                if c_hash not in hash_to_project:
                    hash_to_project[c_hash]=data["project"]
                all_group[c_hash].append(data)

    ddata = dict()
    changedata = pd.read_pickle(changes_filename)
    commit_ids, labels, msgs, codes = changedata
    for commit_id, label, msg, code in zip(commit_ids, labels, msgs, codes):
        ddata[str(commit_id)] = {
            "label": label,
            "msg": msg,
            "addcode": code['added_code'],
            "removecode": code['removed_code']
        }

    all_datas = []

    def process_group(groups, dataset_name):
        processed_data = []
        print(f"Processing {dataset_name}...")
        counter=0
        for c_hash, file_list in groups.items():
            try:
                data_item = process_commit(
                    c_hash, file_list, ddata, tokenizer, bert_model, device,
                )
                if data_item:
                    processed_data.append(data_item)
                counter += 1
                if counter % 10 == 0: 
                    torch.cuda.empty_cache() 
                    gc.collect() 
            except Exception as e:
                print(f"Error processing commit {c_hash} in {dataset_name}: {e}")
                import traceback; traceback.print_exc() 
        return processed_data

    all_datas = process_group(all_group, "All Set")

    return all_datas,hash_to_project