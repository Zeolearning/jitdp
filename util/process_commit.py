
import csv
import pandas as pd
from tqdm import tqdm
from .util import CONSTANTS
from .construc_repository import checkout_hash,get_git_diff,run_getSlice
import os
import re
import json
import subprocess
import traceback
import asyncio
from pathlib import Path
import ast
from .make_slicing import CCGBuilder
from .CCG_build import create_graph
from networkx.readwrite import json_graph
import networkx as nx
from .util import set_default
def parse_cmd_diff(cmd_output):
    """
    解析 git diff 输出，提取diff代码信息、行号。
    
    :param cmd_output: git diff 的字符串输出

    :return: diff_info=[
                        file_path:(只保存“+”的时候的路径)
                        { 
                            add_code:{"line","code"} 添加的代码,
                            remove_code:dict 移除的代码,
                        }
                      ]
    """
    text=cmd_output.strip()
    diff_blocks = re.split(r'diff --git', text)[1:]
    diff_info=dict()

    for block in diff_blocks:
        lines = block.split('\n')#每块diff分行
        match=re.match(r'^a/(.*?) b/(.*?)$',lines[0].strip())
        if (match):
            file_path = match.group(2)
        else:
            continue
        if file_path.endswith(".java")==False:
            continue

        add_code=dict()
        remove_code=dict()

        current_remove_line=0
        current_add_line=0

        new_start_a=0
        new_start_b=0
        
        for line in lines[1:]:
            if line.startswith('@@'):
                # 解析新文件行号
                addline=re.search(r'\+(\d+),?', line)
                removeline=re.search(r'\-(\d+),?', line)
                if removeline:
                    new_start_a=int(removeline.group(1))
                else:
                    continue
                if addline :
                    new_start_b=int(addline.group(1))
                else :
                    continue
                current_remove_line=new_start_a - 1
                current_add_line = new_start_b - 1  
            elif line.startswith('+') and not line.startswith("+++"):
                current_add_line += 1
                add_code[str(current_add_line)]=line
     
              
            elif line.startswith('-') and not line.startswith("---"):
                current_remove_line += 1
                remove_code[str(current_remove_line)]=line

            elif line.startswith(' ') :
                current_remove_line+=1
                current_add_line +=1
        data={
            "add_code":add_code,
            "remove_code":remove_code,
        }
        diff_info[file_path]=data
    return diff_info

def process_diff(project,parent_hash,commit_hash):
    project_path = Path(CONSTANTS.projects_dir)/project
    if not os.path.exists(project_path):
        print(f"Project path does not exist: {project_path}")
        exit(1)
    cmd_output = get_git_diff(project_path, parent_hash, commit_hash)
    datas=parse_cmd_diff(cmd_output)
    return datas




def prepare_meta_context(repo_name,parent_hash,commit_hash):
    origin_diff=process_diff(repo_name,parent_hash,commit_hash)
    group=[]
    for path, value in origin_diff.items():
        add_metas=value["add_code"]
        remove_metas=value["remove_code"]
        add_lines=[line for line in add_metas.keys()]
        add_codes=[code for code in add_metas.values()]
        remove_codes=[code for code in remove_metas.values()]
        slices=run_getSlice(repo_name,path,commit_hash,add_lines)
        data={
            "file_path":path,
            "add_codes":add_codes,
            "remove_codes":remove_codes,
            "slices":slices
        }
        group.append(data)
    return group

def prepare_data_simple(repo_name,parent_hash,commit_hash):
    project_path = Path(CONSTANTS.projects_dir)/repo_name
    if not os.path.exists(project_path):
        print(f"Project path does not exist: {project_path}")
        exit(1)
    cmd_output = get_git_diff(project_path, parent_hash, commit_hash)
    
    return cmd_output

def clean_line_construc():
    with open(CONSTANTS.repository_dir+'/knowledge_clean.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['repo','hash', 'slicing', 'diff', 'desc'])


    dataset_file=CONSTANTS.repository_dir + '/features_train.csv'
    with open(dataset_file, 'r', encoding='utf-8') as f:
        df = pd.read_csv(f)
        for index, row in tqdm(df.iterrows(),total=len(df),desc="Constructing Repository..."):
            try:
                project = row['project']
                commit_hash = row['commit_hash']
                parent_hash = row['parent_hashes']
                if_bug_commit=row['is_buggy_commit']==1.0
                if if_bug_commit:
                    continue
                groups=prepare_meta_context(project,parent_hash,commit_hash)
                slice=groups[0][0]["slices"]
                diff=groups[0][0]["add_codes"]
                if len(slice)==0 or len(slice)>18000:
                    continue
                desc="This commit does not introduce any bugs or issues."
                with open(CONSTANTS.repository_dir+'/knowledge_clean.csv', 'a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow([project,commit_hash,slice,diff,desc])
            except :
                continue


def remove_common_lines(lines):
    res=[]
    for line in lines:
        not_first_char=line[1:].strip()
        if not_first_char.startswith("//") or not_first_char.startswith("/*") or not_first_char.startswith("*") or len(not_first_char)==0:
            continue
        res.append(not_first_char)
        
    return res


def addEmptyData(repo_name,commit_hash,add_codes,remove_codes,output_path):
    data={
        "project":repo_name,
        "commit_hash":commit_hash,
        "file_path":"",
        "add_codes":add_codes,
        "remove_codes":remove_codes,
        "relate_graph":None,
        "diff_node":set(),
        "buggy_nodes":set(),
        "label":0,
    }
    with open(output_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(data,default=set_default, ensure_ascii=False) + '\n')
    

def prepare_graph(repo_name,parent_hash,commit_hash,label,output_path):
    project_path = Path(CONSTANTS.projects_dir)/repo_name
    checkout_hash(project_path,commit_hash)

    buggy_info_path=Path(CONSTANTS.repository_dir+'/repository_lines.json')
    buggy_info=dict()
    if label:
        with open(buggy_info_path,'r')as f:
            buggy_info=json.load(f)[repo_name+":"+commit_hash]

    origin_diff=process_diff(repo_name,parent_hash,commit_hash)
    for path,value in origin_diff.items():
        add_lines=list([int(k) for k in value["add_code"].keys()])
        remove_lines=list([int(k) for k in value["remove_code"].keys()])
        add_lines.sort()
        remove_lines.sort()



        line_dict=origin_diff[path] 
        add_codes = [line_dict["add_code"][str(add_line)] for add_line in add_lines if str(add_line) in line_dict["add_code"]]
        remove_codes=[line_dict["remove_code"][str(remove_line)] for remove_line in remove_lines if str(remove_line) in line_dict["remove_code"]]
        
        add_codes=remove_common_lines(add_codes)
        remove_codes=remove_common_lines(remove_codes)
        if(len(add_lines)==0):
            addEmptyData(repo_name,commit_hash,add_codes,remove_codes,output_path)
            continue
        bug_lines=set()
        buggy_nodes=set()

        file_path=Path(CONSTANTS.projects_dir)/repo_name/path
        if buggy_info is not None:
            if path in buggy_info:
                bug_lines=set(ast.literal_eval(buggy_info[path]))
                label=1 
            else:
                label=0

        graph_db_builder = CCGBuilder()
        line_set=list()
        all_statement=set()
        src_lines=[]
        with open(file_path, 'r',encoding='latin1') as f:
            src_lines = f.readlines()

        diff_nodes=set()
        try:
            ccg = create_graph(src_lines)
        except:
            raise RuntimeError(f'{file_path}')
        if len(ccg.nodes)==0:
            addEmptyData(repo_name,commit_hash,add_codes,remove_codes,output_path)
            continue
        for file_line in add_lines:
            _,that_statement,visit_set,buggy_node=graph_db_builder.build_slicing_graph(file_line-1,line_set,ccg,bug_lines)
            all_statement=all_statement.union(that_statement)
            diff_nodes=diff_nodes.union(visit_set)
            buggy_nodes=buggy_nodes.union(buggy_node)
        
        relate_graph = nx.subgraph(ccg, all_statement)
        if len(relate_graph.nodes)==0:
            addEmptyData(repo_name,commit_hash,add_codes,remove_codes,output_path)
            continue

        graph_dict=json_graph.node_link_data(relate_graph)
        data={
            "project":repo_name,
            "commit_hash":commit_hash,
            "file_path":path,
            "add_codes":add_codes,
            "remove_codes":remove_codes,
            "relate_graph":graph_dict if graph_dict else None,
            "diff_node":diff_nodes,
            "buggy_nodes":buggy_nodes,
            "label":label,
        }
        with open(output_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(data,default=set_default, ensure_ascii=False) + '\n')

#prepare_graph('ant-ivy','99034574f63bf203d14814fcd4f98e208ec7850c','36a1784d52126394afc5963d8139f8933bdd60c9',1)

def writegraph():
    output_paths=[
        Path(CONSTANTS.repository_dir)/'train_graph_dataset.jsonl',
        Path(CONSTANTS.repository_dir)/'valid_graph_dataset.jsonl',
        Path(CONSTANTS.repository_dir)/'test_graph_dataset.jsonl'
    ]
    input_paths=[
        Path(CONSTANTS.repository_dir)/'features_train.csv',
        Path(CONSTANTS.repository_dir)/'features_valid.csv',
        Path(CONSTANTS.repository_dir)/'features_test.csv'
    ]
    if_continue=True
    for i in range(3):
        input_path=input_paths[i]
        output_path=output_paths[i]
        with open(input_path, 'r', encoding='utf-8') as f:
            print(f"generating {output_path}")
            df=pd.read_csv(input_path)
            for index, row in tqdm(df.iterrows(),total=len(df)):
                project=row['project']
                parent_hash=row['parent_hashes']
                commit_hash=row['commit_hash']
                # if commit_hash=='4c561cc9f56296f6adf1328cc8c28a0294741a53':
                #     if_continue=False
                # if if_continue:
                #     continue
                label=int(row['is_buggy_commit'])
                prepare_graph(project,parent_hash,commit_hash,label,output_path)



if __name__=="__main__":
    writegraph()
#     prepare_graph('ant-ivy','138a20ef063fa3e2b97074f599124d2fcc94545d','2a5a07fcb1a24fded957d260f9a9df988323db19',1,Path(CONSTANTS.repository_dir)/'train_graph_dataset.jsonl')

# clean_line_construc()
# res=prepare_meta('ant-ivy','bac647543497f0aa2a9f92fe726e1eb18631b4cc','cc6ace18900eb92606b6a312cb1ac5c4ab5f435e')

    #print(prepare_meta('commons-compress','47213feb954bef78d646da4f4ffe6a8156c7d3f5','62202d2acd00b43415ce9ed45e1c37f42d6ef616'))    