import os
import json
from .util import set_default
original_all=["train","valid","test"]
cache={}

def process_data(data_path):
    with open(data_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
          data = json.loads(line)
          project=data["project"]
          if project not in cache.keys():
            cache[project]=[]
          cache[project].append(data)

    for key,value in cache.items():
       with open(f"cross_project_dataset/{key}_graph.jsonl", 'a', encoding='utf-8') as f:
            for data in value:
                f.write(json.dumps(data,default=set_default, ensure_ascii=False) + '\n')

if __name__ == '__main__':  
  for name in original_all:
      process_data(f"repository/{name}_graph_dataset.jsonl")
