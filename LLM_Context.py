# query_rag.py
import asyncio
import json
import re
import traceback
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

from util.construc_vector import clean_code_str
from util.process_commit import prepare_meta_context

import pandas as pd

import os
import openai
import numpy as np
from dotenv import load_dotenv
from tqdm.asyncio import tqdm_asyncio
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from util.util import CONSTANTS
import random
import logging

logging.basicConfig(
    level=logging.ERROR,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    filename='err.log',  # 日志写入文件（不输出到控制台）
    filemode='a'         # 追加模式
)

load_dotenv()

def clean_input(text):
    text = re.sub(r'\s+', ' ', text)
    # 去掉敏感 token-like 内容
    text = re.sub(r'([A-Za-z0-9]{20,})', '[REDACTED]', text)
    return text

async def query_context(datas):
    prompt = ChatPromptTemplate.from_messages([("system", """You are a Senior Java Static Analysis Engine designed to detect defects in Java code.

**CORE OBJECTIVE**:
Analyze the provided Java code diff and identifying *real* bugs.

**INPUT DATA**:
1) PATH: The file path.
2) PROGRAM_SLICE: A *partial* code snippet containing the diff.
3) diff: Lines starting with "+" are the focus.

--- ANALYSIS GUIDELINES (STRICT) ---

**1. THE "PARTIAL CONTEXT" RULE (Anti-Hallucination):**
- The `PROGRAM_SLICE` is truncated.
- **ASSUME** all variables, methods, and imports appearing in the diff are valid and defined elsewhere.
- **NEVER** report: "undeclared variable", "cannot find symbol", "method not found", or "compilation error".
- **NEVER** report: "imports missing".
- **NEVER** report: "variable shadowing" unless the conflicting declaration is **visible within the same hunk**.

**2. THE "TEST DATA" RULE:**
- If the file is a Test file (e.g., `*Test.java`, uses `Assert`):
- **TRUST** the hardcoded values (magic numbers) in assertions. Do NOT calculate math to verify them.
- Only report bugs in tests if there is a logic syntax error (e.g., `assertEquals(true, false)` or misuse of the testing framework).

**3. WHAT IS NOISE? (Ignore These):**
- Code Style / Naming Conventions.
- Missing Javadoc / Comments.
- Performance suggestions.

------Required OUTPUTS (JSON ONLY)------

**CRITICAL JSON FORMATTING RULE**: 
- Inside "analysis" and "reason" strings, **YOU MUST ESCAPE DOUBLE QUOTES** (e.g., use `\\"` or single quotes `'`).
- Do NOT output Markdown code blocks.

{{
  "analysis": "<Step-by-step reasoning: 1. Understand the intent of the '+' lines. 2. Check for missing context (if yes, ignore). 3. Check for style issues (if yes, ignore). 4. Verify if a critical runtime error exists.>",
  "bug_summary": "<Short description of the critical bug, or empty string>",
  "evidence": [
    {{
      "diff_code": "<Exact code line>",
      "reason": "<Why this causes a crash or logic failure>",
      "severity": "CRITICAL" | "MAJOR" | "MINOR",
      "confidence": "HIGH" | "MEDIUM" | "LOW"
    }}
  ],
  "introduces_bug": "yes" | "no"
}}


"""), ("human", """
Now analyze the inputs below.
---INPUTS---
{DATAS}

---END---
    """)])
    
    llm = ChatOpenAI(
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",  
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        model="qwen-max",
        temperature=0
    )

    try:
        result = await (prompt | llm).ainvoke({"DATAS":datas[:30000]})
    except Exception as e:
        logging.error(f"Error querying context: {e}")
        logging.error(traceback.format_exc())

        raise e

    return result.content


def get_data_context(meta_datas):

    for meta_data in meta_datas:
        group_datas=[]
        #只有添加代码才会引入bug
        if(len(meta_data["add_codes"]))==0:
            continue
        PATH=meta_data["file_path"]
        PROGRAM_SLICE=meta_data["slices"]
        PROGRAM_SLICE=clean_code_str(PROGRAM_SLICE)
        PROGRAM_SLICE=PROGRAM_SLICE.replace("\\n","\n")

        DATA=f'''PATH:{PATH},
          PROGRAM_SLICE:\n"{PROGRAM_SLICE}"
            '''
        group_datas.append(DATA)
    datas="".join(group_datas)

    return datas


async def context_predict_buggy(project, parent_hash, commit_hash):
    meta_datas = prepare_meta_context(project, parent_hash, commit_hash)
    datas = get_data_context(meta_datas)

    if len(meta_datas)==0 or datas.strip()=="":
        return False, "No valid context to analyze.(only no semantic diff)"
    res = await query_context(datas)
    
    try:
        # 1. 寻找 JSON 的起始和结束位置
        start_idx = res.find('{')
        end_idx = res.rfind('}')
        
        # 2. 只有当找到了括号，且顺序正确（{ 在 } 前面）时才提取
        if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
            # 提取纯净的 JSON 字符串 (+1 是因为切片是左闭右开)
            clean_json_str = res[start_idx : end_idx + 1]

            # 3. 解析 JSON
            json_res = json.loads(clean_json_str)
            
            # 4. 获取结果 (使用 .get 避免 Key Error)
            predict_buggy = json_res.get("introduces_bug") == "yes"
            return predict_buggy, clean_json_str
            
        elif start_idx != -1 and end_idx != -1 and start_idx > end_idx:
            # 括号顺序错误，打印错误信息并返回
            logging.error(f"Braces order is incorrect. Response snippet: {res[:100]}")
            return False, ""
        else:
            logging.error(f"Cannot find valid JSON braces in response. Response snippet: {res[:100]}")
            return False, ""
            

    except json.JSONDecodeError as e:
        logging.error(f"JSON parsing failed: {e}. Extracted string was: {res[start_idx:end_idx+1] if 'start_idx' in locals() else 'N/A'}")
        return False, ""
    except Exception as e:
        logging.error(f"Unexpected error in context_predict_buggy: {e}")
        return False, ""

SEM = asyncio.Semaphore(5)
@retry(
    stop=stop_after_attempt(5),                # 最多重试5次
    wait=wait_exponential(multiplier=1, min=2, max=30),  # 指数退避
    retry=retry_if_exception_type(openai.RateLimitError)
)
async def safe_generate_prediction(project, parent_hash, commit_hash):
    async with SEM:  # 控制并发
        buggy, res = await context_predict_buggy(project, parent_hash, commit_hash)
        await asyncio.sleep(random.uniform(0.5, 1.5))  # 在请求间随机等待，减少撞限
        return buggy, res
            
async def quickly_start_context():
    res_path='predictions.csv'
    df=pd.read_csv(res_path)
    df['context_predicted'] = df['context_predicted'].astype(object)
    df['context_reason'] = df['context_reason'].astype(object)
    async def process_one(index, row):
        project = row['project']
        parent_hash = row['parent_hashes']
        commit_hash = row['commit_hash']
        buggy=row['context_predicted']
        reason=row['context_reason']
        ground_truth=row["is_buggy_commit"]==1.0
        if pd.notna(reason) and str(reason).strip() != "" :
            return index, buggy, reason

        try:
            buggy, reason = await safe_generate_prediction(project, parent_hash, commit_hash)
            return index, buggy, reason
        except Exception as e:
            traceback.print_exc()
            logging.error(f"Error processing row {index}: {e}")
            logging.error(e)
            return index, None, None
        
    try:
      tasks = [process_one(index, row) for index, row in df.iterrows()]
      for coro in tqdm_asyncio.as_completed(tasks, total=len(tasks), desc="Generating ..."):
        index, buggy, reason = await coro
        if buggy is not None:
            df.at[index, 'context_predicted'] = 1.0 if buggy else 0.0
            df.at[index, 'context_reason'] = str(reason).replace("\n","").replace("\r","") if reason else ""
        if index%10==0:
            df.to_csv(res_path, index=False)

    finally:
        df.to_csv(res_path, index=False)

def clean_prediction():
    res_path='predictions.csv'
    df=pd.read_csv(res_path)

    df["context_predicted"] = ""
    df["context_reason"] = ""

    df.to_csv(res_path, index=False)

if __name__ == "__main__":
    # writegraph()

    # clean_prediction()
    asyncio.run(quickly_start_context())
    # generate_divergent_output()
    #a,res=asyncio.run(predict_buggy('commons-math','3491906594b069a042f38e7526ca67f46efac2f4','bb014c3ebfe20bf7a7fce5e7a8124121de695350',query_func=query_not_rag_slice,data_get_func=process_not_rag_slice))
    # a,res=asyncio.run(predict_buggy('commons-jcs','d4ed250a0f508e8e3477342d00b090ee3858a40b','67f0a2dca2686b5b740273fb9f20d32ecdd1e5c3'))
    # print(a)
    # print(res)

    # meta_datas = prepare_meta('commons-jcs','d4ed250a0f508e8e3477342d00b090ee3858a40b','67f0a2dca2686b5b740273fb9f20d32ecdd1e5c3')
