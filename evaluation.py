import json
from pathlib import Path
import pickle
import traceback
from util.util import CONSTANTS
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score
def eval_res(pattern ="contxt"):
    res_path="predictions.csv"
    data=pd.read_csv(res_path)
    ground_truth=data['is_buggy_commit']
    DSR_prediction=data[f"{pattern}_predicted"]
    f1 = f1_score(ground_truth, DSR_prediction)
    auc = roc_auc_score(ground_truth, DSR_prediction)
    print(f"F1 Score: {f1:.4f}")
    print(f"AUC Score: {auc:.4f}")
def commit_with_codes(filepath):
    data = pd.read_pickle(filepath)
    commit2codes = []
    idx2label = []
    for _, item in data.iterrows():
        
        commit_id, idx, changed_type, label, raw_changed_line, changed_line = item

        commit2codes.append([commit_id, idx, changed_type,raw_changed_line,label])
        idx2label.append([commit_id, idx, label])
    commit2codes = pd.DataFrame(commit2codes, columns=['commit_id', 'idx', 'changed_type', 'changed_line','label'])
    return commit2codes

def eval_line_level_metrics(pattern="contxt",only_hit=True):
    file_path=Path(CONSTANTS.repository_dir) / "changes_complete_buggy_line_level.pkl"
    commit2codes = commit_with_codes(file_path)
    res_path ="predictions.csv"
    data = pd.read_csv(res_path)
    acc_5 = 0.0
    acc_10 = 0.0
    sample_count = 0
    for index, row in data.iterrows():
        ground_truth = row["is_buggy_commit"] == 1.0
        prediction = row[f"{pattern}_predicted"] == 1.0
        commit_hash = row["commit_hash"]
        cur_codes=commit2codes[commit2codes['commit_id']==commit_hash]
        cur_codes=cur_codes[cur_codes['label']==1]
        context_reason_str = row[f"{pattern}_reason"]
        match_count = 0
        if (only_hit and prediction and ground_truth) or (not only_hit and prediction) :
            sample_count += 1
            if not ground_truth:
                continue
            try:
                reason_dict = json.loads(context_reason_str)
                buggy_prediction_lines=[]
                for line in reason_dict["evidence"]:
                    buggy_prediction_lines.append(line["diff_code"][1:] if line["diff_code"].startswith("+") or line["diff_code"].startswith("-") else line["diff_code"])
                clean_preds = [line.replace(" ", "").replace("\t", "") for line in buggy_prediction_lines]
                clean_ground_truth = cur_codes["changed_line"].astype(str).str.replace(" ", "").str.replace("\t", "")
                # print(clean_preds)
                # print(clean_ground_truth)
                n=len(clean_ground_truth)

                for gt_line in clean_ground_truth:
                    if not gt_line: 
                        continue
                    is_hit = any(gt_line in pred_line for pred_line in clean_preds)
                    if is_hit:
                        match_count += 1
                if match_count >= 5:
                    acc_5 += 1
                else:
                    if n>=5:
                        acc_5 += match_count / 5
                    else:
                        acc_5 += match_count / n
                if match_count >= 10:
                    acc_10 += 1
                else:
                    if n>=10:
                        acc_10 += match_count / 10
                    else:
                        acc_10 += match_count / n
                
            except json.JSONDecodeError:
               print(context_reason_str)
               print(f"JSON解析失败,Commit: {commit_hash}")
            except Exception as e:
               print(f"发生错误: {e}")

    print(f"Acc@5: {acc_5 / sample_count:.4f}")
    print(f"Acc@10: {acc_10 / sample_count:.4f}")
print("Simple Metrics:")
eval_res("simple")
eval_line_level_metrics("simple",only_hit=False)
print("\nContext Metrics:")
eval_res("context")

eval_line_level_metrics("context",only_hit=False)
