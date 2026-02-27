import json
import pickle
import sys
from pathlib import Path
import numpy as np


sys.path.append(str(Path(__file__).resolve().parent.parent))
from networkx.readwrite import json_graph
from util.util import CONSTANTS,preprocess_code_line
import os
import random
import torch
from eval_test import commit_with_codes, deal_with_attns
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.nn import BCELoss,BCEWithLogitsLoss
from torch_geometric.data import Data

from torch.utils.data import DataLoader
from torch_geometric.data import Batch
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.metrics import f1_score, confusion_matrix
from torch.utils.data import WeightedRandomSampler
from transformers import AutoModel
import pandas as pd
import logging
from process_data import get_datas_2
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training2.log", mode='a'), 
        logging.StreamHandler(sys.stdout)          
    ]
)

logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
special_tokens_dict = {'additional_special_tokens': ["[ADD]", "[DEL]"]}
tokenizer.add_special_tokens(special_tokens_dict)




def set_seed(seed_value):
    random.seed(seed_value)

    np.random.seed(seed_value)

    torch.manual_seed(seed_value)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    
MY_SEED = 42 
set_seed(MY_SEED)
from model import GGNNNet

def collate_with_diff(data_list):

    batch = Batch.from_data_list(data_list)

    # 动态计算 diff_idx 的全局偏移
    diff_idx_list = []
    buggy_idx_list=[]
    node_ptr = batch.ptr  # ptr[i] 是第 i 个图的起始节点索引，长度为 len(data_list)+1
    
    for i, data in enumerate(data_list):
        if hasattr(data, 'diff_idx') and data.diff_idx is not None:
            # 将局部索引转为全局索引
            global_diff_idx = data.diff_idx + node_ptr[i]
            diff_idx_list.append(global_diff_idx)
        if hasattr(data, 'buggy_idx') and data.diff_idx is not None:
            global_buggy_idx = data.buggy_idx + node_ptr[i]
            buggy_idx_list.append(global_buggy_idx)        
    if diff_idx_list:
        batch.diff_idx = torch.cat(diff_idx_list, dim=0)
    else:
        batch.diff_idx = torch.tensor([], dtype=torch.long)
    
    if buggy_idx_list:
        batch.buggy_idx=torch.cat(buggy_idx_list, dim=0)
    else:
        batch.buggy_idx=torch.tensor([], dtype=torch.long)
    
    batch.diff_inputs = {
        'input_ids': batch.diff_input_ids.view(batch.batch_size, -1),
        'attention_mask': batch.diff_attention_mask.view(batch.batch_size, -1),
    }

    return batch

threshold = 0.5
best_model_path = "best_model.pt"
def ggnn_train(train_dataset,tokenizer, val_dataset=None,contras_learn=False):
    bs=64

    sample_weights = [1.0 if data.y.item() == 0 else 1.0 for data in train_dataset]  
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=False)
    train_loader = DataLoader(train_dataset, batch_size=bs, sampler=sampler, collate_fn=collate_with_diff,num_workers=4)
    max_steps = len(train_loader) * 30
    val_loader = DataLoader(val_dataset, batch_size=bs*4,collate_fn=collate_with_diff,num_workers=4) if val_dataset is not None else None
    should_valid=len(train_loader) // 5
    model = GGNNNet(768, 768, 768, num_edge_types=3,device=device,tokenizer=tokenizer)


    model=model.to(device)

    optimizer = AdamW([
    {
        'params': [p for n, p in model.named_parameters() if 'codebert' in n], 'lr': 2e-5  
    },
    {
        'params': [p for n, p in model.named_parameters() if 'codebert' not in n], 'lr': 1e-4  
    }
    ])
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0,
                                                num_training_steps=max_steps)
    
    pos_weight = torch.tensor([1]).to(device) 
    criterion = BCEWithLogitsLoss(pos_weight=pos_weight)

    best_f1 = 0  
    best_auc_record = 0.0 
    auc_retention_rate = 0.99
    print("strat training")
    model.set_codebert_frozen(False)

    patient=0
    for epoch in range(50):
        total_loss = 0
        max_patient=25
        for step,batch in enumerate(tqdm(train_loader, desc="Training")):
            model.train()
            batch = batch.to(device)
            x_graph,diff_feature,out,_ = model(batch)
            cls_loss = criterion(out, batch.y.unsqueeze(1).float().to(device))
            if contras_learn:

                contrastive_loss = compute_contrastive_loss(x_graph, batch.y)*0.5+compute_contrastive_loss(diff_feature,batch.y)*0.5
                loss = cls_loss+contrastive_loss*0.1
                print(f"cls loss:{cls_loss},contras loss:{contrastive_loss}")
            else:
                loss = cls_loss
            loss.backward()


            clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

            total_loss += loss.item()

            if  epoch<=0 or (step+1)%should_valid!=0:
                continue

        # ---------- Validate ----------
            if val_loader is not None:
                model.eval()
                val_loss = 0
                total = 0
                all_probs=[]
                all_preds = []
                all_labels = []
                all_hashes = []

                with torch.no_grad():
                    for valbatch in val_loader:
                        valbatch = valbatch.to(device)
                        _,_,val_out,_ = model(valbatch)
       
                        val_prob = torch.sigmoid(val_out)
                        valcls_loss = criterion(val_out, valbatch.y.unsqueeze(1).float())
                        val_loss += valcls_loss.item()
                        
                        total += valbatch.y.size(0)

                     
                        all_probs.extend(val_prob.cpu())
                        all_labels.extend(valbatch.y.cpu().numpy())
                        all_hashes.extend(valbatch.commit_hash)
                
                avg_val_loss = val_loss / len(val_loader)

                all_preds = [(a > threshold).long().squeeze() for a in all_probs]
                val_f1 = process_results(all_hashes, all_preds,all_probs=all_probs,what="valid",max_f1=best_f1)
                try:
                    val_auc = roc_auc_score(all_labels, all_probs)
                except ValueError:
                    val_auc = 0.5
                if val_auc > best_auc_record:
                    best_auc_record = val_auc

                is_f1_better = val_f1 > best_f1
                is_auc_stable = val_auc >= (best_auc_record*auc_retention_rate)
                if is_f1_better and is_auc_stable:
                    best_f1 = val_f1
                    torch.save(model.state_dict(), best_model_path)
                    patient = 0
                    logger.info(f"✨ Saved Best Model! F1: {val_f1:.4f} (AUC: {val_auc:.4f} is stable, Loss: {avg_val_loss:.4f})")
                
                elif is_f1_better and not is_auc_stable:
                    patient += 1
                    logger.info(f"⚠️ F1 improved ({val_f1:.4f}) but AUC dropped significantly ({val_auc:.4f} < {best_auc_record:.4f}). Ignored.")
                
                else:
                    patient += 1
                if patient > max_patient:
                    logger.info(f"🛑 Early stopping triggered. Loss hasn't improved for {max_patient} validations.")
                    
                    return
                
        avg_train_loss=round(total_loss / len(train_loader), 4)
        logger.info(f"Epoch {epoch}, Avg Loss: {avg_train_loss}")

def tran_start():
    # repo_dir = CONSTANTS.repository_dir 
    #tran_dataset = get_datas_2("train", Path(repo_dir)/"changes_train.pkl", Path(repo_dir)/"train_graph_dataset.jsonl",tokenizer,device)
    #get_datas_2("test", Path(repo_dir)/"changes_test.pkl", Path(repo_dir)/"test_graph_dataset.jsonl",tokenizer,device)
    #valid_dataset = get_datas_2("valid", Path(repo_dir)/"changes_valid.pkl", Path(repo_dir)/"valid_graph_dataset.jsonl",tokenizer,device)
    tran_dataset = torch.load("train_.pt",weights_only=False)
    valid_dataset = torch.load("valid_.pt",weights_only=False)

    ggnn_train(tran_dataset, tokenizer,val_dataset=valid_dataset)
def test_start(model=None,name='test',test_dataset=None):
    from eval_test import eval_result
    bs=128
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if test_dataset is None:
        test_dataset = torch.load(f'{name}_.pt', weights_only=False)
    test_loader = DataLoader(test_dataset, batch_size=bs,collate_fn=collate_with_diff)
    if model is None:
        model=GGNNNet(768, 768, 768, num_edge_types=3,device=device,tokenizer=tokenizer).to(device)
        model.load_state_dict(torch.load(best_model_path))


    model.eval()

    all_preds = []
    all_probs = []
    all_labels = []
    all_hashes = []
    attns = []
    # total_valid_buggy_graphs = 0
    # sum_precision_top5 = 0.0
    # sum_precision_top10 = 0.0
    for batch in test_loader:
        with torch.no_grad():
            batch = batch.to(device)
            _,_,out,attn_weights = model(batch,True)
            prob = torch.sigmoid(out)
            all_probs.extend(prob.cpu().numpy())
            preds = (prob > threshold).long().squeeze(1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch.y.cpu().numpy())
            all_hashes.extend(batch.commit_hash)
            attns.append(attn_weights.cpu().numpy())
        
    process_results(all_hashes, all_preds,all_probs,what=name)
    result_path = f"./test_prediction_results2.csv"
    features_path=Path(CONSTANTS.repository_dir)/"features_test.pkl"
    eval_result(result_path,features_path)


    attns = np.concatenate(attns, 0)

    y_preds = all_preds

    cache_buggy_line = Path(CONSTANTS.repository_dir)/'changes_complete_buggy_line_level_cache.pkl'
    if os.path.exists(cache_buggy_line):
        commit2codes, idx2label = pickle.load(open(cache_buggy_line, 'rb'))
    else:
        buggy_line_filepath=Path(CONSTANTS.repository_dir)/'changes_complete_buggy_line_level.pkl'
        commit2codes, idx2label = commit_with_codes(buggy_line_filepath, tokenizer)
        pickle.dump((commit2codes, idx2label), open(cache_buggy_line, 'wb'))

    IFA, top_20_percent_LOC_recall, effort_at_20_percent_LOC_recall, top_10_acc, top_5_acc = [], [], [], [], []
    for example, pred, prob, attn in zip(test_dataset, y_preds, all_probs, attns):

        # calculate
        add_token_id = tokenizer.convert_tokens_to_ids('[ADD]')
        has_add_token = (example.diff_input_ids == add_token_id).any()
        if int(example.y) == 1 and int(pred) == 1 and  has_add_token:
            cur_codes = commit2codes[commit2codes['commit_id'] == example.commit_hash]
            cur_labels = idx2label[idx2label['commit_id'] == example.commit_hash]
            cur_IFA, cur_top_20_percent_LOC_recall, cur_effort_at_20_percent_LOC_recall, cur_top_10_acc, cur_top_5_acc = deal_with_attns(
                example, attn,
                cur_codes,cur_labels,
                tokenizer,True)
            IFA.append(cur_IFA)
            top_20_percent_LOC_recall.append(cur_top_20_percent_LOC_recall)
            effort_at_20_percent_LOC_recall.append(cur_effort_at_20_percent_LOC_recall)
            top_10_acc.append(cur_top_10_acc)
            top_5_acc.append(cur_top_5_acc)

    with open("test@5_line", 'a') as f:
        f.write(
            f'Top-10-ACC: {np.mean(top_10_acc):.4f}, '
            f'Top-5-ACC: {np.mean(top_5_acc):.4f}, '
            f'Recall20%Effort: {np.mean(top_20_percent_LOC_recall):.4f}, '
            f'Effort@20%LOC: {np.mean(effort_at_20_percent_LOC_recall):.4f}, '
            f'IFA: {np.mean(IFA):.4f}\n'
        )

    IFA, top_20_percent_LOC_recall, effort_at_20_percent_LOC_recall, top_10_acc, top_5_acc = [], [], [], [], []
    for example, pred, prob, attn in zip(test_dataset, y_preds, all_probs, attns):
        # calculate
        add_token_id = tokenizer.convert_tokens_to_ids('[ADD]')
        has_add_token = (example.diff_input_ids == add_token_id).any()
        if  int(pred) == 1 and  has_add_token :
            cur_codes = commit2codes[commit2codes['commit_id'] == example.commit_hash]
            cur_labels = idx2label[idx2label['commit_id'] == example.commit_hash]
            cur_IFA, cur_top_20_percent_LOC_recall, cur_effort_at_20_percent_LOC_recall, cur_top_10_acc, cur_top_5_acc = deal_with_attns(
                example, attn,
                cur_codes,cur_labels,
                tokenizer,True)
            IFA.append(cur_IFA)
            top_20_percent_LOC_recall.append(cur_top_20_percent_LOC_recall)
            effort_at_20_percent_LOC_recall.append(cur_effort_at_20_percent_LOC_recall)
            top_10_acc.append(cur_top_10_acc)
            top_5_acc.append(cur_top_5_acc)

    with open("test@5_line", 'a') as f:
        f.write(
            f'Top-10-ACC-a: {np.mean(top_10_acc):.4f}, '
            f'Top-5-ACC-a: {np.mean(top_5_acc):.4f}, '
            f'Recall20%Effort-a: {np.mean(top_20_percent_LOC_recall):.4f}, '
            f'Effort@20%LOC-a: {np.mean(effort_at_20_percent_LOC_recall):.4f}, '
            f'IFA-a: {np.mean(IFA):.4f}\n'
        )


def process_results(all_hashes, all_preds,all_probs=None,what="test",max_f1=0):
    n=len(all_hashes)
    hashes=[]
    real_label=[]
    real_probs=[]
    pred_label=[]
    with open(Path(CONSTANTS.repository_dir)/f"features_{what}.json", 'r',encoding='utf-8') as f:
        data = json.load(f)
        
        for i in range(n):
            hash=all_hashes[i]
            hashes.append(hash)
            pred_label.append(all_preds[i])
            real_label.append(int(float(data[hash]["is_buggy_commit"])))

            if all_probs is not None:
                real_probs.append(float(all_probs[i]))
                
    output_path = f"./{what}_prediction_results2.csv"
    
    real_label = np.array(real_label).astype(int)
    pred_label = np.array(pred_label).astype(int)
    f1 = f1_score( real_label, pred_label, average='binary')
    tn, fp, fn, tp = confusion_matrix(real_label, pred_label).ravel()
    logger.info(f" F1={f1:.4f}, "
                f"TP={tp}, FN={fn}, FP={fp}, TN={tn}")
    if f1>max_f1:
        max_f1=f1
        df = pd.DataFrame({
        "commit_hash": hashes,
        "prob": real_probs,
        "pred_label": pred_label,
        "real_label": real_label,
        })
        df.to_csv(output_path, index=False, encoding='utf-8')
        # print(f"✅ 预测结果已保存到: {output_path}")
        return f1
    return max_f1
def compute_contrastive_loss(embeddings, labels, temperature=0.1):
    """
    embeddings: [B, D] 图嵌入向量
    labels: [B] 每个图的标签
    """
    embeddings = F.normalize(embeddings, dim=1)
    sim_matrix = torch.matmul(embeddings, embeddings.T) / temperature  # [B, B]

    # 构造正样本 mask
    labels = labels.contiguous().view(-1, 1)
    mask = torch.eq(labels, labels.T).float().to(embeddings.device)#[B,B]
    # 防止梯度爆炸
    logits_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)
    sim_matrix = sim_matrix - logits_max.detach()
    exp_sim = torch.exp(sim_matrix) * (1 - torch.eye(labels.size(0), device=embeddings.device))

    log_prob = sim_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-9)

    mean_log_prob_pos = (mask * log_prob).sum(dim=1) / (mask.sum(dim=1) + 1e-9)

    # InfoNCE 损失
    loss = -mean_log_prob_pos.mean()
    return loss



# read_graph('train2')
# read_graph('valid2')
# read_graph('test2')
if __name__ == "__main__":
    for j in range(5):
        tran_start()
        test_start()

