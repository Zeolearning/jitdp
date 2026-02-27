import sys
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score, recall_score,precision_recall_fscore_support,confusion_matrix,auc
import logging
import math
import numpy as np
from sklearn.preprocessing import MinMaxScaler



def convert_dtype_dataframe(df, feature_name):
    df['fix'] = df['fix'].apply(lambda x: float(bool(x)))
    df = df.astype({i: 'float32' for i in feature_name})
    return df
def load_change_metrics_df(data_dir, mode='train'):
    change_metrics = pd.read_pickle(data_dir)
    feature_name = ["ns", "nd", "nf", "entropy", "la", "ld", "lt", "fix", "ndev", "age", "nuc", "exp", "rexp", "sexp"]
    change_metrics = convert_dtype_dataframe(change_metrics, feature_name)

    return change_metrics[['commit_hash'] + feature_name]


def get_recall_at_k_percent_effort(percent_effort, result_df_arg, real_buggy_commits):
    cum_LOC_k_percent = (percent_effort / 100) * result_df_arg.iloc[-1]['cum_LOC']
    buggy_line_k_percent = result_df_arg[result_df_arg['cum_LOC'] <= cum_LOC_k_percent]
    buggy_commit = buggy_line_k_percent[buggy_line_k_percent['label'] == 1]
    recall_k_percent_effort = len(buggy_commit) / float(len(real_buggy_commits))

    return recall_k_percent_effort


def eval_metrics(result_df):
    pred = result_df['defective_commit_pred']
    y_test = result_df['label']

    prec, rec, f1, _ = precision_recall_fscore_support(y_test, pred, average='binary')  # at threshold = 0.5
    tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()
   

    FAR = fp / (fp + tn)  # false alarm rate
    dist_heaven = math.sqrt((pow(1 - rec, 2) + pow(0 - FAR, 2)) / 2.0)  # distance to heaven

    AUC = roc_auc_score(y_test, result_df['defective_commit_prob'])

    result_df['defect_density'] = result_df['defective_commit_prob'] / result_df['LOC']  # predicted defect density
    result_df['actual_defect_density'] = result_df['label'] / result_df['LOC']  # defect density

    result_df = result_df.sort_values(by='defect_density', ascending=False)
    actual_result_df = result_df.sort_values(by='actual_defect_density', ascending=False)
    actual_worst_result_df = result_df.sort_values(by='actual_defect_density', ascending=True)

    result_df['cum_LOC'] = result_df['LOC'].cumsum()
    actual_result_df['cum_LOC'] = actual_result_df['LOC'].cumsum()
    actual_worst_result_df['cum_LOC'] = actual_worst_result_df['LOC'].cumsum()

    real_buggy_commits = result_df[result_df['label'] == 1]

    label_list = list(result_df['label'])

    all_rows = len(label_list)

    # find Recall@20%Effort
    cum_LOC_20_percent = 0.2 * result_df.iloc[-1]['cum_LOC']
    buggy_line_20_percent = result_df[result_df['cum_LOC'] <= cum_LOC_20_percent]
    buggy_commit = buggy_line_20_percent[buggy_line_20_percent['label'] == 1]
    recall_20_percent_effort = len(buggy_commit) / float(len(real_buggy_commits))

    # find Effort@20%Recall
    buggy_20_percent = real_buggy_commits.head(math.ceil(0.2 * len(real_buggy_commits)))
    buggy_20_percent_LOC = buggy_20_percent.iloc[-1]['cum_LOC']
    effort_at_20_percent_LOC_recall = int(buggy_20_percent_LOC) / float(result_df.iloc[-1]['cum_LOC'])

    # find P_opt
    percent_effort_list = []
    predicted_recall_at_percent_effort_list = []
    actual_recall_at_percent_effort_list = []
    actual_worst_recall_at_percent_effort_list = []

    for percent_effort in np.arange(10, 101, 10):
        predicted_recall_k_percent_effort = get_recall_at_k_percent_effort(percent_effort, result_df,
                                                                           real_buggy_commits)
        actual_recall_k_percent_effort = get_recall_at_k_percent_effort(percent_effort, actual_result_df,
                                                                        real_buggy_commits)
        actual_worst_recall_k_percent_effort = get_recall_at_k_percent_effort(percent_effort, actual_worst_result_df,
                                                                              real_buggy_commits)

        percent_effort_list.append(percent_effort / 100)

        predicted_recall_at_percent_effort_list.append(predicted_recall_k_percent_effort)
        actual_recall_at_percent_effort_list.append(actual_recall_k_percent_effort)
        actual_worst_recall_at_percent_effort_list.append(actual_worst_recall_k_percent_effort)

    p_opt = 1 - ((auc(percent_effort_list, actual_recall_at_percent_effort_list) -
                  auc(percent_effort_list, predicted_recall_at_percent_effort_list)) /
                 (auc(percent_effort_list, actual_recall_at_percent_effort_list) -
                  auc(percent_effort_list, actual_worst_recall_at_percent_effort_list)))

    return f1, AUC, recall_20_percent_effort, effort_at_20_percent_LOC_recall, p_opt


def eval_result(result_path, features_path):

    RF_result = pd.read_csv(result_path )

    RF_result.columns = ['test_commit', 'defective_commit_prob', 'defective_commit_pred', 'label']  # for new result

    test_commit_metrics = load_change_metrics_df(features_path, 'test')[['commit_hash', 'la', 'ld']]
    RF_df = pd.DataFrame()
    RF_df['commit_id'] = RF_result['test_commit']
    RF_df = pd.merge(RF_df, test_commit_metrics, left_on='commit_id', right_on='commit_hash', how='inner')
    RF_df = RF_df.drop('commit_hash', axis=1)
    RF_df['LOC'] = RF_df['la'] + RF_df['ld']
    RF_result = pd.merge(RF_df, RF_result, how='inner', left_on='commit_id', right_on='test_commit')
    f1, auc, recall_20_percent_effort, effort_at_20_percent_LOC_recall, p_opt = eval_metrics(RF_result)
    with open("test@5_line", 'a') as f:
        f.write(f'F1: {f1:.4f}, AUC: {auc:.4f}, PCI@20%LOC: {recall_20_percent_effort:.4f}, Effort@20%Recall: {effort_at_20_percent_LOC_recall:.4f}, POpt: {p_opt:.4f}\n')

def get_line_level_metrics(line_score, label):
    scaler = MinMaxScaler()
    line_score = scaler.fit_transform(np.array(line_score).reshape(-1, 1))  # cannot pass line_score as list T-T
    pred = np.round(line_score)

    line_df = pd.DataFrame()
    line_df['scr'] = [float(val) for val in list(line_score)]
    line_df['label'] = label
    line_df = line_df.sort_values(by='scr', ascending=False)
    line_df['row'] = np.arange(1, len(line_df) + 1)

    real_buggy_lines = line_df[line_df['label'] == 1]

    top_10_acc = 0
    top_5_acc = 0

    if len(real_buggy_lines) < 1:
        IFA = len(line_df)
        top_20_percent_LOC_recall = 0
        effort_at_20_percent_LOC_recall = math.ceil(0.2 * len(line_df))

    else:
        IFA = line_df[line_df['label'] == 1].iloc[0]['row'] - 1
        label_list = list(line_df['label'])

        all_rows = len(label_list)

        # find top-10 accuracy
        if all_rows < 10:
            top_10_acc = np.sum(label_list[:all_rows]) / len(label_list[:all_rows])
        else:
            top_10_acc = np.sum(label_list[:10]) / len(label_list[:10])

        # find top-5 accuracy
        if all_rows < 5:
            top_5_acc = np.sum(label_list[:all_rows]) / len(label_list[:all_rows])
        else:
            top_5_acc = np.sum(label_list[:5]) / len(label_list[:5])

        # find recall
        LOC_20_percent = line_df.head(int(0.2 * len(line_df)))
        buggy_line_num = LOC_20_percent[LOC_20_percent['label'] == 1]
        top_20_percent_LOC_recall = float(len(buggy_line_num)) / float(len(real_buggy_lines))

        # find effort @20% LOC recall

        buggy_20_percent = real_buggy_lines.head(math.ceil(0.2 * len(real_buggy_lines)))
        buggy_20_percent_row_num = buggy_20_percent.iloc[-1]['row']
        effort_at_20_percent_LOC_recall = int(buggy_20_percent_row_num) / float(len(line_df))

    return IFA, top_20_percent_LOC_recall, effort_at_20_percent_LOC_recall, top_10_acc, top_5_acc


def deal_with_attns(item, attns,  commit2codes, idx2label, tokenizer,only_adds=True):
    '''
    score for each token
    :param item:
    :param attns:
    :param pred:
    :param commit2codes:
    :param idx2label:
    :return:
    '''
    input_ids_list = item.diff_input_ids.cpu().tolist()
    input_tokens = tokenizer.convert_ids_to_tokens(input_ids_list)
    commit_label = int(item.y)

    # remove msg,cls,eos,del
    begin_pos = input_tokens.index('[ADD]')
    end_pos = input_tokens.index('[DEL]') if '[DEL]' in input_tokens else len(input_tokens) - 1

    attn_df = pd.DataFrame()
    attn_df['token'] = [token.replace('\u0120', '') for token in
                        input_tokens[begin_pos:end_pos]]
    # average score for multi-heads
    attns = attns.mean(axis=0)[begin_pos:end_pos]
    attn_df['score'] = attns
    attn_df = attn_df.sort_values(by='score', ascending=False)
    attn_df = attn_df.groupby('token').max()
    attn_df['token'] = attn_df.index
    attn_df = attn_df.reset_index(drop=True)
    add_count = input_tokens.count('[ADD]')
    delete_count=input_tokens.count('[DEL]')
    line_count=add_count+delete_count-2

    # calculate score for each line in commit
    if only_adds:
        commit2codes = commit2codes[commit2codes['changed_type'] == 'added']  # only count for added lines
        line_count=line_count-(delete_count-1)
    commit2codes = commit2codes.drop('commit_id', axis=1)
    commit2codes = commit2codes.drop('changed_type', axis=1)
    if commit_label==0:
        IFA=line_count
        top_20_percent_LOC_recall = 0
        effort_at_20_percent_LOC_recall = line_count
        top_10_acc=0
        top_5_acc=0

        return IFA, top_20_percent_LOC_recall, effort_at_20_percent_LOC_recall, top_10_acc, top_5_acc
    result_df = pd.merge(commit2codes, attn_df, how='left', on='token')
    result_df = result_df.groupby(['idx']).sum()
    result_df = result_df.reset_index(drop=False)

    result_df = pd.merge(result_df, idx2label, how='inner', on='idx')
    IFA, top_20_percent_LOC_recall, effort_at_20_percent_LOC_recall, top_10_acc, top_5_acc = get_line_level_metrics(
        result_df['score'].tolist(), result_df['label'].tolist())
    return IFA, top_20_percent_LOC_recall, effort_at_20_percent_LOC_recall, top_10_acc, top_5_acc


def commit_with_codes(filepath, tokenizer):
    data = pd.read_pickle(filepath)
    commit2codes = []
    idx2label = []
    for _, item in data.iterrows():
        commit_id, idx, changed_type, label, raw_changed_line, changed_line = item
        line_tokens = [token.replace('\u0120', '') for token in tokenizer.tokenize(changed_line)]
        for token in line_tokens:
            commit2codes.append([commit_id, idx, changed_type, token])
        idx2label.append([commit_id, idx, label])
    commit2codes = pd.DataFrame(commit2codes, columns=['commit_id', 'idx', 'changed_type', 'token'])
    idx2label = pd.DataFrame(idx2label, columns=['commit_id', 'idx', 'label'])
    return commit2codes, idx2label
