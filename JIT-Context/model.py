
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from torch_geometric.nn import MessagePassing
import torch.nn as nn
import torch
from torch_geometric.nn import global_mean_pool,global_max_pool,global_add_pool
from transformers import AutoModel
from torch_geometric.utils import softmax
from torch_scatter import scatter_max, scatter_add,scatter_mean, scatter_std

class WeightedGGNN(MessagePassing):
    def __init__(self, in_channels, out_channels, hidden_dim, num_edge_types):
        super(WeightedGGNN, self).__init__(aggr='add') #add 聚合
        
        
        self.lin_message = nn.Linear(in_channels, out_channels)
        self.att_itself=nn.Linear(in_channels, out_channels)
   
        self.edge_type_att = nn.Parameter(torch.Tensor(num_edge_types, out_channels))
        nn.init.xavier_uniform_(self.edge_type_att)

        self.gru = nn.GRUCell(out_channels, out_channels)
        
        self.diff_tag_embedding = nn.Parameter(torch.randn(hidden_dim))
        self.dropout = nn.Dropout(p=0.2)
    def forward(self, x, edge_index, edge_type, diff_idx=None):
        # x: [num_nodes, 768]
        
        out = x
        for _ in range(3):
            m = self.propagate(edge_index, x=out, edge_type=edge_type)
            out = self.gru(m, out)

        return out

    def message(self, x_j, edge_type):

        msg = self.lin_message(x_j) 
        

        edge_att = self.edge_type_att[edge_type] # [num_edges, 768]
        

        msg = msg * edge_att 
        
        return msg
class GGNNNet(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_dim,num_edge_types,device="cuda",tokenizer=None):
        super().__init__()
        self.ggnn = WeightedGGNN(in_channels, out_channels, hidden_dim,num_edge_types)
        self.fc = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(768*2,768*2),
            nn.ReLU(),
            nn.Linear(768*2,768*2),
            nn.ReLU(),
            nn.Linear(768*2,768*2),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(768*2, 1),
            )


        self.reflect1=nn.Sequential( 

            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.reflect2=nn.Sequential( 

            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),

        )
        self.device=device
        self.tokenizer=tokenizer
        self.codebert=AutoModel.from_pretrained("microsoft/codebert-base",use_safetensors=True,trust_remote_code=True).to(device)
        self.codebert.resize_token_embeddings(len(tokenizer))
    
    
    def set_codebert_frozen(self, frozen=True):
        for param in self.codebert.parameters():
            param.requires_grad = not frozen 
    def _generate_vector(self,inputs,attn=False):
        outputs = self.codebert(**inputs, output_attentions=attn)
        return outputs
    def forward(self, data,attn=False):

        outputs = self._generate_vector(data.diff_inputs,attn=attn)
        diff_feature = outputs.last_hidden_state[:, 0, :]
        current_batch_size = diff_feature.size(0)
        diff_feature=self.reflect1(diff_feature)
        if data.x.size(0) == 0:

            x_graph = torch.zeros(
                current_batch_size, 
                768, 
                device=self.device
            )
        else:

            x = self.ggnn(data.x, data.edge_index, data.edge_type, data.diff_idx)
            x_diff_nodes = x[data.diff_idx]
            
      
            batch_diff_nodes = data.batch[data.diff_idx]
            
            raw_pool = global_max_pool(x_diff_nodes, batch_diff_nodes, size=current_batch_size)

            mask = torch.isinf(raw_pool)
            x_graph = raw_pool.masked_fill(mask, 0.0)

        x_graph = self.reflect2(x_graph)
        
        contat_features = torch.cat([x_graph, diff_feature], dim=1)
        prob = self.fc(contat_features)
        last_layer_attn_weights = outputs.attentions[self.codebert.config.num_hidden_layers - 1][:, :,0].detach() if attn else None
        return x_graph, diff_feature, prob,last_layer_attn_weights
    
    # def localize(self, data):

    #     self.eval()
    #     self.zero_grad()
    #     batch_size=0
    #     with torch.no_grad():
    #         diff_feature = self._generate_vector(data.diff_inputs)
    #         diff_feature = self.reflect1(diff_feature)
    #         batch_size = diff_feature.size(0)
    #     node_embeddings = self.ggnn(data.x, data.edge_index, data.edge_type, data.diff_idx)
        
    #     
    #     pooled_features, argmax = scatter_max(node_embeddings, data.batch, dim=0, dim_size=batch_size)

        
    #    
    #     empty_graph_mask = pooled_features < -1e4 
    #     pooled_features = pooled_features.masked_fill(empty_graph_mask, 0.0)
        
    #     pooled_features.retain_grad()

    #     x_graph = self.reflect2(pooled_features)
    #     contat_features = torch.cat([x_graph, diff_feature], dim=1)
    #     output_logits = self.fc(contat_features)
    #     (output_logits.sum()).backward()

    #     #每一维的梯度
    #     gradients = pooled_features.grad
    #     node_grad = gradients[data.batch]

    #     total_nodes = node_embeddings.size(0)

    #     node_max = pooled_features[data.batch]
    #     eps = 1e-8
    #     pos_mask = (node_grad > 0).float()
    #     score_pos = node_grad 
    #     score_pos = score_pos * pos_mask*(-1)

    #     neg_mask = (node_grad < 0).float()
    #     score_neg = node_grad.abs() * (node_max -node_embeddings + eps)
    #     score_neg = score_neg * neg_mask

    #     node_scores = torch.zeros(total_nodes, device=self.device)
    #     node_scores = (score_pos + score_neg).sum(dim=1)

    #     diff_mask = torch.zeros(total_nodes, device=self.device)
    #     diff_mask[data.diff_idx] = 1.0


    
    #     src, dst = data.edge_index
    #     context_mask = 1.0 - diff_mask # 非 Diff 节点为 1

    #     context_scores = node_scores * context_mask
    #     score_messages = context_scores[src]
    #     aggregated_scores = scatter_mean(score_messages, dst, dim=0, dim_size=total_nodes)
    #     score_messages = context_scores[dst]
    #     aggregated_scores+=scatter_mean(score_messages, src, dim=0, dim_size=total_nodes)
    #     node_scores = node_scores + aggregated_scores

    #     # score_messages = context_scores[src]
    #     # aggregated_scores+= scatter_mean(score_messages, dst, dim=0, dim_size=total_nodes)
    #     # score_messages = context_scores[dst]
    #     # aggregated_scores+=scatter_mean(score_messages, src, dim=0, dim_size=total_nodes)
    #     # node_scores = node_scores + aggregated_scores

    #     node_scores = node_scores * diff_mask


    #     batch_mean = scatter_mean(node_scores, data.batch, dim=0, dim_size=batch_size)
        

    #     batch_sq_mean = scatter_mean(node_scores ** 2, data.batch, dim=0, dim_size=batch_size)
    #     batch_std = (batch_sq_mean - batch_mean ** 2).clamp(min=1e-9).sqrt()
        
    #     mean_per_node = batch_mean[data.batch]
    #     std_per_node = batch_std[data.batch]
        
    #     z_scores = (node_scores - mean_per_node) / (std_per_node + 1e-9)

    #     probs = torch.sigmoid(z_scores)
    #     return probs.detach().cpu().numpy().round(decimals=2)
       