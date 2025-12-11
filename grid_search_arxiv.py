import os
import json
import time
import torch
import random
import itertools
import numpy as np
import pandas as pd
from datetime import datetime
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GraphNorm
import torch_geometric.transforms as T
from ogb.nodeproppred import Evaluator
from ogb.nodeproppred import NodePropPredDataset

# ==========================================
# 1. CONFIGURATION & DIRECTORIES
# ==========================================

# Suppress warnings
import warnings
warnings.filterwarnings("ignore", message=".*weights_only=False.*")

# Output structure
OUTPUT_ROOT = 'outputs'
MODELS_DIR = os.path.join(OUTPUT_ROOT, 'models')
LOGS_DIR = os.path.join(OUTPUT_ROOT, 'logs')
LOG_FILE = os.path.join(OUTPUT_ROOT, 'experiments_log.csv')

for folder in [OUTPUT_ROOT, MODELS_DIR, LOGS_DIR]:
    if not os.path.exists(folder):
        os.makedirs(folder)

# --- HYPERPARAMETER GRID (UPDATED) ---
param_grid = {
    # Architecture
    'num_layers': [3, 4],                # Now we can try deeper nets thanks to residuals
    'hidden_dim': [256, 512],  # Wider layers for better capacity     
    'use_residual': [True, False],       # Grid search will test with and without Skip Connections
    'norm_type': ['batch', 'layer', 'graph'],    
    
    # Optimization
    'lr': [0.01],                        # Fixed to reduce combinations (Scheduler handles dynamics)
    'dropout': [0.5],                    # Fixed (0.5 is usually best for arxiv)
    'drop_edge_p': [0.0, 0.2],               # Edge Dropout for regularization
    'weight_decay': [0, 5e-4],           
    
    # Training
    'epochs': [500]                      # High ceiling for Early Stopping
}

# Fixed settings
PATIENCE = 50
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ==========================================
# 2. MODEL DEFINITION (ADVANCED GCN)
# ==========================================

class GCN(torch.nn.Module):
    # 1. Adicionar drop_edge_p no init
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout, 
                 norm_type='batch', use_residual=False, drop_edge_p=0.0):
        super(GCN, self).__init__()
        
        self.convs = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        self.dropout = dropout
        self.use_residual = use_residual
        self.drop_edge_p = drop_edge_p  # Guardar o valor

        def get_norm(dim):
            if norm_type == 'batch': return torch.nn.BatchNorm1d(dim)
            elif norm_type == 'layer': return torch.nn.LayerNorm(dim)
            elif norm_type == 'graph': return GraphNorm(dim)
            else: return torch.nn.Identity()

        self.convs.append(GCNConv(input_dim, hidden_dim))
        self.norms.append(get_norm(hidden_dim))

        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.norms.append(get_norm(hidden_dim))

        self.convs.append(GCNConv(hidden_dim, output_dim))

    def forward(self, x, adj_t):
        # --- APLICAÇÃO DO DROPEDGE ---
        # Só aplicamos se estiver treinando e se o p > 0
        edge_index_to_use = adj_t
        if self.training and self.drop_edge_p > 0.0:
            # O método .dropout() do SparseTensor retorna uma nova matriz esparsa
            # com valores removidos aleatoriamente.
            edge_index_to_use = adj_t.dropout(p=self.drop_edge_p)
            
        # 1. Input Layer
        # Note que usamos edge_index_to_use agora, não mais adj_t direto
        x = self.convs[0](x, edge_index_to_use)
        x = self.norms[0](x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # 2. Hidden Layers
        for i in range(len(self.convs) - 2):
            x_in = x 
            
            x = self.convs[i+1](x, edge_index_to_use) # Usando arestas dropadas
            x = self.norms[i+1](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            
            if self.use_residual:
                x = x + x_in

        # 3. Output Layer
        x = self.convs[-1](x, edge_index_to_use) # Usando arestas dropadas
        return F.log_softmax(x, dim=1)

# ==========================================
# 3. UTILITY FUNCTIONS
# ==========================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

def train(model, data, train_idx, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.adj_t)
    loss = criterion(out[train_idx], data.y[train_idx])
    loss.backward()
    optimizer.step()
    return loss.item()

@torch.no_grad()
def test(model, data, split_idx, evaluator):
    model.eval()
    out = model(data.x, data.adj_t)
    y_pred = out.argmax(dim=-1, keepdim=True)

    train_acc = evaluator.eval({
        'y_true': data.y[split_idx['train']].view(-1, 1),
        'y_pred': y_pred[split_idx['train']],
    })['acc']
    
    valid_acc = evaluator.eval({
        'y_true': data.y[split_idx['valid']].view(-1, 1),
        'y_pred': y_pred[split_idx['valid']],
    })['acc']
    
    test_acc = evaluator.eval({
        'y_true': data.y[split_idx['test']].view(-1, 1),
        'y_pred': y_pred[split_idx['test']],
    })['acc']

    return train_acc, valid_acc, test_acc

def save_experiment_log(args, results, log_file):
    entry = results.copy()
    entry['args'] = str(args)
    df = pd.DataFrame([entry])
    cols = [c for c in df.columns if c != 'args'] + ['args']
    df = df[cols]
    if not os.path.isfile(log_file):
        df.to_csv(log_file, index=False, header=True)
    else:
        df.to_csv(log_file, mode='a', index=False, header=False)

# ==========================================
# 4. DATA LOADING
# ==========================================

print("Loading dataset...")
dataset = NodePropPredDataset(name='ogbn-arxiv', root='dataset') 
graph, labels = dataset[0]
split_idx = dataset.get_idx_split()
evaluator = Evaluator(name='ogbn-arxiv')

x = torch.from_numpy(graph['node_feat']).to(torch.float)
y = torch.from_numpy(labels).to(torch.long).squeeze(1)
edge_index = torch.from_numpy(graph['edge_index']).to(torch.long)

data = Data(x=x, y=y, edge_index=edge_index)
data = T.ToUndirected()(data)
data = T.ToSparseTensor()(data)

data = data.to(DEVICE)
train_idx = torch.from_numpy(split_idx['train']).to(DEVICE)
valid_idx = torch.from_numpy(split_idx['valid']).to(DEVICE)
test_idx = torch.from_numpy(split_idx['test']).to(DEVICE)

input_dim = data.num_features
output_dim = dataset.num_classes

print(f"Dataset ready. Nodes: {data.num_nodes}, Classes: {output_dim}")

# ==========================================
# 5. GRID SEARCH LOOP
# ==========================================

keys, values = zip(*param_grid.items())
experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]

print(f"\n--- Starting Grid Search ({len(experiments)} configs) ---")
print(f"Results will be saved to: {OUTPUT_ROOT}/")

for i, args in enumerate(experiments):
    print(f"\n[{i+1}/{len(experiments)}] Config: {args}")
    set_seed(42)
    
    # 1. Initialize Model with new params (norm_type, use_residual)
    model = GCN(input_dim, 
                args['hidden_dim'], 
                output_dim, 
                args['num_layers'], 
                args['dropout'],
                norm_type=args['norm_type'],
                use_residual=args['use_residual'],
                drop_edge_p=args['drop_edge_p'] 
                ).to(DEVICE)
    
    # 2. Optimizer & Scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])
    criterion = torch.nn.NLLLoss()
    
    # Cosine Annealing Scheduler (NEW!)
    # Starts at args['lr'] and goes down to 0.001
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args['epochs'], eta_min=0.001)
    
    # Experiment Identifiers
    experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"gcn_arxiv_{experiment_id}.pth"
    save_path = os.path.join(MODELS_DIR, model_filename)
    
    history = {'train_loss': [], 'train_acc': [], 'valid_acc': [], 'test_acc': []}
    best_valid_acc = 0
    best_test_acc = 0
    patience_counter = 0
    final_epoch = 0
    
    start_time = time.time()
    
    # 3. Training Loop
    for epoch in range(1, args['epochs'] + 1):
        loss = train(model, data, train_idx, optimizer, criterion)
        train_acc, valid_acc, test_acc = test(model, data, split_idx, evaluator)
        
        # Step the scheduler
        scheduler.step()
        
        history['train_loss'].append(loss)
        history['train_acc'].append(train_acc)
        history['valid_acc'].append(valid_acc)
        history['test_acc'].append(test_acc)
        
        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            best_test_acc = test_acc
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1
            
        if patience_counter >= PATIENCE:
            print(f"   -> Early stopping at epoch {epoch}")
            final_epoch = epoch
            break
        
        final_epoch = epoch
            
    duration = time.time() - start_time
    
    num_test = len(test_idx)
    margin = 1.96 * np.sqrt((best_test_acc * (1 - best_test_acc)) / num_test)
    print(f"   -> Result: Val={best_valid_acc:.4f}, Test={best_test_acc:.4f} ± {margin:.4f}")
    
    # --- SAVE RESULTS ---
    history_filename = f"history_{experiment_id}.json"
    history_path = os.path.join(LOGS_DIR, history_filename)
    
    with open(history_path, 'w') as f:
        json.dump(history, f)
        
    results = {
        'experiment_id': experiment_id,
        'date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'model_file': save_path,
        'history_file': history_path,
        'duration_seconds': round(duration, 2),
        'actual_epochs': final_epoch,
        'best_valid_acc': best_valid_acc,
        'final_test_accuracy': best_test_acc,
        'test_margin_error': margin,
        'final_train_loss': loss
    }
    
    save_experiment_log(args, results, LOG_FILE)

print("\n--- Grid Search Complete ---")