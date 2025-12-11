import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import torch_geometric.transforms as T
from ogb.nodeproppred import NodePropPredDataset, Evaluator
import pandas as pd
import numpy as np
import os
import itertools
import time
from datetime import datetime
import random
import json

# ==========================================
# 1. CONFIGURATION & DIRECTORIES
# ==========================================

# Define output structure
OUTPUT_ROOT = 'outputs'
MODELS_DIR = os.path.join(OUTPUT_ROOT, 'models')
LOGS_DIR = os.path.join(OUTPUT_ROOT, 'logs')
LOG_FILE = os.path.join(OUTPUT_ROOT, 'experiments_log.csv')

# Ensure directories exist
for folder in [OUTPUT_ROOT, MODELS_DIR, LOGS_DIR]:
    if not os.path.exists(folder):
        os.makedirs(folder)
        print(f"Created directory: {folder}")

# Hyperparameter Grid
param_grid = {
    # Architecture
    'num_layers': [2, 3, 4],             # GCNs work best with few layers (2 or 3)
    'hidden_dim': [128, 256, 512, 1024], # 128 is baseline, 256/512 increase capacity
    
    # Optimization
    'lr': [0.01, 0.005],           # 0.01 is standard, 0.005 for refining
    
    # Regularization (CRITICAL for ogbn-arxiv)
    'dropout': [0.3, 0.5],         # 0.5 prevents memorizing old data
    'weight_decay': [0, 5e-4],     # 5e-4 is the "magic" value from the original GCN paper
    
    # Fixed (do not vary in grid, keep it fixed in the loop)
    'epochs': [500]                # Early stopping will cut it short, set high ceiling
}

# Fixed settings
PATIENCE = 50
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ==========================================
# 2. MODEL DEFINITION (GCN)
# ==========================================

class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout):
        super(GCN, self).__init__()
        
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        self.dropout = dropout

        # Input Layer
        self.convs.append(GCNConv(input_dim, hidden_dim))
        self.bns.append(torch.nn.BatchNorm1d(hidden_dim))

        # Hidden Layers
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.bns.append(torch.nn.BatchNorm1d(hidden_dim))

        # Output Layer
        self.convs.append(GCNConv(hidden_dim, output_dim))

    def forward(self, x, adj_t):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            
        x = self.convs[-1](x, adj_t)
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
    """Saves results with 'args' grouped in a single column."""
    entry = results.copy()
    entry['args'] = str(args) # Convert dict to string
    
    df = pd.DataFrame([entry])
    
    # Reorder columns
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
# Note: Check if you need '../dataset' or just 'dataset'
dataset = NodePropPredDataset(name='ogbn-arxiv', root='dataset') 
graph, labels = dataset[0]
split_idx = dataset.get_idx_split()
evaluator = Evaluator(name='ogbn-arxiv')

x = torch.from_numpy(graph['node_feat']).to(torch.float)
y = torch.from_numpy(labels).to(torch.long).squeeze(1)
edge_index = torch.from_numpy(graph['edge_index']).to(torch.long)

data = T.ToSparseTensor()(T.ToUndirected()(
    T.Data(x=x, y=y, edge_index=edge_index)
))

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
    
    # Initialize
    model = GCN(input_dim, args['hidden_dim'], output_dim, args['num_layers'], args['dropout']).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])
    criterion = torch.nn.NLLLoss()
    
    # Experiment ID
    experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"gcn_arxiv_{experiment_id}.pth"
    save_path = os.path.join(MODELS_DIR, model_filename)
    
    # History tracking
    history = {'train_loss': [], 'train_acc': [], 'valid_acc': [], 'test_acc': []}
    
    best_valid_acc = 0
    best_test_acc = 0
    patience_counter = 0
    final_epoch = 0
    
    start_time = time.time()
    
    # Training Loop
    for epoch in range(1, args['epochs'] + 1):
        loss = train(model, data, train_idx, optimizer, criterion)
        train_acc, valid_acc, test_acc = test(model, data, split_idx, evaluator)
        
        # Collect History
        history['train_loss'].append(loss)
        history['train_acc'].append(train_acc)
        history['valid_acc'].append(valid_acc)
        history['test_acc'].append(test_acc)
        
        # Early Stopping
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
    
    # Metrics
    num_test = len(test_idx)
    margin = 1.96 * np.sqrt((best_test_acc * (1 - best_test_acc)) / num_test)
    print(f"   -> Result: Val={best_valid_acc:.4f}, Test={best_test_acc:.4f} ± {margin:.4f}")
    
    # --- SAVE RESULTS ---
    
    # 1. Save Detailed History (JSON)
    history_filename = f"history_{experiment_id}.json"
    history_path = os.path.join(LOGS_DIR, history_filename)
    
    with open(history_path, 'w') as f:
        json.dump(history, f)
        
    # 2. Save Summary (CSV)
    results = {
        'experiment_id': experiment_id,
        'date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'model_file': save_path,
        'history_file': history_path, # Link to JSON
        'duration_seconds': round(duration, 2),
        'actual_epochs': final_epoch,
        'best_valid_acc': best_valid_acc,
        'final_test_accuracy': best_test_acc,
        'test_margin_error': margin,
        'final_train_loss': loss
    }
    
    save_experiment_log(args, results, LOG_FILE)

print("\n--- Grid Search Complete ---")