
"""
EXPERIMENT 08: Graph-Guided Invariance (Topological Sort Reordering)
Method: Data Dependency Graph (DDG) Analysis via AST.
Goal: Distinguish "Graph Understanding" from "Sequence Memorization".
      - Extract independent statements from code blocks.
      - Reorder them (valid topological sort permutation).
      - If Model Loss stays low -> It understands the Graph (Semantics).
      - If Model Loss spikes -> It memorized the specific Sequence.
Reference: "Graph-based Structural Auditing" (Proposed 2026).
Usage: Copy-paste this entire file into a Kaggle cell.
"""
import ast
import random
import os
import sys
import copy
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Set, Tuple
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset, load_from_disk
from sklearn.metrics import roc_auc_score

# ============================================================================
# Kaggle & Environment Setup
# ============================================================================

def setup_environment():
    print("--- Environment Setup Starting ---")
    try:
        import transformers
        import datasets
    except ImportError:
        print("Installing dependencies...")
        os.system("pip install -q transformers datasets accelerate scikit-learn pandas numpy huggingface_hub")
    
    try:
        from kaggle_secrets import UserSecretsClient
        user_secrets = UserSecretsClient()
        hf_token = user_secrets.get_secret("posioned")
        from huggingface_hub import login
        login(token=hf_token)
        print("Logged in to Hugging Face.")
    except Exception as e:
        print(f"Login Note: {e}")

    kaggle_path = "/kaggle/input/datasets/minh2duy/poisoned-chalice-dataset"
    if os.path.exists(kaggle_path):
        print(f"Dataset found: {kaggle_path}")
    else:
        print("Warning: Standard dataset path not found.")
    print("--- Environment Setup Complete ---")

# ============================================================================
# Graph Analysis: Data Dependency & Reordering
# ============================================================================

class DependencyAnalyzer(ast.NodeVisitor):
    """
    Analyzes reads/writes in a statement to determine dependencies.
    Simplified: Tracks variable names used in Lookups vs Stores.
    """
    def __init__(self):
        self.reads = set()
        self.writes = set()

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Load):
            self.reads.add(node.id)
        elif isinstance(node.ctx, ast.Store):
            self.writes.add(node.id)
        # Del?
    
    def visit_arg(self, node):
        self.writes.add(node.arg)

def get_dependencies(node: ast.AST) -> Tuple[Set[str], Set[str]]:
    analyzer = DependencyAnalyzer()
    analyzer.visit(node)
    return analyzer.reads, analyzer.writes

def is_independent(node1, node2) -> bool:
    """
    Checks if node2 depends on node1 or vice versa.
    Simplified: 
    - Dependent if:
      - node2 reads what node1 writes (RAW)
      - node2 writes what node1 reads (WAR)
      - node2 writes what node1 writes (WAW)
    - Independent if disjoint read/write sets overlap is safe.
    """
    r1, w1 = get_dependencies(node1)
    r2, w2 = get_dependencies(node2)
    
    # Check conflicts
    # RAW: 1 writes X, 2 reads X
    if not w1.isdisjoint(r2): return False
    # WAR: 1 reads X, 2 writes X
    if not r1.isdisjoint(w2): return False
    # WAW: 1 writes X, 2 writes X
    if not w1.isdisjoint(w2): return False
    
    # Also side effects? (Print, file IO). 
    # For now assume pure computation or assume side effects are dependencies.
    # To be safe: if function call, assume it touches everything? 
    # Let's be aggressive for MIA test: assume simple dependency logic.
    return True

class StatementPermuter(ast.NodeTransformer):
    """
    Visits blocks (Module body, FunctionDef body, For body, etc.) 
    and tries to swap adjacent independent statements.
    """
    def __init__(self):
        self.rng = random.Random(42)

    def visit_block(self, nodes: List[ast.AST]) -> List[ast.AST]:
        if len(nodes) < 2: return nodes
        
        # Simple Bubble Swap approach for valid permutation
        # Try to swap node[i] and node[i+1] if independent
        new_nodes = list(nodes)
        swaps_made = 0
        
        # Try N random swaps
        for _ in range(len(nodes)): # limit attempts
            idx = self.rng.randint(0, len(new_nodes) - 2)
            stmt1 = new_nodes[idx]
            stmt2 = new_nodes[idx+1]
            
            # Simple check types: Only swap Expr, Assign, AnnAssign to stay safe
            # Don't swap Control Flow (If/For) for now unless we analyze deeply
            if not isinstance(stmt1, (ast.Assign, ast.Expr, ast.AugAssign, ast.AnnAssign)): continue
            if not isinstance(stmt2, (ast.Assign, ast.Expr, ast.AugAssign, ast.AnnAssign)): continue
            
            if is_independent(stmt1, stmt2):
                # Valid swap!
                new_nodes[idx], new_nodes[idx+1] = new_nodes[idx+1], new_nodes[idx]
                swaps_made += 1
                
        return new_nodes

    def visit_Module(self, node):
        node.body = self.visit_block(node.body)
        self.generic_visit(node)
        return node
    
    def visit_FunctionDef(self, node):
        node.body = self.visit_block(node.body)
        self.generic_visit(node)
        return node

    def visit_For(self, node):
        node.body = self.visit_block(node.body)
        self.generic_visit(node)
        return node
    
    def visit_While(self, node):
        node.body = self.visit_block(node.body)
        self.generic_visit(node)
        return node
    
    def visit_If(self, node):
        node.body = self.visit_block(node.body)
        node.orelse = self.visit_block(node.orelse)
        self.generic_visit(node)
        return node

def generate_graph_perturbation(code: str) -> Optional[str]:
    try:
        tree = ast.parse(code)
        permuter = StatementPermuter()
        transformed = permuter.visit(tree)
        if sys.version_info >= (3, 9):
            return ast.unparse(transformed)
    except:
        return None
    return None

# ============================================================================
# Model Loading
# ============================================================================

def load_model(model_path):
    print(f"Loading model: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        trust_remote_code=True, 
        torch_dtype=dtype, 
        device_map="auto"
    )
    model.eval()
    return model, tokenizer

# ============================================================================
# Graph Invariance Attack
# ============================================================================

class GraphInvarianceAttack:
    def __init__(self, args, model, tokenizer):
        self.args = args
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = args.max_length

    @property
    def name(self):
        return "graph_invariance"

    def calculate_loss(self, text: str) -> float:
        if not text or len(text.strip()) == 0: return np.nan
        try:
            inputs = self.tokenizer(text, return_tensors="pt", max_length=self.max_length, truncation=True).to(self.model.device)
            with torch.no_grad():
                outputs = self.model(**inputs, labels=inputs["input_ids"])
            return outputs.loss.item()
        except:
            return np.nan

    def compute_scores(self, texts: List[str]) -> List[float]:
        print(f"Computing {self.name} scores...")
        scores = []
        
        for text in tqdm(texts, desc="Graph Analysis"):
            # 1. Original Loss
            orig_loss = self.calculate_loss(text)
            if np.isnan(orig_loss):
                scores.append(np.nan)
                continue
                
            # 2. Perturbed (Reordered) Code
            try:
                pert_code = generate_graph_perturbation(text)
            except:
                pert_code = None
                
            if not pert_code or pert_code == text:
                # No valid permutation found or parse error
                # Neutral score or NaN?
                # If we assume "robust", score should be 0 (diff)
                scores.append(0.0)
                continue
            
            # 3. Perturbed Loss
            pert_loss = self.calculate_loss(pert_code)
            
            if np.isnan(pert_loss):
                scores.append(np.nan)
                continue
                
            # 4. Invariance Score = Loss(Reordered) - Loss(Original)
            # High Score (Positive) -> Loss Spiked -> Sequence Memorization
            # Low Score (Near 0) -> Loss Stable -> Graph Understanding
            score = pert_loss - orig_loss
            scores.append(score)
            
        return scores

# ============================================================================
# Experiment Runners
# ============================================================================

class Experiment:
    def __init__(self, args):
        self.args = args
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        random.seed(args.seed)
        self.model, self.tokenizer = load_model(args.model_name)

    def load_data(self):
        subsets = ['Go', 'Java', 'Python', 'Ruby', 'Rust']
        dfs = []
        is_local = os.path.exists(self.args.dataset)
        print(f"Loading data from {self.args.dataset}...")
        for subset in subsets:
            if is_local:
                path = os.path.join(self.args.dataset, subset)
                if not os.path.exists(path): continue
                ds = load_from_disk(path)
                if hasattr(ds, "keys") and "test" in ds.keys(): ds = ds["test"]
            else:
                ds = load_dataset(self.args.dataset, subset, split="test")
            dfs.append(ds.to_pandas())
        if not dfs: raise ValueError("No data found!")
        df = pd.concat(dfs, ignore_index=True)
        df['is_member'] = df['membership'].apply(lambda x: 1 if x == 'member' else 0)
        
        if self.args.sample_fraction < 1.0:
            df = df.sample(frac=self.args.sample_fraction, random_state=self.args.seed)
            print(f"Sampled {len(df)} rows.")
        return df

    def run(self):
        df = self.load_data()
        attacker = GraphInvarianceAttack(self.args, self.model, self.tokenizer)
        scores = attacker.compute_scores(df['content'].tolist())
        df[f"{attacker.name}_score"] = scores
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"EXP08_{self.args.model_name.replace('/', '_')}_{timestamp}.parquet"
        df.to_parquet(self.output_dir / fname, index=False)
        print(f"Saved to {fname}")
        
        try:
            auc = roc_auc_score(df['is_member'], df[f"{attacker.name}_score"].fillna(-999))
            print(f"AUC ({attacker.name}): {auc:.4f}")
        except Exception as e:
            print(f"AUC Error: {e}")

if __name__ == "__main__":
    setup_environment()
    
    class Args:
        model_name = "bigcode/starcoder2-3b"
        if os.path.exists("/kaggle/input/datasets/minh2duy/poisoned-chalice-dataset"):
            dataset = "/kaggle/input/datasets/minh2duy/poisoned-chalice-dataset"
        else:
            dataset = "AISE-TUDelft/Poisoned-Chalice"
        sample_fraction = 0.1
        output_dir = "results"
        max_length = 2048
        seed = 42

    print(f"[EXP08] Model: {Args.model_name}")
    Experiment(Args).run()
