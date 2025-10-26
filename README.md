# 3D-Captioning-VLM

## 🧩 Project Architecture Diagram

Below is the module dependency and data flow for the **3D Captioning System**.  
Each box represents a Python module or class, and arrows indicate the direction of data or function calls.

```mermaid
flowchart TD

%% --- Data Pipeline ---
A1["data/data_loader.py<br/>Cap3DDataset / DataModule"] -->|point clouds + captions| B1["models/__init__.py<br/>CaptionModel"]

%% --- Model Submodules ---
subgraph MODELS["models/"]
    B2["encoders.py<br/>BaseEncoder / DGCNNEncoder / PointBERTEncoder"]
    B3["projection.py<br/>ProjectionLayer"]
    B4["decoders.py<br/>GPT2Decoder"]
end

%% --- CaptionModel Composition ---
B1 --> B2
B1 --> B3
B1 --> B4

%% --- Training Pipeline ---
subgraph TRAINING["training/"]
    C1["trainer.py<br/>Trainer"]
end
C1 --> B1
C1 --> D1

%% --- Evaluation ---
subgraph EVAL["evaluation/"]
    D1["metrics.py<br/>CaptionEvaluator"]
end

%% --- Entry Point ---
E1["main.ipynb / main.py<br/>load_config(), setup_environment(), main()"]

E1 --> A1
E1 --> C1
C1 -->|periodic validation| D1

%% --- Config ---
F1["configs/train_config.yaml<br/>Experiment parameters"]
E1 --> F1
C1 --> F1