# ============================================================
# ml_to_rag_bridge.py
# Connects Part 1 (ML pipeline) with Part 2 (RAG over TM 5-692-1)
#
# PLACEMENT: inside PetroMind-RAG-book\ (next to config.py,
#            database.py, main.py, parser.py, retriever.py)
#
# Flow:
#   1. Load trained LSTM model from ncmpass-pipeline/checkpoints_ncmapss_rul/
#   2. Run inference on a sensor window -> RUL + risk level
#   3. Translate prediction into a natural-language query
#   4. Feed query into Part 2 retriever -> manual procedure chunks
#   5. Combine prediction + procedure into one LLM-grounded answer
# ============================================================

import os
import sys
import numpy as np
import torch

# ── Path setup ───────────────────────────────────────────────
# This file lives in:  main-main (1)/main-main/PetroMind-RAG-book/
# base-pipeline lives: main-main (1)/main-main/base-pipeline/pipeline/
THIS_DIR       = os.path.dirname(os.path.abspath(__file__))
MAIN_MAIN_DIR  = os.path.dirname(THIS_DIR)                          # one level up
BASE_PIPELINE  = os.path.join(MAIN_MAIN_DIR, "base-pipeline")       # pipeline package lives here

if BASE_PIPELINE not in sys.path:
    sys.path.insert(0, BASE_PIPELINE)

# ── Part 1 imports ───────────────────────────────────────────
try:
    from pipeline.config    import PipelineConfig
    from pipeline.rul_model import LSTMRULModel
    print("[bridge] Imported pipeline models OK")
except ImportError as e:
    print(f"[bridge] FATAL: cannot import pipeline: {e}")
    print(f"[bridge] Checked path: {BASE_PIPELINE}")
    sys.exit(1)

# ── Part 2 imports (same folder as this file) ────────────────
from database  import PineconeRAGDatabase
from retriever import PineconeHybridRetriever
from main      import build_prompt, call_llm
from config    import RAGConfig


# ============================================================
# Step 1 — Load the trained ML model
# ============================================================

def load_rul_model(checkpoint_path: str, device: str = "cpu") -> dict:
    """
    Load trained LSTMRULModel from a train_ncmapss.py checkpoint.

    The checkpoint stores everything needed:
        sensor_cols  — list of 41 N-CMAPSS feature names
        mean / std   — normalization params saved during training
        config       — dict of training args (hidden_dim, num_layers, dropout)
        model_state_dict — LSTM weights

    Returns
    -------
    dict: {model, sensor_cols, mean, std}
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    sensor_cols = checkpoint["sensor_cols"]          # 41 column names
    mean        = checkpoint["mean"]                  # torch tensor shape (1,1,41)
    std         = checkpoint["std"]                   # torch tensor shape (1,1,41)
    train_cfg   = checkpoint["config"]                # dict of argparse args

    input_dim = len(sensor_cols)

    cfg = PipelineConfig(
        hidden_dim    = train_cfg["hidden_dim"],
        n_lstm_layers = train_cfg["num_layers"],
        dropout       = train_cfg["dropout"],
    )

    model = LSTMRULModel(input_dim, cfg).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    mean_np = mean.numpy() if hasattr(mean, "numpy") else np.array(mean)
    std_np  = std.numpy()  if hasattr(std,  "numpy") else np.array(std)

    print(f"[bridge] Loaded RUL model  input_dim={input_dim}  "
          f"hidden={cfg.hidden_dim}  layers={cfg.n_lstm_layers}  "
          f"best_rmse={checkpoint.get('rmse', 'N/A')}")

    return {
        "model":       model,
        "sensor_cols": sensor_cols,
        "mean":        mean_np,
        "std":         std_np,
    }


# ============================================================
# Step 2 — Run inference on one sensor window
# ============================================================

def predict_rul(model_bundle: dict, X_window: np.ndarray, device: str = "cpu") -> dict:
    """
    Run RUL inference on ONE raw sensor window.

    Parameters
    ----------
    model_bundle : output of load_rul_model()
    X_window     : np.ndarray shape (window_size, n_features)
                   RAW un-normalized values in sensor_cols order

    Returns
    -------
    dict: {rul, risk_level}
    """
    model = model_bundle["model"]
    mean  = model_bundle["mean"]
    std   = model_bundle["std"]

    # Normalize exactly as done during training
    X_norm = (X_window - mean.squeeze()) / std.squeeze()

    x = torch.as_tensor(X_norm, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        rul_pred = float(model(x).cpu().numpy()[0])

    if rul_pred < 20:
        risk = "HIGH"
    elif rul_pred < 60:
        risk = "MEDIUM"
    else:
        risk = "LOW"

    return {"rul": rul_pred, "risk_level": risk}


# ============================================================
# Step 3 — Translate ML prediction -> natural language query
# ============================================================

def build_maintenance_query(prediction: dict, equipment_type: str = "turbine engine") -> str:
    """
    THE BRIDGE FUNCTION.
    Converts a numeric RUL prediction into a sentence the RAG
    retriever can search against in TM 5-692-1.

    equipment_type should match what the ML model was actually trained on.
    For N-CMAPSS data this is a turbofan/gas-turbine engine (HPT/LPT
    degradation), so use "turbine engine" or "gas turbine" — not "bearing".
    """
    rul  = prediction["rul"]
    risk = prediction["risk_level"]

    if risk == "HIGH":
        return (
            f"The {equipment_type} compressor/turbine section is showing signs of "
            f"imminent failure with an estimated remaining useful life of {rul:.0f} "
            f"cycles. What inspection, overhaul, and replacement procedure should "
            f"be followed before it fails?"
        )
    elif risk == "MEDIUM":
        return (
            f"The {equipment_type} compressor/turbine section shows early "
            f"performance degradation with an estimated {rul:.0f} cycles "
            f"remaining. What maintenance action and monitoring schedule is "
            f"recommended at this stage?"
        )
    else:
        return (
            f"What is the routine preventive maintenance schedule for a "
            f"{equipment_type} under normal operating conditions?"
        )


# ============================================================
# Step 4+5 — Full end-to-end pipeline
# ============================================================

def run_full_pipeline(
    model_bundle: dict,
    X_window: np.ndarray,
    db: PineconeRAGDatabase,
    retriever: PineconeHybridRetriever,
    equipment_type: str = "turbine engine",
    device: str = "cpu",
) -> dict:
    """
    Single call that runs the complete pipeline:
    sensor window -> RUL prediction -> RAG query ->
    manual chunks -> LLM answer
    """
    # Step 1: ML prediction
    prediction = predict_rul(model_bundle, X_window, device=device)
    print(f"[ML]     RUL={prediction['rul']:.1f}  Risk={prediction['risk_level']}")

    # Step 2: Bridge query
    query = build_maintenance_query(prediction, equipment_type)
    print(f"[Bridge] Query: {query}")

    # Step 3: RAG retrieval (Part 2 unchanged)
    chunks = retriever.retrieve(query)
    print(f"[RAG]    Retrieved {len(chunks)} chunk(s)")
    for c in chunks:
        m = c["metadata"]
        print(f"           [{c['id']}] {m.get('label','N/A')} "
              f"pp.{m.get('page_start','?')}-{m.get('page_end','?')}")

    # Step 4: Build prompt injecting ML prediction context
    base_prompt = build_prompt(query, chunks)
    full_prompt = (
        f"ML MODEL PREDICTION (from live sensor data):\n"
        f"  Estimated RUL : {prediction['rul']:.1f} cycles\n"
        f"  Risk Level    : {prediction['risk_level']}\n"
        f"  Equipment     : {equipment_type}\n\n"
        f"{base_prompt}"
    )

    # Step 5: LLM generates grounded answer
    answer = call_llm(full_prompt)

    return {
        "prediction": prediction,
        "query":      query,
        "chunks":     chunks,
        "answer":     answer,
    }


# ============================================================
# Main — run the full pipeline
# ============================================================

if __name__ == "__main__":
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # ── 1. Checkpoint path (relative to this file) ───────────
    CHECKPOINT = os.path.join(
        MAIN_MAIN_DIR,
        "ncmpass-pipeline",
        "checkpoints_ncmapss_rul",
        "best_model.pt",
    )

    # ── 2. Set up Part 2 RAG ─────────────────────────────────
    from parser import parse_document

    if not os.path.exists(RAGConfig.PDF_PATH):
        print(f"Error: '{RAGConfig.PDF_PATH}' not found.")
        sys.exit(1)

    print("Step 1: Parsing TM 5-692-1...")
    chunks_doc = parse_document(RAGConfig.PDF_PATH)

    print("Step 2: Connecting to Pinecone...")
    db    = PineconeRAGDatabase()
    stats = db.index.describe_index_stats()

    if stats.get("total_vector_count", 0) == len(chunks_doc):
        print(f"  Index already has {len(chunks_doc)} vectors — skipping upload.")
        for c in chunks_doc:
            db.chunk_store[c["id"]] = c
    else:
        print(f"  Uploading {len(chunks_doc)} chunks...")
        db.upload_documents(chunks_doc)

    print("Step 3: Initialising retriever...")
    retriever = PineconeHybridRetriever(db)

    # ── 3. Load Part 1 trained model ─────────────────────────
    print("Step 4: Loading RUL model...")
    model_bundle = load_rul_model(CHECKPOINT, device=DEVICE)

    # ── 4. Build a real sensor window from test data ─────────
    # Using N-CMAPSS test split for a real window
    try:
        sys.path.insert(0, os.path.join(MAIN_MAIN_DIR, "ncmpass-pipeline"))
        from ncmapss_loader import load_ncmapss_smart

        DATA_DIR = r"E:\petromind\data"
        print("Step 5: Loading real test window from N-CMAPSS...")
        df_test = load_ncmapss_smart(
            data_dir=DATA_DIR,
            split="test",
            sample_every=1,
            verbose=False,
        )

        # Get one engine's last 30 cycles as a real window
        sensor_cols  = model_bundle["sensor_cols"]
        unit_id      = df_test["unit_id"].iloc[0]
        unit_df      = df_test[df_test["unit_id"] == unit_id].sort_values("cycle")
        window_data  = unit_df[sensor_cols].values[-30:]   # last 30 cycles

        if len(window_data) < 30:
            raise ValueError(f"Not enough cycles: {len(window_data)}")

        true_rul = unit_df["rul"].values[-1]
        print(f"  Using unit={unit_id}  true_rul={true_rul:.1f}  "
              f"window_shape={window_data.shape}")

    except Exception as e:
        print(f"[WARN] Could not load real data ({e}) — using random window")
        window_size  = 30
        n_features   = len(model_bundle["sensor_cols"])
        window_data  = np.random.randn(window_size, n_features).astype(np.float32)

    # ── 5. Run the full connected pipeline ───────────────────
    print("\nStep 6: Running full pipeline...")
    result = run_full_pipeline(
        model_bundle   = model_bundle,
        X_window       = window_data,
        db             = db,
        retriever      = retriever,
        equipment_type = "turbine engine",
        device         = DEVICE,
    )

    print("\n" + "=" * 60)
    print("FINAL ANSWER")
    print("=" * 60)
    print(result["answer"])
    print("=" * 60)
    print(f"\nPrediction: RUL={result['prediction']['rul']:.1f}  "
          f"Risk={result['prediction']['risk_level']}")