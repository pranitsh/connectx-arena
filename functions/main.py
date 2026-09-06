import os
import time
import json
import random
import traceback
import concurrent.futures
from firebase_functions import https_fn, options
from firebase_admin import initialize_app, firestore

# MUST be in global scope for the @on_call decorator to validate tokens
try:
    initialize_app()
except ValueError:
    pass

ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets")

AGENT_FILES = {
    "mlp-small": "mlp_default_params_final.zip",
    "mlp-medium": "mlp_mid_params_final.zip",
    "mlp-large": "mlp_large_params_final.zip",
    "cnn-small": "cnn_16_filters_final.zip",
    "cnn-large": "cnn_32_filters_final.zip",
}

loaded_models = {}
_db = None

def get_db():
    global _db
    if _db is None:
        _db = firestore.client()
    return _db

def get_model(agent_name):
    from stable_baselines3 import DQN
    if agent_name not in loaded_models:
        model_path = os.path.join(ASSETS_DIR, AGENT_FILES[agent_name])
        loaded_models[agent_name] = DQN.load(model_path, device="cpu")
    return loaded_models[agent_name]

def get_valid_columns(board_state):
    return [c for c in range(7) if board_state[0][c] == 0]

def predict_move(model, obs):
    action, _ = model.predict(obs, deterministic=True)
    return int(action)

def validate_payload(board_state, agent_name):
    if agent_name not in AGENT_FILES:
        raise ValueError(f"Illegal agent name: {agent_name}")
    
    if not isinstance(board_state, list) or len(board_state) != 6:
        raise ValueError("Board must be a 2D array of exactly 6 rows.")
        
    for row in board_state:
        if not isinstance(row, list) or len(row) != 7:
            raise ValueError("Each row must contain exactly 7 columns.")
        for cell in row:
            if cell not in (0, 1, 2, 0.0, 1.0, 2.0):
                raise ValueError(f"Invalid cell value detected: {cell}")

@https_fn.on_call(
    cors=options.CorsOptions(cors_origins="*", cors_methods=["POST"]),
    enforce_app_check=True 
)
def get_ai_move(req: https_fn.Request) -> dict:
    import numpy as np
    
    try:
        if req.auth is None:
            raise ValueError("User must be authenticated.")
        
        uid = req.auth.uid
        app_id = getattr(req.app, 'app_id', 'unverified_local_app') if req.app else 'unverified_local_app'
        
        data = req.data
        board_state = data.get("board")
        agent_name = data.get("agent", "mlp-small")

        # 1. Strict Payload Defense
        validate_payload(board_state, agent_name)

        db = get_db()
        rate_limit_ref = db.collection("rate_limits").document(uid)
        doc = rate_limit_ref.get()
        current_time = time.time()
        
        if doc.exists:
            last_request = doc.to_dict().get("last_request", 0)
            if current_time - last_request < 0.5:
                raise ValueError("Rate limit exceeded.")
        
        rate_limit_ref.set({"last_request": current_time})

        # 3. Model Inference with Timeout Protection
        model = get_model(agent_name)
        valid_cols = get_valid_columns(board_state)
        
        if not valid_cols:
            raise ValueError("No valid moves available on board.")

        obs = np.array(board_state, dtype=np.float32).reshape(1, 6, 7)
        action = None
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(predict_move, model, obs)
            try:
                action = future.result(timeout=3.0)
            except concurrent.futures.TimeoutError:
                print("Prediction timed out. Falling back to random.")
                action = random.choice(valid_cols)
        
        if action not in valid_cols:
            action = random.choice(valid_cols)
            
        # 4. Save Game Move to Firestore (Serialize array to JSON string)
        db.collection("games").add({
            "uid": uid,
            "app_check_id": app_id,
            "agent": agent_name,
            "board_state": json.dumps(board_state),
            "ai_action": action,
            "timestamp": firestore.SERVER_TIMESTAMP
        })
        
        return {"column": action}
    
    except Exception as e:
        print(f"\n--- CRITICAL ERROR IN GET_AI_MOVE ---")
        traceback.print_exc() 
        print("---------------------------------------\n")
        
        raise https_fn.HttpsError(
            code=https_fn.FunctionsErrorCode.INTERNAL,
            message=f"Server Error: {str(e)}"
        )