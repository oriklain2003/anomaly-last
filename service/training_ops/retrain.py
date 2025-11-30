import sys
# Fix for DLL load error on Windows by importing torch first
try:
    import torch
except ImportError:
    pass

import logging
import sqlite3
from pathlib import Path
import shutil

# Add root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training_ops.db_utils import TRAINING_DB_PATH, init_dbs

# Import training modules directly
from mlboost.train import run_training as train_xgboost
from ml_deep_cnn.train import run_training as train_cnn
from ml_deep.train import run_training as train_deep_dense
from ml_transformer.train import run_training as train_transformer

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DB_PATH = Path("last.db")

def merge_datasets():
    """
    Merges the base dataset (last.db) and the feedback dataset (training.db)
    into a single consolidated database for training.
    """
    logger.info("Merging datasets...")
    init_dbs() # Ensure training DB exists
    
    # We will copy the Training DB (which contains feedback) to a new 'consolidated.db'
    # and then attach 'last.db' and insert records from it that ARE NOT in the training DB.
    
    CONSOLIDATED_DB_PATH = Path("training_ops/consolidated.db")
    
    if CONSOLIDATED_DB_PATH.exists():
        CONSOLIDATED_DB_PATH.unlink()
        
    # Start with a copy of the feedback/training DB
    if TRAINING_DB_PATH.exists():
        shutil.copy(TRAINING_DB_PATH, CONSOLIDATED_DB_PATH)
    else:
        init_dbs() 
        shutil.copy(TRAINING_DB_PATH, CONSOLIDATED_DB_PATH)

    conn = sqlite3.connect(str(CONSOLIDATED_DB_PATH))
    cursor = conn.cursor()
    
    # Attach Base DB
    if BASE_DB_PATH.exists():
        logger.info(f"Attaching base DB: {BASE_DB_PATH}")
        cursor.execute("ATTACH DATABASE ? AS base", (str(BASE_DB_PATH),))
        
        # Merge Normal Flights
        logger.info("Merging Normal Flights...")
        cursor.execute("""
            INSERT INTO main.flight_tracks 
            SELECT * FROM base.flight_tracks 
            WHERE flight_id NOT IN (SELECT flight_id FROM main.flight_tracks)
        """)
        
        # Merge Anomalies
        try:
            cursor.execute("SELECT count(*) FROM base.anomalous_tracks")
            has_anom = True
        except sqlite3.OperationalError:
            has_anom = False
            
        if has_anom:
            logger.info("Merging Anomalous Flights...")
            cursor.execute("""
                INSERT INTO main.anomalous_tracks 
                SELECT * FROM base.anomalous_tracks 
                WHERE flight_id NOT IN (SELECT flight_id FROM main.anomalous_tracks)
            """)
            
        conn.commit()
        conn.close()
        logger.info("Merge complete.")
    else:
        logger.warning("Base DB not found. Training only on feedback data.")
        
    return CONSOLIDATED_DB_PATH

def main():
    print("=== Retraining Pipeline Started ===")
    
    # 1. Prepare Data
    db_path = merge_datasets()
    
    # 2. Train Models
    print("\n--- Training XGBoost ---")
    try:
        train_xgboost(db_path, Path("mlboost/output/xgb_model.joblib"))
        logger.info("XGBoost training successful")
    except Exception as e:
        logger.error(f"XGBoost training failed: {e}")

    print("\n--- Training Deep CNN ---")
    try:
        train_cnn(db_path, Path("ml_deep_cnn/output"), epochs=50)
        logger.info("CNN training successful")
    except Exception as e:
        logger.error(f"CNN training failed: {e}")

    print("\n--- Training Deep Dense (Autoencoder) ---")
    try:
        train_deep_dense(db_path, Path("ml_deep/output"), epochs=50)
        logger.info("Deep Dense training successful")
    except Exception as e:
        logger.error(f"Deep Dense training failed: {e}")

    print("\n--- Training Transformer ---")
    try:
        train_transformer(db_path, Path("ml_transformer/output"), epochs=50)
        logger.info("Transformer training successful")
    except Exception as e:
        logger.error(f"Transformer training failed: {e}")
    
    print("\n=== Retraining Complete ===")
    print("New models are ready in their respective output directories.")

if __name__ == "__main__":
    main()
