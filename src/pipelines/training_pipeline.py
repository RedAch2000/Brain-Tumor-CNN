import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from src.components import (
    DataIngestionStep, 
    DataValidationStep, 
    DataTransformationModule, 
    BrainTumorModel
)
from pathlib import Path

def run_training_pipeline():
    #### STEP 01: Data Ingestion
    print("\n--- Starting Data Ingestion ---")
    ingestion = DataIngestionStep()
    raw_data_path = ingestion.run()

    #### STEP 02: Data Validation
    print("\n--- Starting Data Validation ---")
    validation = DataValidationStep()
    validation_results = validation.run()
    
    # We extract weights calculated in Validation (if you uncommented that logic)
    # For now, let's assume we pass them or calculate them here
    class_weights = validation_results.get("class_weights", None)

    #### STEP 03: Data Transformation (LightningDataModule)
    print("\n--- Starting Data Transformation ---")
    data_module = DataTransformationModule()
    # data_module.setup() is called automatically by the Trainer, 
    # but we can call it manually if we need to inspect data.

    #### STEP 04: Model Initialization
    print("\n--- Initializing Model ---")
    model = BrainTumorModel(lr=1e-4, class_weights=class_weights)

    # NEW: Define Checkpoint Behavior
    checkpoint_callback = ModelCheckpoint(
        dirpath=Path("models/checkpoints"),     # Where to save
        filename="brain-tumor-{epoch:02d}-{val_f1:.2f}", # Name format
        monitor="val_f1",                 # Metric to track
        mode="max",                       # We want the highest F1
        save_top_k=1,                     # Keep only the 1 best model
        save_last=True                    # ALSO save 'last.ckpt' automatically
    )

    # Optional: Add Early Stopping so you don't waste GPU time
    early_stop_callback = EarlyStopping(
        monitor="val_f1", 
        patience=5, 
        mode="max"
    )

    #### STEP 05: Training with GPU
    print("\n--- Starting Training ---")
    trainer = L.Trainer(
        accelerator="gpu", 
        devices=1, 
        max_epochs=30,
        precision="16-mixed",
        callbacks=[checkpoint_callback, early_stop_callback], # Add callbacks here
        default_root_dir="logs/" # Where logs and default outputs go
    )

    trainer.fit(model, datamodule=data_module)
    print("\nPipeline Execution Finished Successfully!")
    print("---------------------------------------")
    print(f"\nBest model saved at: {checkpoint_callback.best_model_path}")
    print(f"Last model saved at: {checkpoint_callback.last_model_path}")

if __name__ == "__main__":
    run_training_pipeline()