# Stress Classification using TensorFlow LSTM 

> **Repository:** ScenerYOne/Stress_Classification_TensorflowLSTM
> **Phase:** Modeling & Evaluation
> **Preceding Phase:** [ScenerYOne/Preprocess_Datamodel](https://github.com/ScenerYOne/Preprocess_Datamodel)

##  Project Context

This project represents the **Modeling Phase** of the Stress Classification pipeline. It is designed to ingest preprocessed, normalized, and structured data outputs from the [Preprocess_Datamodel](https://github.com/ScenerYOne/Preprocess_Datamodel) repository to train a Deep Learning model.

**Goal:** To detect and classify stress levels from sequential data using **Long Short-Term Memory (LSTM)** networks, leveraging their ability to learn long-term dependencies in time-series or textual sequences.

**Key Enhanceement:** This implementation goes beyond static model training by incorporating **Automated Hyperparameter Tuning (Keras Tuner)** and **K-Fold Cross-Validation** to ensure model robustness and generalization.

##  Pipeline Overview

1.  **Data Ingestion:** Load processed datasets (typically, `.npy`, `.csv`) with shape ( `Sample`, `Time_Steps`,`Features`)
2.  **Hyperparameter Search:** Utilizes Keras Tuner to dynamically find the optimal architecture:
   - Units: Searches between 32 to 256 units.
   - Units: Layers: Dynamically adjusts between 1 to 5 LSTM layers.
   - Dropout: Optimizes dropout rates (0.05 - 0.2) to prevent overfitting.
   - Optimizer: Selects between Adam and SGD with varying learning rates.
4.  **Robust Evaluation (5-Fold CV):** Every hyperparameter trial is evaluated using 5-Fold Cross-Validation. The reported accuracy is the mean of these 5 folds, ensuring the model performs well across different data splits.
5.  **Logging & Visualization:** Automatically saves confusion matrices, model files (.keras/.h5), and detailed metrics (JSON) for every trial.

##  Model Architecture

The architecture is built to handle sequential inputs (Time-Series or Tokenized Text):

```python
# Conceptual Architecture
Model: "Sequential_LSTM"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 Input_Layer                 (Batch, Time_Steps, Feat) 0         
                                                                 
 LSTM_Layer_1 (Bidirectional)(Batch, Time_Steps, 128)  [Param]   
                                                                 
 Dropout                     (Batch, Time_Steps, 128)  0         
                                                                 
 LSTM_Layer_2                (Batch, 64)               [Param]   
                                                                 
 Dense_Layer                 (Batch, 32)               [Param]   
                                                                 
 Output_Layer (Softmax)      (Batch, Num_Classes)      [Param]   
=================================================================
```
## Tech Stack
  - Framework: TensorFlow 2.x, Keras
  - Optimization: Keras Tuner (`kt.HyperModel`)
  - Evaluation: Scikit-Learn (`KFold`, `classification_report`, `confusion_matrix`)
  - Data Handling: NumPy, Pandas
  - *Visualization: Matplotlib, Seaborn (for Confusion Matrix & Loss Curves , Heatmaps)

## Repository Structure

```python
ScenerYOne/Stress_Classification_TensorflowLSTM/
├── models/               # Saved trained models (.h5 / .keras)
├── data/                 # Directory for processed data (Input from Preprocess_Datamodel)
├── notebooks/
│   └── LSTM_Training.ipynb  # Main training notebook
├── src/
│   ├── model_builder.py     # Function to build LSTM model
│   └── train.py             # Script to execute training loop
├── requirements.txt      # Dependencies
└── README.md             # Project documentation
```
## Getting Started
### 1.Prerequisites
  Ensure you have the output files from the Preprocess_Datamodel phase placed in the `data/` directory.

### 2.Installation
  Clone this repository and install the required packages:
  ```python
  git clone [https://github.com/ScenerYOne/Stress_Classification_TensorflowLSTM.git](https://github.com/ScenerYOne/Stress_Classification_TensorflowLSTM.git)
  cd Stress_Classification_TensorflowLSTM
  pip install -r requirements.txt
  ```
### 3.Usage
  Run the training script (or open the Notebook):
  ```python
  python src/train.py 
  ```
### 4.Output Interpretation
  After running, check the result/ directory. Each trial folder contains:
  - metrics_data_{id}.json: specific performance numbers including Training Time, Mean Validation Accuracy (across 5 folds), F1-Score (Weighted), Precision, and Recall.
  - confusion_matrix_{id}.png: To visually inspect which classes are being confused (e.g., distinguishing between 'High' and 'Medium' stress).
 
## Related Repositories
 - Phase 1: Data Cleaning & Preprocessing: ScenerYOne/Preprocess_Datamodel
