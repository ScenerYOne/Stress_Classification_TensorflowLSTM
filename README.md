# Stress Classification using TensorFlow LSTM 

> **Repository:** ScenerYOne/Stress_Classification_TensorflowLSTM
> **Phase:** Modeling & Evaluation
> **Preceding Phase:** [ScenerYOne/Preprocess_Datamodel](https://github.com/ScenerYOne/Preprocess_Datamodel)

##  Project Context

This project represents the **Modeling Phase** of the Stress Classification pipeline. It is designed to ingest preprocessed, normalized, and structured data outputs from the [Preprocess_Datamodel](https://github.com/ScenerYOne/Preprocess_Datamodel) repository to train a Deep Learning model.

**Goal:** To detect and classify stress levels from sequential data using **Long Short-Term Memory (LSTM)** networks, leveraging their ability to learn long-term dependencies in time-series or textual sequences.

##  Pipeline Overview

1.  **Data Ingestion:** Load processed datasets (e.g., `.npy`, `.csv`, or pickle files) generated in the preprocessing phase.
2.  **Model Architecture:** Construct a sequential LSTM model using TensorFlow/Keras.
3.  **Training:** Optimize model weights using relevant loss functions and optimizers.
4.  **Evaluation:** Assess performance using Accuracy, Precision, Recall, and F1-Score.

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
  - Language: Python 3.x
  - Data Handling: NumPy, Pandas
  - *Visualization: Matplotlib, Seaborn (for Confusion Matrix & Loss Curves)

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
  Ensure you have the output files from the Preprocess_Datamodel phase placed in the `data/` directory. The model expects input data with shape: `(Samples, Time_Steps, Features)`.

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
  python src/train.py --epochs 20 --batch_size 32
  ```

## Related Repositories
 - Phase 1: Data Cleaning & Preprocessing: ScenerYOne/Preprocess_Datamodel
