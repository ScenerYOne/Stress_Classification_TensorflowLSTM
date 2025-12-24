# Stress Classification using TensorFlow LSTM 

> **Repository:** ScenerYOne/Stress_Classification_TensorflowLSTM
> **Phase:** Modeling & Evaluation + Real-time Deployment
> **Preceding Phase:** [ScenerYOne/Stress-Detection-Project-EDA-PPG-Signal-Preprocessing](https://github.com/ScenerYOne/Stress-Detection-Project-EDA-PPG-Signal-Preprocessing)

##  Project Context

This project represents the **Modeling Phase** of the Stress Classification pipeline. It is designed to ingest preprocessed, normalized, and structured data outputs from the [ScenerYOne/Stress-Detection-Project-EDA-PPG-Signal-Preprocessing](https://github.com/ScenerYOne/Stress-Detection-Project-EDA-PPG-Signal-Preprocessing) repository to train a Deep Learning model.

**Goal:** To detect and classify stress levels from sequential physiological data using **Long Short-Term Memory (LSTM)** networks, leveraging their ability to learn long-term dependencies in time-series sequences.

**Key Enhancement:** This implementation goes beyond static model training by incorporating **Automated Hyperparameter Tuning (Keras Tuner)** and **K-Fold Cross-Validation** to ensure model robustness and generalization.

##  Team Contributions

**My Responsibilities:**
- **Complete Data Management:** Collected, cleaned, preprocessed, and prepared all physiological sensor data
- **Model Development:** Designed, trained, and optimized three deep learning architectures (ANN, LSTM, GRU)
- **Performance Testing:** Conducted comprehensive evaluation and comparison across all models
- **LSTM Implementation:** Selected and fine-tuned LSTM as the primary model for deployment

**Team Members' Contributions:**
- Designed and developed the prototype device and user interface
- Managed web development and system integration

##  Model Selection & Justification

### Experimental Models Comparison

Three deep learning architectures were developed and tested:

**1. ANN (Artificial Neural Network)**
- Standard feedforward architecture
- Best for: Static, non-sequential data
- Limitation: Cannot capture temporal dependencies

**2. LSTM (Long Short-Term Memory)** ✓ **Selected Model**
- Specialized recurrent architecture with memory cells
- **Why LSTM was chosen:**
  - **Ideal for Time-Series Data:** Physiological signals (heart rate, EDA, PPG) are inherently sequential and time-dependent
  - **Long-term Memory Capability:** Can remember and learn from patterns across extended time windows, crucial for detecting stress that develops gradually
  - **Handles Sensitive Data:** LSTM's gating mechanisms (forget, input, output gates) effectively filter noise while preserving critical physiological changes
  - **Sequential Pattern Recognition:** Superior at detecting subtle stress buildup patterns over continuous time periods
  - **Context Awareness:** Learns relationships between multiple physiological signals occurring simultaneously
  - **Proven Performance:** Achieved highest accuracy and F1-score among the three models tested

**3. GRU (Gated Recurrent Unit)**
- Simplified recurrent architecture with fewer parameters
- Best for: Sequential data with moderate dependencies
- Limitation: Less effective for complex long-term patterns compared to LSTM

##  Pipeline Overview

1. **Data Ingestion:** Load processed datasets (typically, `.npy`, `.csv`) with shape (`Sample`, `Time_Steps`, `Features`)
2. **Hyperparameter Search:** Utilizes Keras Tuner to dynamically find the optimal architecture:
   - Units: Searches between 32 to 256 units
   - Layers: Dynamically adjusts between 1 to 5 LSTM layers
   - Dropout: Optimizes dropout rates (0.05 - 0.2) to prevent overfitting
   - Optimizer: Selects between Adam and SGD with varying learning rates
3. **Robust Evaluation (5-Fold CV):** Every hyperparameter trial is evaluated using 5-Fold Cross-Validation. The reported accuracy is the mean of these 5 folds, ensuring the model performs well across different data splits
4. **Logging & Visualization:** Automatically saves confusion matrices, model files (.keras/.h5), and detailed metrics (JSON) for every trial

##  Model Architecture

The LSTM architecture is optimized to handle sequential physiological inputs:

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

**Input Features:** EDA features, PPG features, HRV indices
**Time Steps:** 30 seconds window of physiological data
**Output Classes:** Low, Medium, High stress levels

##  Real-time Deployment & Integration
### Model Deployment Architecture
The trained LSTM model is deployed as a REST API service on **Azure App Service**, enabling real-time stress level predictions from incoming sensor data.

#### **1. Model API Endpoint**

The API receives preprocessed sensor data and returns stress level predictions with confidence scores. It accepts POST requests with physiological features and responds with classification results (Low/Medium/High) along with probability distributions.

```python
# API Structure
POST /api/predict
Content-Type: application/json

{
  "sensor_data": {
    "heart_rate": [72, 75, 78, ...],
    "temperature": [36.5, 36.6, ...],
    "motion": [0.2, 0.3, 0.4, ...],
    "timestamp": "2025-12-19T10:30:00Z"
  }
}

Response:
{
  "stress_level": "Medium",
  "confidence": 0.87,
  "prediction_time": "2025-12-19T10:30:05Z",
  "raw_probabilities": {
    "Low": 0.08,
    "Medium": 0.87,
    "High": 0.05
  }
}
```

#### **2. Signal Processing Flow (Emotibit → Database → Dashboard)**

**Step 1: Data Collection from Emotibit**
- Emotibit wearable device continuously collects physiological signals (heart rate, skin conductance, temperature, motion)
- Data is transmitted via WiFi/Bluetooth every 1-5 seconds
- Signals are sent to Azure IoT Hub or directly to the backend API

**Step 2: Real-time Data Pipeline**
```python
Emotibit Device → Azure IoT Hub → Azure Function (Preprocessor) 
                                          ↓
                                   MongoDB Atlas
                                          ↓
                                   Model API (Prediction)
                                          ↓
                                   MongoDB Atlas (Store Results)
                                          ↓
                                   Frontend Dashboard (Display)
```
**Preprocessing Pipeline:**
- Azure Functions process raw signals to extract features
- EDA features include raw values, cleaned signals, tonic/phasic components, and SCR (Skin Conductance Response) metrics
- PPG features include raw values, cleaned signals, heart rate, and quality indicators
- HRV (Heart Rate Variability) indices are calculated including time-domain, frequency-domain, and nonlinear metrics
- Processed features are stored in **preprocessed_data** collection

**Model Prediction Pipeline:**
- Preprocessed features are fed into the trained LSTM model via Model API
- Model outputs stress classification (Low/Medium/High) with confidence scores
- Prediction results are stored in **stress_predictions** collection
- Results are immediately pushed to the frontend dashboard via railway
  
#### **3. Real-time Dashboard (24/7 Monitoring)**

**Frontend Technology Stack:**
- Framework: Next.js / React for responsive UI
- Real-time Updates: WebSocket for instant data push (auto-refresh every 5 seconds)
- Charting: Chart.js / Recharts for data visualization
- State Management: React Query for automatic data refreshing

**Real-time Update Mechanism:**
- Dashboard automatically refreshes every 5 seconds for live stress levels
- Historical data updates every 1 minute
- railway connection ensures instant updates when new predictions arrive
- MongoDB Change Streams trigger real-time data broadcast to all connected clients


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

##  Model Performance Summary

**LSTM (Selected Model):**
- Best suited for time-series physiological data
- Superior long-term dependency learning
- Highest accuracy in stress classification
- Optimal for real-time deployment

**ANN (Baseline):**
- Standard benchmark comparison
- Limited temporal understanding

**GRU (Alternative):**
- Faster training than LSTM
- Suitable for shorter sequences
- Lower performance on long-term patterns
  
## Related Repositories
 - Phase 1: Data Cleaning & Preprocessing: [ScenerYOne/Stress-Detection-Project-EDA-PPG-Signal-Preprocessing](https://github.com/ScenerYOne/Stress-Detection-Project-EDA-PPG-Signal-Preprocessing) 

## Senior Project
This repository contains my senior project.  
You can view a brief project scope and overview here:  
🔗 https://linkbio.co/wu-senior-care-support
