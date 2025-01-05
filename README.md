XGBOOST Model for Heart Diseaset UCI


## Project Structure

```python
deployment_demo_model_training/
├── data/                     # Input data folder (ignored by Git)
├── models/                   # Trained models folder (ignored by Git)
├── src/                      # Source code
│   ├── data/                 # Data-related modules
│   │   ├── data_loader.py    # Data loading
│   │   ├── data_splitter.py  # Splits data into train/test sets
│   │   ├── data_processor.py # Data preprocessing and transformation
│   ├── model/                # Model-related modules
│   │   ├── trainer.py        # Model training
│   │   ├── evaluator.py      # Model evaluation
│   │   ├── saver.py          # Saves the trained model
│   ├── main.py               # Orchestrates the training pipeline
├── .gitignore                # Specifies files and folders to ignore by Git
├── pyproject.toml            # Poetry configuration
├── README.md                 # Project documentation (this file)
```
---

## Requirements

- Python 3.10 or higher
- Poetry for dependency management

### Dependencies

All dependencies are listed in `pyproject.toml`. Key libraries include:
- `pandas`: Data manipulation
- `xgboost`: Model training
- `scikit-learn`: Data splitting and evaluation
- `joblib`: Saving and loading models

---

## Setup

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/gillopy/deployment_model_xgboost_heart_disease
   cd deployment_model_xgboost_heart_disease
   ```

2. **Install Dependencies**:
   Use Poetry to install all required dependencies.
   ```bash
   poetry install
   ```

3. **Prepare the Environment**:
   - Ensure the `data/` folder contains your dataset.
   - The `models/` folder will store trained models.

---

## Usage

Run the training pipeline using the following command:
```bash
poetry run python src/main.py
```

### Output
- The trained model will be saved in the `models/` folder with the current date appended to its name.

---

## Details

### Data

- **Input Data**:
  - Place the dataset in the `data/` folder.
  - The dataset should include a target column for supervised learning.

- **Preprocessing**:
  - Missing values are handled and numerical features are scaled.
  - Specify columns for imputation in `data_processor.py`.

### Training

- **Model**:
  - The pipeline uses `XGBoost` for training.
  - Default parameters are defined in `trainer.py` but can be modified as needed.

- **Evaluation**:
  - Model performance is evaluated on a test set.
  - Metrics like accuracy are printed to the console.

- **Saving**:
  - Trained models are saved with a timestamp for version control.

---
