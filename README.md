# Merchant Potential Scoring Model

## Project Overview

This project develops a data-driven merchant scoring model to predict the likelihood of a merchant becoming high-value (active and high transaction volume). The model follows a credit risk modeling framework and is designed to support business decision-making in merchant acquisition and prioritization.

## Features

- **Data Preprocessing**: Automated data loading, cleaning, and feature engineering
- **Feature Binning**: Monotonic binning with WOE (Weight of Evidence) calculation
- **Model Training**: Logistic regression with statistical validation
- **Scorecard Generation**: Convert model outputs to business-friendly scoring system
- **Performance Evaluation**: AUC, KS statistics, and ROC analysis
- **Segmentation**: ABCD merchant segmentation based on scores

## Project Structure

```
Merchant-Potential-Scoring-Model/
├── main.py                     # Main entry point
├── config.yaml                 # Configuration file
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── src/                        # Source code modules
│   ├── __init__.py
│   ├── data_preprocessing.py   # Data loading and preprocessing
│   ├── binning_woe.py         # Feature binning and WOE calculation
│   ├── model_training.py       # Model training and evaluation
│   └── scorecard_generator.py  # Scorecard generation
├── data/                       # Data files
├── models/                     # Trained model artifacts
├── results/                    # Output results
├── tests/                      # Unit tests (optional)
└── Merchant Potential Scoring Model.ipynb  # Original notebook
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Merchant-Potential-Scoring-Model
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Quick Start

1. Place your data file in the `data/` directory
2. Update the configuration in `config.yaml` if needed
3. Run the main script:

```bash
python main.py
```

### Configuration

Edit `config.yaml` to customize:
- Input/output file paths
- Feature selection
- Binning parameters
- Model settings
- Scorecard configuration

### Output Files

The model generates several output files:
- `results/海牙地区中餐厅_评分卡打分结果.xlsx` - Final scores and segments
- `models/model_parameters.json` - Trained model parameters
- Various plots and analysis results

## Model Pipeline

1. **Data Loading**: Load merchant data from Excel file
2. **Feature Analysis**: Calculate correlations, VIF, and descriptive statistics
3. **Binning**: Perform monotonic binning for continuous features
4. **WOE Calculation**: Calculate Weight of Evidence for each bin
5. **Model Training**: Train logistic regression model
6. **Scorecard Generation**: Convert model to scoring system
7. **Evaluation**: Assess model performance (AUC, KS)
8. **Segmentation**: Assign merchants to A/B/C/D segments

## Key Features Used

The model uses the following key features:
- **营业时长** (Business Hours) - Strongest predictor (IV=0.57)
- **竞争指数** (Competition Index) - Market competition level (IV=0.36)
- **人气指数** (Popularity Index) - Customer traffic (IV=0.25)
- **经营年限** (Years in Business) - Business experience (IV=0.23)
- **外卖** (Delivery Service) - Delivery availability (IV=0.11)

## Model Performance

- **AUC**: 0.7672 - Good discrimination ability
- **KS**: 0.4779 - Excellent model stability
- **Score Range**: 300-900 points
- **Segmentation**: A (≥700), B (650-700), C (600-650), D (<600)

## Dependencies

- pandas >= 1.5.0
- numpy >= 1.21.0
- scikit-learn >= 1.1.0
- scipy >= 1.9.0
- statsmodels >= 0.13.0
- matplotlib >= 3.5.0

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License.

## Contact

For questions or suggestions, please contact the development team.

---

**Note**: This model is designed for business decision support and should be used in conjunction with domain expertise and business context.
