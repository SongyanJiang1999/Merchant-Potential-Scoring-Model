"""
Model training and evaluation module for merchant potential scoring
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from typing import Tuple, Dict, Any


class ModelTrainer:
    """Handle model training and evaluation"""
    
    def __init__(self, target_col: str = "YabandPay客户"):
        self.target_col = target_col
        self.model = None
        self.coefficients = None
        self.intercept = None
        
    def train_logistic_regression(self, X: pd.DataFrame, y: pd.Series, 
                                 max_iter: int = 1000, solver: str = "lbfgs") -> LogisticRegression:
        """
        Train logistic regression model
        
        Args:
            X: Feature matrix
            y: Target variable
            max_iter: Maximum iterations
            solver: Solver to use
            
        Returns:
            Trained logistic regression model
        """
        self.model = LogisticRegression(max_iter=max_iter, solver=solver)
        self.model.fit(X, y)
        
        self.coefficients = self.model.coef_[0]
        self.intercept = self.model.intercept_[0]
        
        # Display model coefficients
        coef_df = pd.DataFrame({
            "WOE变量名": X.columns,
            "系数β": self.coefficients
        })
        print("\n===== 基于 WOE 的 Logistic 回归系数 =====")
        print(coef_df)
        print(f"\n截距 β0 = {self.intercept}")
        
        return self.model
    
    def evaluate_model(self, y_true: pd.Series, scores: pd.Series, 
                      plot_roc: bool = True) -> Tuple[float, float, pd.DataFrame]:
        """
        Evaluate model performance using AUC and KS statistics
        
        Args:
            y_true: True labels
            scores: Predicted scores
            plot_roc: Whether to plot ROC curve
            
        Returns:
            Tuple of (AUC, KS, KS dataframe)
        """
        # Calculate AUC
        auc_value = roc_auc_score(y_true, scores)
        
        # Calculate KS
        df = pd.DataFrame({"y": y_true, "score": scores}).sort_values("score", ascending=False)
        
        # Assume y=1 is "good" and y=0 is "bad"
        df["good"] = (df["y"] == 1).astype(int)
        df["bad"] = (df["y"] == 0).astype(int)
        
        total_good = df["good"].sum()
        total_bad = df["bad"].sum()
        
        df["cum_good_pct"] = df["good"].cumsum() / (total_good if total_good > 0 else 1)
        df["cum_bad_pct"] = df["bad"].cumsum() / (total_bad if total_bad > 0 else 1)
        
        df["ks"] = df["cum_good_pct"] - df["cum_bad_pct"]
        ks_value = df["ks"].max()
        
        # Plot ROC curve
        if plot_roc:
            self._plot_roc_curve(y_true, scores, auc_value)
        
        return auc_value, ks_value, df
    
    def _plot_roc_curve(self, y_true: pd.Series, scores: pd.Series, auc_value: float):
        """Plot ROC curve"""
        fpr, tpr, _ = roc_curve(y_true, scores)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f"AUC = {auc_value:.3f}")
        plt.plot([0, 1], [0, 1], "--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend()
        plt.grid(alpha=0.4, linestyle="--")
        plt.tight_layout()
        plt.show()
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get model summary information"""
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        
        return {
            "coefficients": self.coefficients,
            "intercept": self.intercept,
            "n_features": len(self.coefficients),
            "feature_names": list(self.model.feature_names_in_) if hasattr(self.model, 'feature_names_in_') else None
        }
