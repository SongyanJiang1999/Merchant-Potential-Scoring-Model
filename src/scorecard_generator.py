"""
Scorecard generation module for merchant potential scoring
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple


class ScorecardGenerator:
    """Generate credit scoring card from trained model"""
    
    def __init__(self, pdo: int = 50, base_score: int = 600, base_odds: float = 1/20):
        """
        Initialize scorecard generator
        
        Args:
            pdo: Points to double the odds
            base_score: Base score
            base_odds: Base odds at base score
        """
        self.pdo = pdo
        self.base_score = base_score
        self.base_odds = base_odds
        self.factor = pdo / np.log(2)
        self.offset = None
        self.scorecard_tables = {}
        
    def calculate_offset(self, intercept: float):
        """Calculate offset for scorecard scaling"""
        self.offset = self.base_score + self.factor * (np.log(self.base_odds) - intercept)
        return self.offset
    
    def calculate_scores(self, X_woe: pd.DataFrame, coefficients: np.ndarray, 
                        intercept: float, score_range: Tuple[int, int] = (300, 900)) -> pd.Series:
        """
        Calculate scores for all samples
        
        Args:
            X_woe: WOE transformed features
            coefficients: Model coefficients
            intercept: Model intercept
            score_range: Target score range (min, max)
            
        Returns:
            Series of calculated scores
        """
        # Calculate offset
        self.calculate_offset(intercept)
        
        # Calculate linear part
        linear_part = intercept + X_woe.values @ coefficients
        
        # Scale to target range
        lin_min = linear_part.min()
        lin_max = linear_part.max()
        
        if lin_max == lin_min:
            raise ValueError("linear_part has no variation, check model or data.")
        
        score_min, score_max = score_range
        score_range_width = score_max - score_min
        
        scores = score_min + (linear_part - lin_min) / (lin_max - lin_min) * score_range_width
        
        print("\n===== 评分卡刻度参数 =====")
        print(f"PDO        = {self.pdo}")
        print(f"BASE_SCORE = {self.base_score}")
        print(f"BASE_ODDS  = {self.base_odds}")
        print(f"Factor     = {self.factor}")
        print(f"Offset     = {self.offset}")
        
        return scores
    
    def generate_scorecard_tables(self, X_woe: pd.DataFrame, coefficients: np.ndarray,
                                 bin_map: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Generate scorecard tables for each feature
        
        Args:
            X_woe: WOE transformed features
            coefficients: Model coefficients
            bin_map: Dictionary of binning tables for each feature
            
        Returns:
            Dictionary of scorecard tables
        """
        feature_woe_cols = X_woe.columns.tolist()
        
        for col_woe, beta in zip(feature_woe_cols, coefficients):
            # Get original variable name
            orig_var = col_woe.replace("_WOE", "")
            
            # Get binning table
            if orig_var not in bin_map:
                continue
                
            bt = bin_map[orig_var].copy()
            
            # Calculate score contribution for each bin
            bt["Score_contrib"] = self.factor * beta * bt["WOE"]
            
            self.scorecard_tables[orig_var] = bt[["bin", "WOE", "Score_contrib"]]
            
            print(f"\n===== {orig_var} 的分箱评分表 =====")
            print(self.scorecard_tables[orig_var])
        
        return self.scorecard_tables
    
    def assign_segments(self, scores: np.ndarray, 
                       thresholds: Dict[str, int] = None) -> pd.Series:
        """
        Assign segments based on scores
        
        Args:
            scores: Score array or series
            thresholds: Dictionary of thresholds for each segment
            
        Returns:
            Series of segment assignments
        """
        if thresholds is None:
            thresholds = {
                'A': 700,  # High potential
                'B': 650,  # Medium-high potential
                'C': 600,  # Average potential
                'D': 0     # Low potential
            }
        
        def assign_segment(score):
            if score >= thresholds['A']:
                return "A"
            elif score >= thresholds['B']:
                return "B"
            elif score >= thresholds['C']:
                return "C"
            else:
                return "D"
        
        # Convert to list if numpy array
        if isinstance(scores, np.ndarray):
            scores_list = scores.tolist()
        else:
            scores_list = scores
            
        segments = [assign_segment(score) for score in scores_list]
        segments_series = pd.Series(segments, index=scores.index if hasattr(scores, 'index') else None)
        
        print("\n===== 各档位数量统计 =====")
        print(pd.Series(segments).value_counts().sort_index())
        
        return segments_series
    
    def get_scorecard_summary(self) -> Dict:
        """Get summary of scorecard parameters"""
        return {
            'pdo': self.pdo,
            'base_score': self.base_score,
            'base_odds': self.base_odds,
            'factor': self.factor,
            'offset': self.offset,
            'n_features': len(self.scorecard_tables)
        }
    
    def export_scorecard(self, data: pd.DataFrame, scores: pd.Series, 
                        segments: pd.Series, output_path: str):
        """
        Export scorecard results to Excel
        
        Args:
            data: Original data with scores and segments
            scores: Calculated scores
            segments: Assigned segments
            output_path: Output file path
        """
        # Create output dataframe
        output_data = data.copy()
        output_data["scorecard_score"] = scores
        output_data["segment"] = segments
        
        # Export to Excel
        output_data.to_excel(output_path, index=False)
        print(f"\n评分卡打分与分层已导出到：{output_path}")
        
        return output_data
