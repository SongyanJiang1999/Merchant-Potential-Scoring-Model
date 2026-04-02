"""
Binning and Weight of Evidence (WOE) calculation module
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple


class BinningWOE:
    """Handle feature binning and WOE calculation"""
    
    def __init__(self, target_col: str = "YabandPay客户"):
        self.target_col = target_col
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
    
    def bin_and_check_monotonic(self, df: pd.DataFrame, feature: str, target: str,
                               n_bins: int = 5, monotonic_trend: str = "auto",
                               init_bins: int = 20, min_bin_pct: float = 0.05,
                               plot: bool = True) -> pd.DataFrame:
        """
        Perform monotonic binning and calculate WOE/IV
        
        Args:
            df: Input dataframe
            feature: Feature name to bin
            target: Target variable name
            n_bins: Maximum number of bins
            monotonic_trend: "auto", "ascending", or "descending"
            init_bins: Initial number of bins
            min_bin_pct: Minimum percentage of samples per bin
            plot: Whether to plot results
            
        Returns:
            DataFrame with binning results and WOE/IV values
        """
        tmp = df[[feature, target]].dropna().copy()
        tmp = tmp.sort_values(feature)
        x = tmp[feature].values
        y = tmp[target].values

        n = len(tmp)
        if n == 0:
            raise ValueError("数据为空，无法分箱。")

        # 1. Initial equal-frequency binning
        indices = np.arange(n)
        groups = np.array_split(indices, min(init_bins, n))

        bins = []
        for g in groups:
            if len(g) == 0:
                continue
            xv = x[g]
            yv = y[g]
            count = len(g)
            good = (yv == 1).sum()
            bad = (yv == 0).sum()
            bins.append({
                "left": xv.min(),
                "right": xv.max(),
                "count": count,
                "good": good,
                "bad": bad,
                "event_rate": good / count if count > 0 else 0.0,
                "mean_x": xv.mean()
            })

        # Helper function to merge two adjacent bins
        def merge_two(b1, b2):
            total_count = b1["count"] + b2["count"]
            if total_count == 0:
                event_rate = 0.0
                mean_x = 0.0
            else:
                event_rate = (b1["good"] + b2["good"]) / total_count
                mean_x = (b1["mean_x"] * b1["count"] + b2["mean_x"] * b2["count"]) / total_count
            return {
                "left": b1["left"],
                "right": b2["right"],
                "count": total_count,
                "good": b1["good"] + b2["good"],
                "bad": b1["bad"] + b2["bad"],
                "event_rate": event_rate,
                "mean_x": mean_x
            }

        # 2. Merge bins with too few samples
        min_count = max(1, int(min_bin_pct * n))
        merged = True
        while merged and len(bins) > 1:
            merged = False
            for i, b in enumerate(bins):
                if b["count"] < min_count:
                    # Merge with adjacent bin (prefer the one with fewer samples)
                    if i == 0:
                        j = 1
                    elif i == len(bins) - 1:
                        j = i - 1
                    else:
                        j = i - 1 if bins[i - 1]["count"] <= bins[i + 1]["count"] else i + 1

                    new_bin = merge_two(bins[min(i, j)], bins[max(i, j)])

                    new_bins = []
                    for k, b2 in enumerate(bins):
                        if k in (i, j):
                            continue
                        new_bins.append(b2)
                    new_bins.append(new_bin)
                    new_bins = sorted(new_bins, key=lambda d: d["left"])
                    bins = new_bins
                    merged = True
                    break

        # 3. Determine monotonic direction
        er = np.array([b["event_rate"] for b in bins])
        mx = np.array([b["mean_x"] for b in bins])

        if monotonic_trend == "auto":
            if len(bins) >= 2 and er.std() > 0:
                corr = np.corrcoef(mx, er)[0, 1]
                trend = "ascending" if corr >= 0 else "descending"
            else:
                trend = "ascending"
        else:
            trend = monotonic_trend

        def is_monotonic(er_values, mode):
            if mode == "ascending":
                return np.all(np.diff(er_values) >= -1e-12)
            else:  # descending
                return np.all(np.diff(er_values) <= 1e-12)

        # 4. Merge adjacent bins to enforce monotonicity and control bin count
        while (len(bins) > n_bins) or (not is_monotonic(er, trend) and len(bins) > 1):
            viol_index = None
            if trend == "ascending":
                for i in range(len(er) - 1):
                    if er[i+1] < er[i] - 1e-12:
                        viol_index = i
                        break
            else:  # descending
                for i in range(len(er) - 1):
                    if er[i+1] > er[i] + 1e-12:
                        viol_index = i
                        break

            if viol_index is None:
                # Already monotonic but still too many bins, merge the last two
                viol_index = len(bins) - 2

            new_bin = merge_two(bins[viol_index], bins[viol_index + 1])
            new_bins = []
            for k, b in enumerate(bins):
                if k in (viol_index, viol_index + 1):
                    continue
                new_bins.append(b)
            new_bins.append(new_bin)
            new_bins = sorted(new_bins, key=lambda d: d["left"])
            bins = new_bins
            er = np.array([b["event_rate"] for b in bins])

        # 5. Calculate WOE / IV
        total_good = (y == 1).sum()
        total_bad = (y == 0).sum()
        eps = 1e-6

        rows = []
        for b in bins:
            dist_good = b["good"] / total_good if total_good > 0 else 0
            dist_bad = b["bad"] / total_bad if total_bad > 0 else 0
            woe = np.log((dist_good + eps) / (dist_bad + eps)) if dist_good > 0 and dist_bad > 0 else 0.0
            iv = (dist_good - dist_bad) * woe
            rows.append({
                "bin": f"[{b['left']:.4f}, {b['right']:.4f}]",
                "总数": b["count"],
                "good(1)": b["good"],
                "bad(0)": b["bad"],
                "客户率": b["event_rate"],
                "dist_good": dist_good,
                "dist_bad": dist_bad,
                "WOE": woe,
                "IV": iv
            })

        bin_table = pd.DataFrame(rows)

        print(f"\n===== {feature} 的WOE和IV =====")
        print(bin_table)

        # 6. Plot results
        if plot and len(bin_table) > 0:
            self._plot_binning_results(bin_table, feature)

        return bin_table
    
    def _plot_binning_results(self, bin_table: pd.DataFrame, feature: str):
        """Plot binning results"""
        x_axis = range(len(bin_table))
        x_labels = bin_table["bin"].astype(str)

        # Event rate plot
        plt.figure(figsize=(10, 6))
        plt.plot(x_axis, bin_table["客户率"], marker='o')
        plt.xticks(x_axis, x_labels, rotation=45)
        plt.ylabel("客户率")
        plt.title(f"{feature} 分箱后客户率（单调性检查）")
        plt.tight_layout()
        plt.show()

        # WOE plot
        plt.figure(figsize=(10, 6))
        plt.plot(x_axis, bin_table["WOE"], marker='o')
        plt.xticks(x_axis, x_labels, rotation=45)
        plt.ylabel("WOE")
        plt.title(f"{feature} 分箱后 WOE（单调性检查）")
        plt.tight_layout()
        plt.show()
    
    def woe_for_binary(self, df: pd.DataFrame, feature: str, target: str) -> pd.DataFrame:
        """Calculate WOE for binary features"""
        tmp = df[[feature, target]].dropna()
        total_good = (tmp[target] == 1).sum()
        total_bad = (tmp[target] == 0).sum()
        eps = 1e-6

        rows = []
        for val, sub in tmp.groupby(feature):
            good = (sub[target] == 1).sum()
            bad = (sub[target] == 0).sum()
            total = good + bad

            dist_good = good / total_good if total_good > 0 else eps
            dist_bad = bad / total_bad if total_bad > 0 else eps

            woe = np.log((dist_good + eps) / (dist_bad + eps))
            iv = (dist_good - dist_bad) * woe

            rows.append({
                feature: val,
                '总数': total,
                'good(1)': good,
                'bad(0)': bad,
                '客户率': good / total if total > 0 else 0.0,
                'dist_good': dist_good,
                'dist_bad': dist_bad,
                'WOE': woe,
                'IV': iv
            })

        result = pd.DataFrame(rows)
        print(f'\n===== {feature} 的WOE和IV =====')
        print(result)
        return result
    
    def calculate_iv_summary(self, df: pd.DataFrame, numeric_features: List[str], 
                           binary_features: List[str]) -> pd.DataFrame:
        """Calculate IV summary for all features"""
        numeric_iv_results = []
        
        for col in numeric_features:
            bin_table = self.bin_and_check_monotonic(df, col, self.target_col, plot=False)
            total_iv = bin_table["IV"].sum()
            numeric_iv_results.append({"变量名": col, "类型": "数值型", "IV": total_iv})
        
        binary_iv_results = []
        for col in binary_features:
            bin_table = self.woe_for_binary(df, col, self.target_col)
            total_iv = bin_table["IV"].sum()
            binary_iv_results.append({"变量名": col, "类型": "二值型", "IV": total_iv})
        
        results = numeric_iv_results + binary_iv_results
        return pd.DataFrame(results).sort_values(by="IV", ascending=False)
    
    def apply_woe(self, series: pd.Series, bin_table: pd.DataFrame) -> np.ndarray:
        """Apply WOE transformation to a feature series"""
        woe_list = []
        
        for val in series:
            assigned = False
            for _, row in bin_table.iterrows():
                # Parse interval
                left, right = row["bin"][1:-1].split(",")
                left, right = float(left), float(right)

                if left <= val <= right:
                    woe_list.append(row["WOE"])
                    assigned = True
                    break
            
            if not assigned:
                woe_list.append(0)  # Default value if no bin matches
        
        return np.array(woe_list)
