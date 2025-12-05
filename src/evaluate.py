"""
Evaluation metrics for temperature predictions.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def evaluate_predictions(y_true, y_pred) -> dict:
    """
    Calculate evaluation metrics
    
    Args:
        y_true: Ground truth (numpy array or pandas series)
        y_pred: Predictions (numpy array or pandas series)
    
    Returns:
        dictionary with MAE, RMSE, R², and bias
    """
    # Convert to numpy arrays if pandas series/dataframe
    if hasattr(y_true, 'values'):
        y_true = y_true.values
    if hasattr(y_pred, 'values'):
        y_pred = y_pred.values
    
    # ensure 1D numpy arrays
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    
    # remove NaN pairs
    valid_mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[valid_mask]
    y_pred = y_pred[valid_mask]
    
    if len(y_true) == 0:
        raise ValueError("No valid prediction pairs found. All are NaN.")
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    bias = np.mean(y_pred - y_true)
    
    metrics = {
        "MAE": mae,
        "RMSE": rmse,
        "R²": r2,
        "Bias": bias,
        "Min Error": np.min(y_pred - y_true),
        "Max Error": np.max(y_pred - y_true)
    }
    
    return metrics

def print_metrics(metrics: dict, title: str = "Evaluation Metrics"):
    """Print evaluation metrics"""
    print(f"{title:^60}")
    print(f"  MAE (Mean Absolute Error):    {metrics['MAE']:>8.3f} °C")
    print(f"  RMSE (Root Mean Squared):     {metrics['RMSE']:>8.3f} °C")
    print(f"  R² (Coefficient of Det.):     {metrics['R²']:>8.3f}")
    print(f"  Bias (Mean Error):            {metrics['Bias']:>8.3f} °C")
    print(f"  Min Error:                    {metrics['Min Error']:>8.3f} °C")
    print(f"  Max Error:                    {metrics['Max Error']:>8.3f} °C")

def compare_models(
    y_true: np.ndarray,
    pred_baseline: np.ndarray,
    pred_hybrid: np.ndarray,
    model_names: tuple = ("Simple Kriging", "Hybrid Model")
) -> pd.DataFrame:
    """
    Compare two models on the same test set
    
    Returns:
        DataFrame with metrics for both models and improvement percentages
    """
    # Calculate metrics for both
    metrics_baseline = evaluate_predictions(y_true, pred_baseline)
    metrics_hybrid = evaluate_predictions(y_true, pred_hybrid)
    
    # Calculate improvements
    improvements = {}
    for key in metrics_baseline.keys():
        if key in ['R²']:
            # for R², higher is better
            improvements[key] = metrics_hybrid[key] - metrics_baseline[key]
        else:
            # for errors, lower is better
            improvements[key] = metrics_baseline[key] - metrics_hybrid[key]
    
    # Create comparison DataFrame
    comparison = pd.DataFrame({
        model_names[0]: metrics_baseline,
        model_names[1]: metrics_hybrid,
        'Improvement': improvements
    })
    
    # Add percentage improvements
    pct_improvements = {}
    for key in metrics_baseline.keys():
        if key == 'R²':
            # for R², calculate absolute difference
            pct_improvements[key] = 100 * improvements[key]
        elif key == 'Bias':
            # for bias, calculate absolute improvement
            pct_improvements[key] = np.abs(metrics_baseline[key]) - np.abs(metrics_hybrid[key])
        else:
            # for errors, calculate percentage reduction
            if metrics_baseline[key] != 0:
                pct_improvements[key] = 100 * improvements[key] / metrics_baseline[key]
            else:
                pct_improvements[key] = 0
    
    comparison['% Improvement'] = pd.Series(pct_improvements)
    
    return comparison

def print_model_comparison(comparison_df: pd.DataFrame, title: str = "Model Comparison"):
    """Print model comparison"""
    print(f"{title:^80}")
    
    # format the dataframe
    for metric in comparison_df.index:
        baseline = comparison_df.iloc[comparison_df.index == metric, 0].values[0]
        hybrid = comparison_df.iloc[comparison_df.index == metric, 1].values[0]
        improvement = comparison_df.iloc[comparison_df.index == metric, 2].values[0]
        pct_improvement = comparison_df.iloc[comparison_df.index == metric, 3].values[0]
        
        # color coding for improvement
        if metric == 'R²':
            status = "✓" if improvement > 0 else "✗"
        elif metric == 'Bias':
            status = "✓" if abs(hybrid) < abs(baseline) else "✗"
        else:
            status = "✓" if improvement > 0 else "✗"
        
        print(f"{metric:25s}: Baseline: {baseline:8.3f}  ->  Hybrid: {hybrid:8.3f}  "
              f"({improvement:+.3f}, {pct_improvement:+5.1f}%) {status}")