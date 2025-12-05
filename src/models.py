"""
HRMTA model pipeline. Robust Spatial-Physics Stacking.
"""
import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import HuberRegressor 
from pykrige.ok import OrdinaryKriging
from typing import List, Tuple
import warnings

from .config import (
    LIGHTGBM_PARAMS, CRS_POLAND,
    RESIDUAL_KRIGING_VARIOGRAM, USE_RESIDUAL_KRIGING
)

# Baseline
class SimpleKrigingBaseline(BaseEstimator, RegressorMixin):
    """Simple Ordinary Kriging baseline."""
    def __init__(self, variogram_model: str = "spherical"):
        self.variogram_model = variogram_model
        self.kriging_model = None
        self.mean_temp = None
    
    def fit(self, gdf: gpd.GeoDataFrame):
        print("\nTraining Simple Kriging Baseline...")
        self.mean_temp = gdf['temp'].mean()
        gdf_proj = gdf.to_crs(CRS_POLAND)
        x = gdf_proj.geometry.x.values
        y = gdf_proj.geometry.y.values
        z = gdf['temp'].values
        
        try:
            # Add tiny jitter to prevent singular matrix if points are identical
            x = x + np.random.uniform(-0.1, 0.1, size=x.shape)
            y = y + np.random.uniform(-0.1, 0.1, size=y.shape)
            
            self.kriging_model = OrdinaryKriging(
                x, y, z, variogram_model=self.variogram_model,
                verbose=False, enable_plotting=False
            )
            print(f"✓ Simple Kriging trained on {len(x)} points")
        except Exception as e:
            print(f"❌ Kriging failed: {e}")
            self.kriging_model = None
        return self
    
    def predict(self, gdf: gpd.GeoDataFrame) -> np.ndarray:
        if self.kriging_model is None: return np.full(len(gdf), self.mean_temp)
        gdf_proj = gdf.to_crs(CRS_POLAND)
        try:
            predictions, _ = self.kriging_model.execute(
                'points', gdf_proj.geometry.x.values, gdf_proj.geometry.y.values
            )
            return predictions
        except Exception:
            return np.full(len(gdf), self.mean_temp)

# HRMTA core architecture
class PhysicsTrendEnvMLModel(BaseEstimator, RegressorMixin):
    """
    Robust Stacking:
    Stage 1: Macro-Trend (Huber on DEM + Lat + Lon)
    Stage 2: Meso-EnvML (Regularized LightGBM on residuals)
    Stage 3: Micro-Kriging (Local residuals)
    """
    def __init__(self, trend_features: List[str], env_features: List[str], 
                 lgbm_params: dict = None, use_kriging: bool = USE_RESIDUAL_KRIGING):
        self.trend_features = trend_features
        self.env_features = env_features
        self.lgbm_params = lgbm_params or LIGHTGBM_PARAMS.copy()
        self.use_kriging = use_kriging
        
        self.trend_model = None
        self.ml_model = None
        self.kriging_model = None
        self.trend_scaler = RobustScaler()
        self.env_medians = {}

    def _get_data(self, gdf, features, scaler=None, fit=False, impute=False):
        """Helper to extract and scale/impute data."""
        X = pd.DataFrame(index=gdf.index)
        
        for col in features:
            if col in gdf.columns:
                vals = gdf[col].values.copy()
                if impute:
                    if fit: self.env_medians[col] = np.nanmedian(vals)
                    vals[np.isnan(vals)] = self.env_medians.get(col, 0)
                X[col] = vals
            else:
                 X[col] = self.env_medians.get(col, 0)

        # Final NaN check (fill with 0 if median failed)
        X = X.fillna(0) 
        
        if scaler:
            if fit: return scaler.fit_transform(X)
            return scaler.transform(X)
        return X.values

    def fit(self, gdf: gpd.GeoDataFrame):
        print("\nTraining the Robust Stacking Model...")
        df_train = gdf.dropna(subset=['temp'] + self.trend_features)
        y = df_train['temp'].values
        
        # Stage 1: Macro-Trend
        print(f"   Stage 1: Macro-Trend (Huber on {self.trend_features})")
        X_trend = self._get_data(df_train, self.trend_features, self.trend_scaler, fit=True, impute=True)
        self.trend_model = HuberRegressor(epsilon=1.35, max_iter=300)
        self.trend_model.fit(X_trend, y)
        resid1 = y - self.trend_model.predict(X_trend)
        print(f"      Base RMSE: {np.sqrt(np.mean(resid1**2)):.3f}°C")
        
        # Stage 2: EnvML
        print(f"   Stage 2: Regularized LightGBM on {len(self.env_features)} env features")
        X_env = self._get_data(df_train, self.env_features, fit=True, impute=True)
        from lightgbm import LGBMRegressor
        with warnings.catch_warnings(): # silence LGBM warnings
            warnings.simplefilter("ignore")
            self.ml_model = LGBMRegressor(**self.lgbm_params)
            self.ml_model.fit(X_env, resid1)
        resid2 = resid1 - self.ml_model.predict(X_env)
        current_rmse = np.sqrt(np.mean(resid2**2))
        r2_imp = 1 - (np.var(resid2) / (np.var(resid1) + 1e-10))
        print(f"      EnvML added R²: {r2_imp:.3f} | Remaining RMSE: {current_rmse:.3f}°C")

        # Stage 3: Kriging
        if self.use_kriging and current_rmse > 0.15 and len(df_train) > 50:
            print(f"   Stage 3: Micro-Scale Kriging ({RESIDUAL_KRIGING_VARIOGRAM})")
            gdf_proj = df_train.to_crs(CRS_POLAND)
            
            # downsample for speed if needed, keep core structure
            if len(gdf_proj) > 3000:
                idx = np.random.choice(len(gdf_proj), 3000, replace=False)
                x_k, y_k, z_k = gdf_proj.geometry.x.values[idx], gdf_proj.geometry.y.values[idx], resid2[idx]
            else:
                x_k, y_k, z_k = gdf_proj.geometry.x.values, gdf_proj.geometry.y.values, resid2
            
            try:
                # Jitter for robustness
                x_k += np.random.uniform(-0.1, 0.1, size=x_k.shape)
                y_k += np.random.uniform(-0.1, 0.1, size=y_k.shape)
                self.kriging_model = OrdinaryKriging(
                    x_k, y_k, z_k, variogram_model=RESIDUAL_KRIGING_VARIOGRAM,
                    verbose=False, enable_plotting=False,
                    nlags=20 # robust setting
                )
                print("      ✓ Kriging fitted")
            except Exception as e:
                print(f"      ⚠️ Kriging failed: {e}, skipping Stage 3")
                self.kriging_model = None
        else:
            print("   Stage 3: Skipped (residuals small or insufficient data)")
            self.kriging_model = None
        return self
    
    def predict(self, gdf: gpd.GeoDataFrame) -> np.ndarray:
        # Stage 1
        X_trend = self._get_data(gdf, self.trend_features, self.trend_scaler, fit=False, impute=True)
        preds = self.trend_model.predict(X_trend)
        
        # Stage 2
        X_env = self._get_data(gdf, self.env_features, fit=False, impute=True)
        preds += self.ml_model.predict(X_env)
        
        # Stage 3
        if self.kriging_model:
            gdf_proj = gdf.to_crs(CRS_POLAND)
            try:
                k_pred, _ = self.kriging_model.execute(
                    'points', gdf_proj.geometry.x.values, gdf_proj.geometry.y.values,
                    backend='loop' # Slower but safer against memory/singular errors
                )
                # Sanity clamp kriging residuals to prevent explosions far from data
                k_pred = np.clip(k_pred, -5.0, 5.0) 
                preds += k_pred
            except Exception:
                pass # Fallback to S1+S2
        return preds

    def get_feature_importance(self) -> pd.DataFrame:
        if self.ml_model is None: return pd.DataFrame()
        imp = self.ml_model.feature_importances_
        if imp.sum() > 0: imp = 100.0 * (imp / imp.sum())
        df = pd.DataFrame({'feature': self.env_features, 'importance': imp}).sort_values('importance', ascending=False)
        
        # Add Trend features conceptually
        trend_rows = pd.DataFrame({'feature': [f'[S1] {f}' for f in self.trend_features], 'importance': [np.nan]*len(self.trend_features)})
        return pd.concat([trend_rows, df]).reset_index(drop=True)

class EnsembleHybridModel(BaseEstimator, RegressorMixin):
    """Ensemble wrapper for the model."""
    def __init__(self, trend_features, env_features, n_models=5, seeds=None, **kwargs):
        self.trend_features = trend_features
        self.env_features = env_features
        self.n_models = n_models
        self.seeds = seeds or list(range(42, 42 + n_models))
        self.model_kwargs = kwargs
        self.models = []

    def fit(self, gdf: gpd.GeoDataFrame):
        print(f"\nTraining the Ensemble ({self.n_models} models)...")
        self.models = []
        for i, seed in enumerate(self.seeds[:self.n_models], 1):
            # Inject diversity
            current_kwargs = self.model_kwargs.copy()
            lgbm_params = current_kwargs.get('lgbm_params', LIGHTGBM_PARAMS.copy()).copy()
            lgbm_params['random_state'] = seed
            
            # remove lgbm_params from current_kwargs to avoid duplicate
            current_kwargs.pop('lgbm_params', None) 
            
            # Slight data subsampling for diversity (bagging)
            gdf_sample = gdf.sample(frac=0.95, random_state=seed)
            
            model = PhysicsTrendEnvMLModel(
                self.trend_features, 
                self.env_features, 
                lgbm_params=lgbm_params,
                **current_kwargs
            )
            model.fit(gdf_sample)
            self.models.append(model)
        return self

    def predict_with_uncertainty(self, gdf: gpd.GeoDataFrame) -> Tuple[np.ndarray, np.ndarray]:
        preds = np.array([model.predict(gdf) for model in self.models])
        # Robust statistics
        mean_pred = np.nanmedian(preds, axis=0)
        # Uncertainty = Interquartile Range scaled to sigma
        q75, q25 = np.nanpercentile(preds, [75 ,25], axis=0)
        uncertainty = (q75 - q25) / 1.35
        return mean_pred, uncertainty

    def get_feature_importance(self) -> pd.DataFrame:
        all_dfs = [m.get_feature_importance().set_index('feature') for m in self.models]
        if not all_dfs: return pd.DataFrame()
        combined = pd.concat(all_dfs, axis=1)
        df = pd.DataFrame({
            'feature': combined.index,
            'importance': combined.mean(axis=1).values,
            'importance_std': combined.std(axis=1).values
        }).sort_values('importance', ascending=False)
        return df