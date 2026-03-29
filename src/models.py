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
from scipy.spatial import cKDTree
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
# Source trust weights for Kriging nugget and LightGBM sample weighting
SOURCE_TRUST = {
    'IMGW': 1.0, 'EDWIN': 1.0,
    'TRAX': 0.9,
    'NETATMO': 0.6,
}
SOURCE_TRUST_DEFAULT = 0.5


class PhysicsTrendEnvMLModel(BaseEstimator, RegressorMixin):
    """
    Robust Stacking:
    Stage 1: Macro-Trend (Huber on DEM + Lat + Lon)
    Stage 2: Meso-EnvML (Regularized LightGBM on residuals)
    Stage 3: Micro-Kriging (Local residuals)
    """
    # NWP temperature features absorbed into the Adaptive Prior (not passed to LightGBM)
    _PRIOR_ABSORBED = {'nwp_t2m', 'icon_t2m'}

    def __init__(self, trend_features: List[str], env_features: List[str], 
                 lgbm_params: dict = None, use_kriging: bool = USE_RESIDUAL_KRIGING,
                 kriging_scale: float = 0.7,
                 use_adaptive_prior: bool = True,
                 nwp_trust_sigma: float = 1.0):
        self.trend_features = trend_features
        self.env_features = env_features
        self.lgbm_params = lgbm_params or LIGHTGBM_PARAMS.copy()
        self.use_kriging = use_kriging
        self.kriging_scale = kriging_scale
        self.use_adaptive_prior = use_adaptive_prior
        self.nwp_trust_sigma = nwp_trust_sigma
        
        self.trend_model = None
        self.ml_model = None
        self.kriging_model = None
        self.kriging_station_tree = None  # For adaptive scaling
        self.trend_scaler = RobustScaler()
        self.env_medians = {}
        self._has_adaptive_prior = False  # Set during fit

    # NWP-derived features that should keep NaN (LightGBM handles natively)
    _NWP_FEATURES = {
        'nwp_t2m', 'nwp_t2m_anomaly', 'nwp_cloud', 'nwp_wind',
        'nwp_local_error', 'nwp_signed_bias', 'nwp_regime_stability',
        'nwp_elev_mismatch', 'nwp_model_agreement',
        'icon_t2m', 'icon_cloud', 'icon_wind',
        'decoupling_index', 'radiation_loss', 'cold_pool_activation',
        'wind_exposure', 'calm_clear_night',
        'hand_cold_pool',
        'nwp_trust',
    }
    
    # Features excluded from LightGBM (redundant bijection of nwp_debiased_error)
    _LGBM_EXCLUDE = {'nwp_trust'}

    # NWP quality gate thresholds (calibrated)
    _GATE_EXCELLENT_LO = 0.2
    _GATE_EXCELLENT_HI = 1.0
    _GATE_POOR_LO = 1.5
    _GATE_POOR_HI = 3.0
    _GATE_POOR_FLOOR = 0.4
    # Prior safety
    _PRIOR_SAFETY_THRESHOLD = 1.02
    # Multi-model consensus
    _CONSENSUS_SCALE = 1.5

    def _get_data(self, gdf, features, scaler=None, fit=False, impute=False):
        """Helper to extract and scale/impute data."""
        X = pd.DataFrame(index=gdf.index)
        
        for col in features:
            is_nwp = col in self._NWP_FEATURES
            if col in gdf.columns:
                vals = gdf[col].values.copy()
                if impute and not is_nwp:
                    if fit: self.env_medians[col] = np.nanmedian(vals)
                    vals[np.isnan(vals)] = self.env_medians.get(col, 0)
                X[col] = vals
            else:
                # Missing column: NaN for NWP, median/0 for terrain
                X[col] = np.nan if is_nwp else self.env_medians.get(col, 0)

        # Fill remaining NaN in NON-NWP columns only
        non_nwp_cols = [c for c in X.columns if c not in self._NWP_FEATURES]
        X[non_nwp_cols] = X[non_nwp_cols].fillna(0)
        
        if scaler:
            if fit: return scaler.fit_transform(X)
            return scaler.transform(X)
        return X.values

    def _compute_adaptive_prior(self, gdf, huber_pred):
        """Compute the spatially-gated adaptive prior.

        Returns:
        - T_prior: array of prior temperature estimates.
        - has_prior: True if NWP data was available for prior computation.
        """
        has_nwp = (
            'nwp_t2m' in gdf.columns and
            'nwp_signed_bias' in gdf.columns and
            gdf['nwp_t2m'].notna().sum() > 30
        )

        if not has_nwp:
            return huber_pred, False

        nwp_t2m = gdf['nwp_t2m'].values.copy()
        nwp_bias = gdf['nwp_signed_bias'].fillna(0).values

        if 'nwp_debiased_error' in gdf.columns:
            nwp_error = gdf['nwp_debiased_error'].fillna(3.0).values
        elif 'nwp_local_error' in gdf.columns:
            nwp_error = gdf['nwp_local_error'].fillna(3.0).values
        else:
            # Rough fallback
            if 'temp' in gdf.columns:
                nwp_error = np.abs(nwp_t2m - gdf['temp'].values)
                nwp_error = np.where(np.isfinite(nwp_error), nwp_error, 3.0)
            else:
                nwp_error = np.full(len(gdf), 3.0)

        # Gaussian decay trust
        nwp_trust = np.exp(-(nwp_error ** 2) / (2 * self.nwp_trust_sigma ** 2))

        # Debiased HARMONIE
        nwp_debiased = nwp_t2m + nwp_bias

        # Multi-model consensus: blend HARMONIE and ICON weighted by agreement.
        if 'icon_t2m' in gdf.columns:
            icon_t2m = gdf['icon_t2m'].values.copy()
            icon_valid = np.isfinite(icon_t2m) & np.isfinite(nwp_t2m)

            if icon_valid.sum() > 30:
                # Agreement on raw temperatures
                model_diff = np.abs(nwp_t2m - icon_t2m)
                agreement = np.exp(-(model_diff / self._CONSENSUS_SCALE) ** 2)

                # Debias ICON with domain-mean correction
                if 'temp' in gdf.columns:
                    temps = gdf['temp'].values
                    bias_mask = icon_valid & np.isfinite(temps)
                    if bias_mask.sum() > 30:
                        self._icon_domain_bias = float(
                            np.nanmedian(temps[bias_mask] - icon_t2m[bias_mask]))

                icon_bias = getattr(self, '_icon_domain_bias', 0.0)
                icon_debiased = icon_t2m + icon_bias

                # Blend debiased temperatures
                nwp_debiased = np.where(
                    icon_valid,
                    agreement * nwp_debiased + (1.0 - agreement) * icon_debiased,
                    nwp_debiased
                )

        # Handle NaN in NWP
        nwp_valid = np.isfinite(nwp_debiased)
        nwp_debiased = np.where(nwp_valid, nwp_debiased, huber_pred)
        nwp_trust = np.where(nwp_valid, nwp_trust, 0.0)

        # Adaptive prior
        T_prior = nwp_trust * nwp_debiased + (1.0 - nwp_trust) * huber_pred

        return T_prior, True

    def fit(self, gdf: gpd.GeoDataFrame):
        print("\nTraining the Robust Stacking Model...")
        df_train = gdf.dropna(subset=['temp'] + self.trend_features)
        y = df_train['temp'].values
        
        # Stage 1: Macro-Trend
        print(f"   Stage 1: Macro-Trend (Huber on {self.trend_features})")
        X_trend = self._get_data(df_train, self.trend_features, self.trend_scaler, fit=True, impute=True)
        self.trend_model = HuberRegressor(epsilon=1.35, max_iter=300)
        self.trend_model.fit(X_trend, y)
        huber_pred = self.trend_model.predict(X_trend)
        resid1 = y - huber_pred
        print(f"      Base RMSE: {np.sqrt(np.mean(resid1**2)):.3f}°C")
        
        # Stage 1.5: Spatially-Gated Adaptive Prior
        if self.use_adaptive_prior:
            T_prior, self._has_adaptive_prior = self._compute_adaptive_prior(df_train, huber_pred)
            if self._has_adaptive_prior:
                resid_for_lgbm = y - T_prior
                prior_rmse = np.sqrt(np.mean(resid_for_lgbm**2))
                nwp_trust_mean = df_train.get('nwp_local_error', pd.Series([3.0]))
                if hasattr(nwp_trust_mean, 'mean'):
                    trust_vals = np.exp(-(nwp_trust_mean.fillna(3.0).values ** 2) / (2 * self.nwp_trust_sigma ** 2))
                    print(f"   Stage 1.5: Adaptive Prior (NWP trust: mean={trust_vals.mean():.3f})")
                else:
                    print(f"   Stage 1.5: Adaptive Prior")
                huber_rmse = np.sqrt(np.mean(resid1**2))
                print(f"      Prior RMSE: {prior_rmse:.3f}°C (vs Huber: {huber_rmse:.3f}°C)")
                # Safety guard
                if prior_rmse > huber_rmse * self._PRIOR_SAFETY_THRESHOLD:
                    resid_for_lgbm = resid1
                    self._has_adaptive_prior = False
                    print(f"      Prior suppressed (degradation: "
                          f"+{(prior_rmse/huber_rmse - 1)*100:.1f}%)")
            else:
                resid_for_lgbm = resid1
                print("   Stage 1.5: No NWP available, using Huber residuals")
        else:
            resid_for_lgbm = resid1
            self._has_adaptive_prior = False
        
        # Stage 2: EnvML
        lgbm_features = list(self.env_features)
        self._lgbm_features = lgbm_features  # Store for predict()
        
        print(f"   Stage 2: Regularized LightGBM on {len(lgbm_features)} env features")
        X_env = self._get_data(df_train, lgbm_features, fit=True, impute=True)
    
        # Compute sample weights
        if 'source' in df_train.columns:
            source_weight = df_train['source'].map(SOURCE_TRUST).fillna(SOURCE_TRUST_DEFAULT).values
        else:
            source_weight = np.ones(len(df_train))
        
        if 'qc_confidence' in df_train.columns:
            qc_weight = df_train['qc_confidence'].fillna(1.0).values
            sample_weight = source_weight * qc_weight
        else:
            sample_weight = source_weight
        
        # Spatial representativeness weighting
        if '_spatial_density_w' in df_train.columns:
            # Use precomputed weights
            spatial_w = df_train['_spatial_density_w'].values.copy()
            sample_weight = sample_weight * spatial_w
            n_dense = int(np.sum(spatial_w < 0.5))
            n_isolated = int(np.sum(spatial_w > 2.0))
            print(f"      Spatial density weighting (precomputed): {n_dense} dense "
                  f"(w<0.5), {n_isolated} isolated (w>2.0)")
        elif 'x_pl' in df_train.columns and 'y_pl' in df_train.columns and len(df_train) > 500:
            # Standalone fallback
            spatial_w = self._compute_density_weights(df_train)
            sample_weight = sample_weight * spatial_w
            n_dense = int(np.sum(spatial_w < 0.5))
            n_isolated = int(np.sum(spatial_w > 2.0))
            print(f"      Spatial density weighting (local): {n_dense} dense "
                  f"(w<0.5), {n_isolated} isolated (w>2.0)")

        # Nighttime consumer PWS weight attenuation
        has_ccn = ('calm_clear_night' in df_train.columns and
                   'decoupling_index' in df_train.columns and
                   'source' in df_train.columns)
        if has_ccn:
            ccn = df_train['calm_clear_night'].fillna(0).values
            di = df_train['decoupling_index'].fillna(0).values
            is_pws = (~df_train['source'].isin(['IMGW', 'EDWIN', 'TRAX'])).values.astype(float)
            
            # Continuous decay
            night_decay = 1.0 - 0.75 * ccn * np.clip(di, 0, 1) * is_pws
            sample_weight = sample_weight * night_decay
            
            n_attenuated = int(np.sum(night_decay < 0.9))
            if n_attenuated > 0:
                mean_decay = night_decay[night_decay < 0.9].mean()
                print(f"      Nighttime consumer attenuation: {n_attenuated} stations "
                      f"(mean weight factor: {mean_decay:.2f})")

        from lightgbm import LGBMRegressor
        with warnings.catch_warnings(): # silence LGBM warnings
            warnings.simplefilter("ignore")
            self.ml_model = LGBMRegressor(**self.lgbm_params)
            self.ml_model.fit(X_env, resid_for_lgbm, sample_weight=sample_weight)
        resid2 = resid_for_lgbm - self.ml_model.predict(X_env)
        current_rmse = np.sqrt(np.mean(resid2**2))
        r2_imp = 1 - (np.var(resid2) / (np.var(resid_for_lgbm) + 1e-10))
        print(f"      EnvML added R²: {r2_imp:.3f} | Remaining RMSE: {current_rmse:.3f}°C")

        # Stage 3: Kriging
        if self.use_kriging and current_rmse > 0.15 and len(df_train) > 50:
            self._fit_kriging(df_train, resid2)
        else:
            if self.use_kriging:
                print("   Stage 3: Skipped (residuals small or insufficient data)")
            self.kriging_model = None
        return self
    
    @staticmethod
    def _compute_density_weights(gdf, radius_m=15000.0):
        """Compute inverse-sqrt-density spatial weights.
        
        Args:
        - gdf: GeoDataFrame with 'x_pl' and 'y_pl' columns (EPSG:2180).
        - radius_m: Search radius in meters (default: 15km).
        
        Returns:
        - weights: array of shape (len(gdf),), mean=1.0.
        """
        coords = np.column_stack([gdf['x_pl'].values, gdf['y_pl'].values])
        tree = cKDTree(coords)
        counts = tree.query_ball_point(coords, r=radius_m, return_length=True)
        counts = np.asarray(counts, dtype=float).clip(min=1)
        w = 1.0 / np.sqrt(counts)
        w /= w.mean()
        return w
    
    def _fit_kriging(self, df_train, residuals):
        """Fit Ordinary Kriging on given residuals."""
        print(f"   Stage 3: Micro-Scale Kriging ({RESIDUAL_KRIGING_VARIOGRAM})")
        gdf_proj = df_train.to_crs(CRS_POLAND)
        
        # Store training station coords for adaptive scaling at predict time
        all_x = gdf_proj.geometry.x.values.copy()
        all_y = gdf_proj.geometry.y.values.copy()
        self.kriging_station_tree = cKDTree(np.column_stack([all_x, all_y]))
        
        # Kriging station selection

        # Step A: Trust filter
        if 'source' in df_train.columns:
            all_sources = df_train['source'].values
            all_trust = np.array([SOURCE_TRUST.get(s, SOURCE_TRUST_DEFAULT) for s in all_sources])
            if 'qc_confidence' in df_train.columns:
                all_qc = df_train['qc_confidence'].values
                all_effective = all_trust * np.clip(all_qc, 0.01, 1.0)
            else:
                all_effective = all_trust

            krig_threshold = 0.5
            trust_mask = all_effective >= krig_threshold
            n_filtered = (~trust_mask).sum()
            if n_filtered > 0:
                print(f"      Filtered {n_filtered} low-trust stations from Kriging input "
                      f"(threshold={krig_threshold:.1f})")
        else:
            trust_mask = np.ones(len(gdf_proj), dtype=bool)
        
        # Step B: Downsample trusted stations if still too many for Kriging
        trusted_indices = np.where(trust_mask)[0]
        max_kriging_stations = 3000

        if len(trusted_indices) > max_kriging_stations:
            idx = np.random.choice(trusted_indices, max_kriging_stations, replace=False)
            x_k, y_k, z_k = all_x[idx], all_y[idx], residuals[idx]
        else:
            x_k = all_x[trusted_indices]
            y_k = all_y[trusted_indices]
            z_k = residuals[trusted_indices]
        
        if len(x_k) < 30:
            print(f"      ⚠️ Too few trusted stations for Kriging ({len(x_k)}), skipping Stage 3")
            self.kriging_model = None
        else:
            try:
                x_k = x_k + np.random.uniform(-0.1, 0.1, size=x_k.shape)
                y_k = y_k + np.random.uniform(-0.1, 0.1, size=y_k.shape)
                self.kriging_model = OrdinaryKriging(
                    x_k, y_k, z_k, variogram_model=RESIDUAL_KRIGING_VARIOGRAM,
                    verbose=False, enable_plotting=False,
                    nlags=20,
                    exact_values=False
                )
                # Adaptive decay: shorter in dense station networks
                avg_spacing = np.sqrt(3.12e11 / max(len(x_k), 1))
                self._kriging_decay_dist = float(np.clip(avg_spacing, 5000.0, 20000.0))
                print(f"      ✓ Kriging fitted on {len(x_k)} stations (decay={self._kriging_decay_dist/1000:.1f}km)")
            except Exception as e:
                print(f"      ⚠️ Kriging failed: {e}, skipping Stage 3")
                self.kriging_model = None
    
    def predict(self, gdf: gpd.GeoDataFrame) -> np.ndarray:
        # Stage 1: Huber trend
        X_trend = self._get_data(gdf, self.trend_features, self.trend_scaler, fit=False, impute=True)
        huber_pred = self.trend_model.predict(X_trend)
        
        # Stage 1.5: Adaptive Prior (if fitted with NWP)
        if self._has_adaptive_prior:
            T_prior, _ = self._compute_adaptive_prior(gdf, huber_pred)
            preds = T_prior.copy()
        else:
            preds = huber_pred.copy()
        
        # Stage 2: LightGBM correction (on same features as training)
        X_env = self._get_data(gdf, self._lgbm_features, fit=False, impute=True)
        lgbm_correction = self.ml_model.predict(X_env)
        
        # NWP-quality-aware correction dampening (prediction-time only)
        if self._has_adaptive_prior and 'nwp_debiased_error' in gdf.columns:
            nwp_err = gdf['nwp_debiased_error'].fillna(3.0).values
            scale = np.ones_like(nwp_err)

            # Excellent NWP: suppress LightGBM (Prior already near-perfect)
            excellent_mask = nwp_err < self._GATE_EXCELLENT_HI
            scale[excellent_mask] = np.clip(
                (nwp_err[excellent_mask] - self._GATE_EXCELLENT_LO)
                / (self._GATE_EXCELLENT_HI - self._GATE_EXCELLENT_LO),
                0.0, 1.0
            )

            # Poor NWP: attenuate LightGBM (NWP features unreliable)
            poor_mask = nwp_err > self._GATE_POOR_LO
            scale[poor_mask] = np.clip(
                1.0 - (1.0 - self._GATE_POOR_FLOOR)
                * (nwp_err[poor_mask] - self._GATE_POOR_LO)
                / (self._GATE_POOR_HI - self._GATE_POOR_LO),
                self._GATE_POOR_FLOOR, 1.0
            )

            lgbm_correction = lgbm_correction * scale
        
        preds += lgbm_correction
        
        # Stage 3: Kriging (only when run standalone, not from ensemble)
        if self.kriging_model:
            preds = self._apply_kriging(gdf, preds)
        return preds
    
    def _apply_kriging(self, gdf, preds):
        """Apply Kriging correction to predictions."""
        gdf_proj = gdf.to_crs(CRS_POLAND)
        try:
            pred_x = gdf_proj.geometry.x.values
            pred_y = gdf_proj.geometry.y.values
            n_pts = len(pred_x)
            
            CHUNK_SIZE = 5000
            k_pred = np.zeros(n_pts)
            
            for start in range(0, n_pts, CHUNK_SIZE):
                end = min(start + CHUNK_SIZE, n_pts)
                chunk_x = pred_x[start:end]
                chunk_y = pred_y[start:end]
                try:
                    chunk_pred, _ = self.kriging_model.execute(
                        'points', chunk_x, chunk_y,
                        backend='vectorized'
                    )
                    k_pred[start:end] = chunk_pred
                except Exception:
                    try:
                        chunk_pred, _ = self.kriging_model.execute(
                            'points', chunk_x, chunk_y,
                            backend='loop'
                        )
                        k_pred[start:end] = chunk_pred
                    except Exception:
                        pass
            
            k_pred = np.clip(k_pred, -5.0, 5.0)
            
            if self.kriging_station_tree is not None:
                pred_coords = np.column_stack([pred_x, pred_y])
                dists, _ = self.kriging_station_tree.query(pred_coords, k=1)
                decay_dist = getattr(self, '_kriging_decay_dist', 20000.0)
                adaptive_scale = self.kriging_scale * np.exp(-dists / decay_dist)
            else:
                adaptive_scale = self.kriging_scale
            
            preds = preds + k_pred * adaptive_scale
        except Exception:
            pass
        return preds

    def get_feature_importance(self) -> pd.DataFrame:
        if self.ml_model is None: return pd.DataFrame()
        imp = self.ml_model.feature_importances_
        if imp.sum() > 0: imp = 100.0 * (imp / imp.sum())
        feature_names = self._lgbm_features if hasattr(self, '_lgbm_features') else self.env_features
        df = pd.DataFrame({'feature': feature_names, 'importance': imp}).sort_values('importance', ascending=False)
        
        # Add Trend features conceptually
        trend_rows = pd.DataFrame({'feature': [f'[S1] {f}' for f in self.trend_features], 'importance': [np.nan]*len(self.trend_features)})
        prior_label = '[S1.5] Adaptive Prior (NWP-gated)' if self._has_adaptive_prior else '[S1.5] Huber only'
        prior_row = pd.DataFrame({'feature': [prior_label], 'importance': [np.nan]})
        return pd.concat([trend_rows, prior_row, df]).reset_index(drop=True)

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
        self.kriging_model = None
        self.kriging_station_tree = None
        kriging_scale = self.model_kwargs.get('kriging_scale', 0.7)
        
        has_projected = ('x_pl' in gdf.columns and 'y_pl' in gdf.columns)
        if has_projected and len(gdf) > 500:
            spatial_w = PhysicsTrendEnvMLModel._compute_density_weights(gdf)
            gdf = gdf.copy()
            gdf['_spatial_density_w'] = spatial_w
            
            n_dense = int(np.sum(spatial_w < 0.5))
            n_isolated = int(np.sum(spatial_w > 2.0))
            w_min, w_max = spatial_w.min(), spatial_w.max()
            print(f"[Ensemble] Precomputed spatial density weights on {len(gdf)} stations")
            print(f"           {n_dense} dense (w<0.5), {n_isolated} isolated (w>2.0), "
                  f"range [{w_min:.2f}, {w_max:.2f}]")
        
        for i, seed in enumerate(self.seeds[:self.n_models], 1):
            current_kwargs = self.model_kwargs.copy()
            lgbm_params = current_kwargs.get('lgbm_params', LIGHTGBM_PARAMS.copy()).copy()
            lgbm_params['random_state'] = seed
            current_kwargs.pop('lgbm_params', None)
            current_kwargs.pop('kriging_scale', None)
            
            bag_frac = 0.95
            gdf_sample = gdf.sample(frac=bag_frac, random_state=seed)

            # Individual models run S1+S2 only
            model = PhysicsTrendEnvMLModel(
                self.trend_features, 
                self.env_features, 
                lgbm_params=lgbm_params,
                use_kriging=False,
                kriging_scale=kriging_scale,
                **current_kwargs
            )
            model.fit(gdf_sample)
            self.models.append(model)
        
        # Feature importance pruning
        MIN_FEATURES = 10
        PRUNE_THRESHOLD = 1.5
        MIN_DROP = 3
        
        if len(self.env_features) > MIN_FEATURES:
            imp_df = self.get_feature_importance()
            if len(imp_df) > 0:
                total = imp_df['importance'].sum()
                if total > 0:
                    imp_df['pct'] = 100.0 * imp_df['importance'] / total
                    low_imp = imp_df[imp_df['pct'] < PRUNE_THRESHOLD]['feature'].tolist()
                    
                    n_remaining = len(self.env_features) - len(low_imp)
                    if n_remaining < MIN_FEATURES:
                        low_imp = low_imp[:len(self.env_features) - MIN_FEATURES]
                    
                    if len(low_imp) >= MIN_DROP:
                        pruned_features = [f for f in self.env_features if f not in low_imp]
                        print(f"\n[Pruning] Dropping {len(low_imp)} low-importance features "
                              f"(<{PRUNE_THRESHOLD}%): {low_imp}")
                        print(f"[Pruning] Retraining with {len(pruned_features)} features "
                              f"(was {len(self.env_features)})")
                        
                        self.env_features = pruned_features
                        self.models = []
                        for i, seed in enumerate(self.seeds[:self.n_models], 1):
                            current_kwargs = self.model_kwargs.copy()
                            lgbm_params = current_kwargs.get('lgbm_params', LIGHTGBM_PARAMS.copy()).copy()
                            lgbm_params['random_state'] = seed
                            current_kwargs.pop('lgbm_params', None)
                            current_kwargs.pop('kriging_scale', None)
                            bag_frac = 0.92
                            gdf_sample = gdf.sample(frac=bag_frac, random_state=seed)
                            model = PhysicsTrendEnvMLModel(
                                self.trend_features,
                                pruned_features,
                                lgbm_params=lgbm_params,
                                use_kriging=False,
                                kriging_scale=kriging_scale,
                                **current_kwargs
                            )
                            model.fit(gdf_sample)
                            self.models.append(model)
        
        # Post-Ensemble Residual Kriging (PERK)
        if USE_RESIDUAL_KRIGING and len(gdf) > 50:
            print(f"\n[PERK] Post-Ensemble Residual Kriging")
            df_train = gdf.dropna(subset=['temp'])
            y = df_train['temp'].values
            
            # Consensus S1+S2 prediction (mean of ensemble)
            all_preds = np.array([m.predict(df_train) for m in self.models])
            consensus = np.nanmean(all_preds, axis=0)
            consensus_resid = y - consensus
            consensus_rmse = np.sqrt(np.mean(consensus_resid**2))
            print(f"[PERK] Consensus S1+S2 RMSE: {consensus_rmse:.3f}°C")
            
            if consensus_rmse > 0.15:
                # Pre-Kriging Residual Clamping
                if 'source' in df_train.columns:
                    source_trust = np.array([SOURCE_TRUST.get(s, SOURCE_TRUST_DEFAULT)
                                            for s in df_train['source'].values])
                    clamp_bounds = np.where(source_trust >= 0.9, 4.0, 2.0)
                    n_clamped = int(np.sum(np.abs(consensus_resid) > clamp_bounds))
                    if n_clamped > 0:
                        consensus_resid = np.clip(consensus_resid, -clamp_bounds, clamp_bounds)
                        print(f"[PERK] Clamped {n_clamped} extreme residuals "
                              f"(trusted: ±4.0°C, consumer: ±2.0°C)")

                # QC-confidence-scaled residuals for Kriging
                if 'qc_confidence' in df_train.columns:
                    qc_conf = df_train['qc_confidence'].fillna(0.5).values
                    qc_scale = np.sqrt(np.clip(qc_conf, 0.01, 1.0))
                    consensus_resid = consensus_resid * qc_scale
                    n_attenuated = int(np.sum(qc_scale < 0.8))
                    if n_attenuated > 0:
                        print(f"[PERK] QC-scaled {n_attenuated} low-confidence residuals "
                              f"(mean scale: {qc_scale[qc_scale < 0.8].mean():.2f})")

                # Fit Kriging once on clamped, QC-scaled consensus residuals
                kriging_model = PhysicsTrendEnvMLModel(
                    self.trend_features, self.env_features,
                    use_kriging=True, kriging_scale=kriging_scale
                )
                kriging_model._fit_kriging(df_train, consensus_resid)
                self.kriging_model = kriging_model.kriging_model
                self.kriging_station_tree = kriging_model.kriging_station_tree
                self._kriging_scale = kriging_scale
                self._kriging_decay_dist = getattr(kriging_model, '_kriging_decay_dist', 20000.0)
            else:
                print("[PERK] Consensus residuals too small, skipping Kriging")
        
        return self

    def predict_with_uncertainty(self, gdf: gpd.GeoDataFrame) -> Tuple[np.ndarray, np.ndarray]:
        # Ensemble S1+S2 predictions (no Kriging inside models)
        preds = np.array([model.predict(gdf) for model in self.models])
        
        # Mean minimizes MSE (our RMSE metric), more appropriate than median
        mean_pred = np.nanmean(preds, axis=0)
        
        # Uncertainty = Interquartile Range scaled to sigma
        q75, q25 = np.nanpercentile(preds, [75, 25], axis=0)
        uncertainty = (q75 - q25) / 1.35
        
        # Post-Ensemble Kriging correction (single Kriging on consensus residuals)
        if self.kriging_model is not None:
            gdf_proj = gdf.to_crs(CRS_POLAND)
            try:
                pred_x = gdf_proj.geometry.x.values
                pred_y = gdf_proj.geometry.y.values
                n_pts = len(pred_x)
                
                CHUNK_SIZE = 5000
                k_pred = np.zeros(n_pts)
                k_var = np.zeros(n_pts)
                
                for start in range(0, n_pts, CHUNK_SIZE):
                    end = min(start + CHUNK_SIZE, n_pts)
                    chunk_x = pred_x[start:end]
                    chunk_y = pred_y[start:end]
                    try:
                        chunk_pred, chunk_variance = self.kriging_model.execute(
                            'points', chunk_x, chunk_y,
                            backend='vectorized'
                        )
                        k_pred[start:end] = chunk_pred
                        k_var[start:end] = chunk_variance
                    except Exception:
                        try:
                            chunk_pred, chunk_variance = self.kriging_model.execute(
                                'points', chunk_x, chunk_y,
                                backend='loop'
                            )
                            k_pred[start:end] = chunk_pred
                            k_var[start:end] = chunk_variance
                        except Exception:
                            pass
                
                k_pred = np.clip(k_pred, -5.0, 5.0)
                
                # Variance-Based Attenuation
                try:
                    psill = self.kriging_model.variogram_model_parameters[0]
                    nugget = self.kriging_model.variogram_model_parameters[2]
                    total_sill = psill + nugget
                    if psill > 1e-6:
                        variance_scale = np.clip(
                            (total_sill - k_var) / psill, 0.0, 1.0
                        )
                        adaptive_scale = self._kriging_scale * variance_scale
                    else:
                        # Pure nugget model
                        adaptive_scale = self._kriging_scale * 0.5
                except Exception:
                    # Fallback
                    adaptive_scale = self._kriging_scale
                
                # Regime-Aware Trust-Gated PERK
                PERK_MIN_RETENTION = 0.4
                if 'nwp_trust' in gdf.columns:
                    trust = gdf['nwp_trust'].fillna(0.0).values
                    di = gdf['decoupling_index'].fillna(0.0).values if 'decoupling_index' in gdf.columns else 0.0
                    effective_trust = trust * (1.0 - di)
                    regime_factor = PERK_MIN_RETENTION + (1.0 - PERK_MIN_RETENTION) * (1.0 - effective_trust)
                    adaptive_scale = adaptive_scale * regime_factor
                
                mean_pred += k_pred * adaptive_scale
            except Exception:
                pass  # Fallback to S1+S2 only
        
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