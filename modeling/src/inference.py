"""
Inference script for Business Risk Prediction Model
Loads trained model and scaler to make predictions on new business data.

Usage:
    from inference import RiskPredictor
    
    predictor = RiskPredictor()
    result = predictor.predict({
        "rating_x_reviews": 150.5,
        "review_count": 50,
        ...
    })
    
    print(result["risk_score"])  # 0.73
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Union, Optional
import warnings
from huggingface_hub import hf_hub_download

warnings.filterwarnings('ignore')


class RiskPredictor:
    """
    Production-ready risk prediction class.
    Handles model loading, feature validation, and predictions.
    """
    
    def __init__(
        self,
        model_path: str = "../models/xgboost_model.pkl",
        scaler_path: str = "../../data_pipeline/artifacts/scaler.pkl",
        features_path: str = "features.txt",
        auto_download: bool = True,
        hf_repo_id: str = "PatZhang0214/business-risk-prediction-dataset"
    ):
        """
        Initialize the risk predictor.
        
        Args:
            model_path: Path to trained XGBoost model
            scaler_path: Path to fitted StandardScaler
            features_path: Path to features.txt file
            auto_download: If True, download model from HuggingFace if not found locally
            hf_repo_id: HuggingFace repository ID for model download
        """
        self.model_path = Path(model_path)
        self.scaler_path = Path(scaler_path)
        self.features_path = Path(features_path)
        self.auto_download = auto_download
        self.hf_repo_id = hf_repo_id
        
        # Load components
        self.features = self._load_features()
        self.model = self._load_model()
        self.scaler = self._load_scaler()
        
        print(f"✅ RiskPredictor initialized with {len(self.features)} features")
    
    def _load_features(self) -> List[str]:
        """Load feature names from features.txt"""
        if not self.features_path.exists():
            raise FileNotFoundError(
                f"Features file not found: {self.features_path}\n"
                f"Run train.py first to generate features.txt"
            )
        
        with open(self.features_path, 'r') as f:
            features = [line.strip() for line in f if line.strip()]
        
        print(f"📄 Loaded {len(features)} features from {self.features_path}")
        return features
    
    def _load_model(self):
        """Load trained model, with optional auto-download from HuggingFace"""
        if not self.model_path.exists():
            if self.auto_download:
                print(f"⚠️  Model not found at {self.model_path}")
                print(f"📥 Attempting to download from HuggingFace: {self.hf_repo_id}")
                try:
                    local_path = hf_hub_download(
                        repo_id=self.hf_repo_id,
                        filename="models/xgboost_model.pkl",
                        repo_type="dataset"
                    )
                    model = joblib.load(local_path)
                    print(f"✅ Model downloaded and loaded from HuggingFace")
                    return model
                except Exception as e:
                    raise FileNotFoundError(
                        f"Failed to download model from HuggingFace: {e}\n"
                        f"Please train a model first with train.py"
                    )
            else:
                raise FileNotFoundError(
                    f"Model not found at {self.model_path}\n"
                    f"Please train a model first with train.py or set auto_download=True"
                )
        
        model = joblib.load(self.model_path)
        print(f"📦 Model loaded from {self.model_path}")
        return model
    
    def _load_scaler(self):
        """Load fitted scaler, with optional auto-download"""
        if not self.scaler_path.exists():
            if self.auto_download:
                print(f"⚠️  Scaler not found at {self.scaler_path}")
                print(f"📥 Attempting to download from HuggingFace: {self.hf_repo_id}")
                try:
                    local_path = hf_hub_download(
                        repo_id=self.hf_repo_id,
                        filename="artifacts/scaler.pkl",
                        repo_type="dataset"
                    )
                    scaler = joblib.load(local_path)
                    print(f"✅ Scaler downloaded and loaded from HuggingFace")
                    return scaler
                except Exception as e:
                    raise FileNotFoundError(
                        f"Failed to download scaler from HuggingFace: {e}\n"
                        f"Please run data preprocessing pipeline first"
                    )
            else:
                raise FileNotFoundError(
                    f"Scaler not found at {self.scaler_path}\n"
                    f"Please run data preprocessing pipeline first or set auto_download=True"
                )
        
        scaler = joblib.load(self.scaler_path)
        print(f"📊 Scaler loaded from {self.scaler_path}")
        return scaler
    
    def _validate_input(self, data: Dict[str, float]) -> None:
        """
        Validate input data has all required features.
        
        Args:
            data: Dictionary with feature values
            
        Raises:
            ValueError: If required features are missing
        """
        missing_features = set(self.features) - set(data.keys())
        
        if missing_features:
            raise ValueError(
                f"Missing required features: {missing_features}\n"
                f"Required features: {self.features}"
            )
        
        # Check for None/NaN values
        invalid_features = {k: v for k, v in data.items() if v is None or (isinstance(v, float) and np.isnan(v))}
        if invalid_features:
            raise ValueError(
                f"Features cannot be None or NaN: {list(invalid_features.keys())}"
            )
    
    def _prepare_input(self, data: Dict[str, float]) -> np.ndarray:
        """
        Prepare input data for model prediction.
        
        Args:
            data: Dictionary with feature values
            
        Returns:
            Scaled feature array ready for prediction
        """
        # Extract features in correct order
        feature_values = [data[feature] for feature in self.features]
        
        # Convert to numpy array and reshape
        X = np.array(feature_values).reshape(1, -1)
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        return X_scaled
    
    def predict(self, data: Dict[str, float]) -> Dict[str, Union[float, bool, str]]:
        """
        Predict business risk for a single business.
        
        Args:
            data: Dictionary containing all required features
            
        Returns:
            Dictionary with prediction results:
                - risk_score: Probability business stays open (0-1)
                - will_close: Predicted outcome (True if will close)
                - closure_probability: Probability business will close (0-1)
                - confidence: Confidence level (low/medium/high)
                
        Example:
            >>> predictor = RiskPredictor()
            >>> result = predictor.predict({
            ...     "rating_x_reviews": 150.5,
            ...     "review_count": 50,
            ...     ...
            ... })
            >>> print(result)
            {
                'risk_score': 0.73,
                'will_close': False,
                'closure_probability': 0.27,
                'confidence': 'high'
            }
        """
        # Validate input
        self._validate_input(data)
        
        # Prepare features
        X_scaled = self._prepare_input(data)
        
        # Get predictions
        prediction = self.model.predict(X_scaled)[0]  # 0 or 1
        probabilities = self.model.predict_proba(X_scaled)[0]  # [prob_closed, prob_open]
        
        prob_open = probabilities[1]
        prob_close = probabilities[0]
        
        # Determine confidence level
        confidence_score = max(prob_open, prob_close)
        if confidence_score >= 0.8:
            confidence = "high"
        elif confidence_score >= 0.6:
            confidence = "medium"
        else:
            confidence = "low"
        
        return {
            "risk_score": float(prob_open),
            "will_close": bool(prediction == 0),
            "closure_probability": float(prob_close),
            "confidence": confidence
        }
    
    def predict_batch(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Predict business risk for multiple businesses.
        
        Args:
            data: DataFrame containing all required features
            
        Returns:
            DataFrame with original data plus prediction columns
            
        Example:
            >>> df = pd.DataFrame([
            ...     {"rating_x_reviews": 150.5, "review_count": 50, ...},
            ...     {"rating_x_reviews": 200.0, "review_count": 75, ...}
            ... ])
            >>> results = predictor.predict_batch(df)
        """
        # Validate all rows have required features
        missing_cols = set(self.features) - set(data.columns)
        if missing_cols:
            raise ValueError(
                f"DataFrame missing required columns: {missing_cols}"
            )
        
        # Extract and scale features
        X = data[self.features].values
        X_scaled = self.scaler.transform(X)
        
        # Get predictions
        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)
        
        # Create results DataFrame
        results = data.copy()
        results['risk_score'] = probabilities[:, 1]
        results['will_close'] = predictions == 0
        results['closure_probability'] = probabilities[:, 0]
        
        # Add confidence
        max_probs = np.max(probabilities, axis=1)
        results['confidence'] = pd.cut(
            max_probs,
            bins=[0, 0.6, 0.8, 1.0],
            labels=['low', 'medium', 'high']
        )
        
        return results
    
    def get_feature_importance(self, top_n: int = 10) -> pd.DataFrame:
        """
        Get feature importance from the model.
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            DataFrame with features and their importance scores
        """
        if not hasattr(self.model, 'feature_importances_'):
            raise AttributeError("Model does not have feature_importances_")
        
        importance_df = pd.DataFrame({
            'feature': self.features,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False).head(top_n)
        
        return importance_df
    
    def explain_prediction(self, data: Dict[str, float], top_n: int = 5) -> Dict:
        """
        Explain a prediction by showing top contributing features.
        
        Args:
            data: Business data
            top_n: Number of top features to show
            
        Returns:
            Dictionary with prediction and top features
        """
        result = self.predict(data)
        top_features = self.get_feature_importance(top_n)
        
        # Get actual values for top features
        feature_values = {
            feature: data[feature]
            for feature in top_features['feature'].values
        }
        
        return {
            **result,
            'top_features': feature_values,
            'feature_importance': top_features.to_dict('records')
        }


def main():
    """Example usage of RiskPredictor"""
    
    # Initialize predictor
    predictor = RiskPredictor()
    
    # Example business data (you'll need to provide actual values for all features)
    example_business = {
        "rating_x_reviews": 150.5,
        "review_count": 50,
        "num_categories": 3,
        "years_in_business": 5.2,
        "num_checkins": 100,
        "has_checkin": 1,
        "pcpi": 0.5,  # Assuming these are already scaled
        "poverty_rate": -0.3,
        "median_household_income": 0.8,
        "unemployment_rate": -0.2,
        "avg_weekly_wages": 0.6,
        "latitude": 40.7128,
        "longitude": -74.0060,
        # Add all category features...
        "cat_Restaurants": 1,
        "cat_Food": 0,
        # ... (you need all 60+ features from features.txt)
    }
    
    # Make prediction
    print("\n" + "="*60)
    print("BUSINESS RISK PREDICTION")
    print("="*60)
    
    try:
        result = predictor.predict(example_business)
        
        print(f"\nRisk Score (Stay Open): {result['risk_score']:.2%}")
        print(f"Closure Probability:    {result['closure_probability']:.2%}")
        print(f"Prediction:             {'WILL CLOSE' if result['will_close'] else 'WILL STAY OPEN'}")
        print(f"Confidence:             {result['confidence'].upper()}")
        
        # Show top features
        print("\n" + "="*60)
        print("TOP FEATURE IMPORTANCE")
        print("="*60)
        top_features = predictor.get_feature_importance(5)
        print(top_features.to_string(index=False))
        
    except ValueError as e:
        print(f"\n❌ Error: {e}")
        print("\nPlease provide all required features from features.txt")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    main()