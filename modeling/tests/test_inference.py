"""
Tests for inference.py
Run with: pytest test_inference.py -v
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path to import inference
sys.path.insert(0, str(Path(__file__).parent))

# from src.inference import RiskPredictor
from inference import RiskPredictor


@pytest.fixture
def predictor():
    """Create a RiskPredictor instance for testing"""
    return RiskPredictor(auto_download=False)


@pytest.fixture
def sample_business(predictor):
    """Create sample business data with all required features"""
    # Load features from features.txt
    with open('features.txt', 'r') as f:
        features = [line.strip() for line in f if line.strip()]
    
    # Create dummy data for all features
    data = {}
    for feature in features:
        if feature.startswith('cat_'):
            data[feature] = np.random.choice([0, 1])  # Binary for categories
        elif feature == 'has_checkin':
            data[feature] = np.random.choice([0, 1])
        else:
            data[feature] = np.random.uniform(-2, 2)  # Scaled numeric features
    
    return data


class TestRiskPredictor:
    """Test suite for RiskPredictor class"""
    
    def test_initialization(self, predictor):
        """Test that predictor initializes correctly"""
        assert predictor.model is not None
        assert predictor.scaler is not None
        assert len(predictor.features) > 0
    
    def test_features_loaded(self, predictor):
        """Test that features are loaded from features.txt"""
        assert isinstance(predictor.features, list)
        assert len(predictor.features) > 0
        # Should have numeric + category features
        assert 'rating_x_reviews' in predictor.features or 'rating' in predictor.features
    
    def test_model_loaded(self, predictor):
        """Test that model has predict methods"""
        assert hasattr(predictor.model, 'predict')
        assert hasattr(predictor.model, 'predict_proba')
    
    def test_scaler_loaded(self, predictor):
        """Test that scaler has transform method"""
        assert hasattr(predictor.scaler, 'transform')
    
    def test_predict_single(self, predictor, sample_business):
        """Test single prediction"""
        result = predictor.predict(sample_business)
        
        # Check result structure
        assert 'risk_score' in result
        assert 'will_close' in result
        assert 'closure_probability' in result
        assert 'confidence' in result
        
        # Check value types
        assert isinstance(result['risk_score'], float)
        assert isinstance(result['will_close'], bool)
        assert isinstance(result['closure_probability'], float)
        assert isinstance(result['confidence'], str)
        
        # Check value ranges
        assert 0 <= result['risk_score'] <= 1
        assert 0 <= result['closure_probability'] <= 1
        assert result['confidence'] in ['low', 'medium', 'high']
        
        # Check probabilities sum to 1
        assert abs(result['risk_score'] + result['closure_probability'] - 1.0) < 0.01
    
    def test_predict_validation_missing_features(self, predictor, sample_business):
        """Test that missing features raise ValueError"""
        incomplete_data = {k: v for k, v in list(sample_business.items())[:5]}
        
        with pytest.raises(ValueError, match="Missing required features"):
            predictor.predict(incomplete_data)
    
    def test_predict_validation_none_values(self, predictor, sample_business):
        """Test that None values raise ValueError"""
        sample_business[list(sample_business.keys())[0]] = None
        
        with pytest.raises(ValueError, match="cannot be None"):
            predictor.predict(sample_business)
    
    def test_predict_batch(self, predictor, sample_business):
        """Test batch prediction"""
        # Create DataFrame with multiple businesses
        df = pd.DataFrame([sample_business] * 5)
        
        results = predictor.predict_batch(df)
        
        # Check output structure
        assert len(results) == 5
        assert 'risk_score' in results.columns
        assert 'will_close' in results.columns
        assert 'closure_probability' in results.columns
        assert 'confidence' in results.columns
        
        # Check all predictions are valid
        assert all(0 <= score <= 1 for score in results['risk_score'])
        assert all(conf in ['low', 'medium', 'high'] for conf in results['confidence'])
    
    def test_get_feature_importance(self, predictor):
        """Test feature importance extraction"""
        importance_df = predictor.get_feature_importance(top_n=5)
        
        assert len(importance_df) == 5
        assert 'feature' in importance_df.columns
        assert 'importance' in importance_df.columns
        assert all(importance_df['importance'] >= 0)
    
    def test_explain_prediction(self, predictor, sample_business):
        """Test prediction explanation"""
        explanation = predictor.explain_prediction(sample_business, top_n=3)
        
        # Should include prediction results
        assert 'risk_score' in explanation
        assert 'will_close' in explanation
        
        # Should include feature information
        assert 'top_features' in explanation
        assert 'feature_importance' in explanation
        assert len(explanation['top_features']) <= 3
    
    def test_prediction_consistency(self, predictor, sample_business):
        """Test that same input gives same prediction"""
        result1 = predictor.predict(sample_business)
        result2 = predictor.predict(sample_business)
        
        assert result1['risk_score'] == result2['risk_score']
        assert result1['will_close'] == result2['will_close']
    
    def test_confidence_levels(self, predictor, sample_business):
        """Test that confidence levels are assigned correctly"""
        result = predictor.predict(sample_business)
        
        max_prob = max(result['risk_score'], result['closure_probability'])
        
        if max_prob >= 0.8:
            assert result['confidence'] == 'high'
        elif max_prob >= 0.6:
            assert result['confidence'] == 'medium'
        else:
            assert result['confidence'] == 'low'


def test_features_file_exists():
    """Test that features.txt exists"""
    assert Path('features.txt').exists(), "features.txt not found. Run train.py first."


def test_model_file_exists():
    """Test that model file exists"""
    model_path = Path('models/xgboost_model.pkl')
    if not model_path.exists():
        pytest.skip("Model not found. Run train.py first or enable auto_download.")


if __name__ == "__main__":
    pytest.main([__file__, '-v'])