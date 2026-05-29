from point_classifier.data import generate_dataset, get_features_target
from point_classifier.model import train_model


def test_train_model_returns_fitted_gradient_boosting_model() -> None:
    dataset = generate_dataset(n_samples=50, radius=1.0, shift=1.5, random_state=42)
    features, target = get_features_target(dataset)

    model = train_model(features, target, random_state=42)

    assert hasattr(model, "estimators_")
    assert len(model.predict(features.head(5))) == 5
