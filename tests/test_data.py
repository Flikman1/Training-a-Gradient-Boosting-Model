from point_classifier.data import FEATURE_COLUMNS, TARGET_COLUMN, generate_dataset, get_features_target


def test_generate_dataset_has_expected_number_of_rows() -> None:
    dataset = generate_dataset(n_samples=25, radius=1.0, shift=1.5, random_state=42)
    assert len(dataset) == 50


def test_generate_dataset_targets_are_binary() -> None:
    dataset = generate_dataset(n_samples=30, radius=1.0, shift=1.5, random_state=42)
    assert set(dataset[TARGET_COLUMN].unique()) == {0, 1}


def test_get_features_target_returns_expected_shape() -> None:
    dataset = generate_dataset(n_samples=10, radius=1.0, shift=1.5, random_state=42)
    features, target = get_features_target(dataset)

    assert list(features.columns) == FEATURE_COLUMNS
    assert features.shape == (20, 2)
    assert target.shape == (20,)
