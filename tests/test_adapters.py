from gabion.pebble.adapters import load_adapter


def test_load_builtin_adapter() -> None:
    adapter = load_adapter("gabion.user_models.linear:LinearAdapter")
    assert adapter.__class__.__name__ == "LinearAdapter"
