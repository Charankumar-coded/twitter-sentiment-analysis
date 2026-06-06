import main
from predict import predict


def test_clean_text_basic():
    cleaned = main.clean_text("I LOVE this! https://example.com")
    assert cleaned == "i love this"


def test_prediction_returns_label():
    main.main()
    result = predict("I love this amazing product and it works great!")
    assert result.startswith("positive") or result.startswith("negative")
