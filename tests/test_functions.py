from app import confidence_band, recommendation_for


def test_u1_high_confidence():
    assert confidence_band(0.90) == "High"


def test_u2_medium_confidence():
    assert confidence_band(0.70) == "Medium"


def test_u3_low_confidence():
    assert confidence_band(0.50) == "Low"


def test_u4_high_confidence_second_value():
    assert confidence_band(0.95) == "High"


def test_u5_medium_confidence_second_value():
    assert confidence_band(0.75) == "Medium"


def test_u6_low_confidence_second_value():
    assert confidence_band(0.40) == "Low"


def test_u7_underutilised_high():
    assert recommendation_for(True, "High") == "Shutdown"


def test_u8_underutilised_medium():
    assert recommendation_for(True, "Medium") == "Resize"


def test_u9_underutilised_low():
    assert recommendation_for(True, "Low") == "No Action"


def test_u10_not_underutilised_high():
    assert recommendation_for(False, "High") == "No Action"