import pytest
import ipm

@pytest.fixture
def matcher():
    m = ipm.Matcher()
    m.add_pattern(path='res/logo_drpepper.jpg', name='Dr Pepper')
    m.add_pattern(path='res/logo_pepsi.jpg', name='Pepsi')
    return m


def test_pepsi(matcher):
    patterns = matcher.match(path='res/can_pepsi.jpg')
    assert patterns[0].name == 'Pepsi'

def test_drpepper(matcher):
    patterns = matcher.match(path='res/can_drpepper.jpg')
    assert patterns[0].name == 'Dr Pepper'

def test_drpepper_cherry(matcher):
    patterns = matcher.match(path='res/can_drpepper_cherry.jpg')
    assert patterns[0].name == 'Dr Pepper'