import simphony
import pytest

def func(x):
    return x + 1

# def test_answer():
#     assert func(3) == 5

class TestClass(object):
    def test_one(self):
        x = "this"
        assert 'h' in x

    # def test_two(self):
    #     x = "hello"
    #     assert hasattr(x, 'check')

def f():
    raise SystemExit(1)

def test_mytest():
    with pytest.raises(SystemExit):
        f()