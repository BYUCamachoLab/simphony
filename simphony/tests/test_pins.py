# Copyright © Simphony Project Contributors
# Licensed under the terms of the MIT License
# (see simphony/__init__.py for details)

import pytest

from simphony import Model
from simphony.pins import Pin, PinList


@pytest.fixture
def component():
    class SomeModel(Model):
        pin_count = 2

    return SomeModel()


@pytest.fixture
def pin1(component):
    return Pin(component, "pin1")


@pytest.fixture
def pin2(component):
    return Pin(component, "pin2")


@pytest.fixture
def pinlist(component):
    return PinList(component, 2)


class TestPins:
    def test_rename(self, pin1):
        pin1.rename("test")
        assert pin1.name == "test"

    def test_connection(self, pin1, pin2):
        pin1.connect(pin2)
        assert pin1._isconnected()
        assert pin2._isconnected()
        assert pin1._connection == pin2

    def test_disconnect(self, pin1, pin2):
        pin1.connect(pin2)
        assert pin1._connection == pin2

        pin1.disconnect()
        assert not pin1._isconnected()
        assert not pin2._isconnected()

    def test_access(self, pinlist):
        pin1 = pinlist[0]
        pin2 = pinlist[1]

        assert pin1.name == "pin1"
        assert pin2.name == "pin2"

        pinlist.rename("a", "b")

        assert pin1.name == "a"
        assert pin2.name == "b"

        assert pin1 == pinlist["a"]
        assert pin2 == pinlist["b"]
