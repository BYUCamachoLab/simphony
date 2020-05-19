# -*- coding: utf-8 -*-
#
# Copyright © Simphony Project Contributors
# Licensed under the terms of the MIT License
# (see simphony/__init__.py for details)

import pytest

import simphony.library.ebeam as ebeam
from simphony.netlist import Pin, PinList


class TestPin:
    class TestCreate:
        def test_noargs(self):
            p1 = Pin(None, None)
            with pytest.raises(AttributeError):
                p1.element
            with pytest.raises(AttributeError):
                p1.index

        def test_args(self):
            name = "n1"
            p1 = Pin(None, name)
            assert p1.name == name

    def test_rename(self):
        name = "oldname"
        p1 = Pin(None, name)
        name = "newname"
        p1.name = name
        assert p1.name == name


class TestPinlist:
    class TestInstantiate:
        def test_create_with_strings(self):
            pinnames = ["n1", "n2", "n3"]
            pinlist = PinList(None, *pinnames)
            assert len(pinlist) == 3

            name = iter(pinnames)
            for pin in pinlist:
                assert pin.pinlist is pinlist
                assert pin.name == next(name)

        def test_create_with_pins(self):
            length = 4
            pins = [Pin(None, "n" + str(i)) for i in range(length)]
            pinlist = PinList(None, *pins)
            assert len(pinlist) == length

            npin = iter(pins)
            for pin in pinlist:
                assert pin.pinlist is pinlist
                assert pin is next(npin)

        def test_create_with_mixed_args(self):
            scrambled = ["n1", "n2", Pin(None, "n3"), Pin(None, "n4"), "n5"]
            pinlist = PinList(None, *scrambled)
            assert len(pinlist) == 5

            for pin in pinlist:
                assert pin.pinlist is pinlist

    class TestGet:
        def setup_method(self):
            self.length = 4
            self.pins = [Pin(None, "n" + str(i)) for i in range(self.length)]
            self.pinlist = PinList(None, *self.pins)

        def test_get_with_int(self):
            for i in range(self.length):
                assert self.pinlist[i] == self.pins[i]
                assert self.pinlist[i] is self.pins[i]

        def test_get_with_str(self):
            for i in range(self.length):
                name = self.pins[i].name
                assert self.pinlist[name] == self.pins[i]
                assert self.pinlist[name] is self.pins[i]

        def test_get_with_object(self):
            for i in range(self.length):
                pin = self.pins[i]
                assert self.pinlist[pin] == self.pins[i]
                assert self.pinlist[pin] is pin

    class TestSet:
        # pinlist.pins = ('out', 'in', 'mix')
        # pinlist.pins = ('n1')
        pass

    class TestSwitchPin:
        pass

    class TestOperators:
        def setup_method(self):
            self.length = 8
            self.pins = [Pin(None, "n" + str(i)) for i in range(self.length)]
            self.pinlist1 = PinList(None, *self.pins[: int(self.length / 2)])
            self.pinlist2 = PinList(None, *self.pins[int(self.length / 2) :])

        def test_add(self):
            self.pinlist_new = self.pinlist1 + self.pinlist2
            assert self.pinlist_new is not self.pinlist1
            assert self.pinlist_new is not self.pinlist2

            piter = iter(self.pins)
            for pin in self.pinlist_new:
                assert pin is next(piter)

        def test_add_empty_to_containing(self):
            p1 = PinList(None)
            p2 = self.pinlist1
            pinlist_new = p1 + p2
            for pin in pinlist_new:
                assert pin.pinlist is pinlist_new

        def test_remove(self):
            self.test_add()
            self.pinlist_new.remove("n1")
            assert self.pinlist_new["n7"].index == 6
            self.pinlist_new.remove("n4")
            assert self.pinlist_new["n7"].index == 5
