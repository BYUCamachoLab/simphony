import pytest


from simphony.models import Model


class TestModelDeclaration:
    def test_missing_sparams(self):
        with pytest.raises(TypeError):
            class BadModel(Model):
                pass

    def test_missing_onames(self):
        with pytest.raises(TypeError):
            class BadModel(Model):
                def s_params(self, wl):
                    pass

    def test_missing_ocount(self):
        with pytest.raises(TypeError):
            class BadModel(Model):
                def s_params(self, wl):
                    pass

    def test_ocount_and_onames(self):
        with pytest.raises(TypeError):
            class BadModel(Model):
                ocount = 2
                onames = ["o0", "o1"]

                def s_params(self, wl):
                    pass


class TestModelContextAccessibility:
    def test_model(self):
        pass


class TestModelPins:
    def test_model(self):
        pass

    def test_get_oport_by_name(self):
        pass

    def test_get_eport_by_name(self):
        pass

    def test_get_oport_by_index(self):
        pass

    def test_get_eport_by_index(self):
        pass

    def test_get_next_unconnected_oport(self):
        pass

    def test_get_next_unconnected_eport(self):
        pass

    def test_get_next_unconnected_oport_all_taken(self):
        pass

    def test_get_next_unconnected_eport_all_taken(self):
        pass

    def test_duplicate_oport_name(self):
        pass

    def test_duplicate_eport_name(self):
        pass


class TestModelCaching:
    def test_model_instance_attributes_constant(self):
        pass

    def test_model_instance_attributes_variable(self):
        pass
