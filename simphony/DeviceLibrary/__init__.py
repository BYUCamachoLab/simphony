
models = {}

def register_component_model(cls):
    models[cls.__name__] = cls
