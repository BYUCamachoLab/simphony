
class DuplicateModelError(AttributeError):
    def __init__(self, component_type):
        """Error for a duplicate component type.

        Parameters
        ----------
        component_type : UniqueModelName
            The object containing the name of the model that is a duplicate
        errors : *
            Any errors corresponding to this error.
        """
        super().__init__("Component with name \'" + component_type + "\' already exists. Can you rename the component?")