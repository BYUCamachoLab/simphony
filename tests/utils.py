from sax import SDict  # type: ignore
from typing import get_origin


def is_sdict(obj: object) -> bool:
    # Check base object type of SDict
    SDict_base_type = get_origin(SDict)
    if SDict_base_type is not None:
        return isinstance(obj, SDict_base_type)
    return isinstance(obj, type(SDict))
