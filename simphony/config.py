"""loads a configuration from 3 files, high priority overwrites low priority:

1. A config.yml found in the current working directory (high priority)
2. ~/.config/simphony.yml specific for the machine
3. the default config is in this file (lowest priority)
"""

__all__ = ["CONFIG"]

import logging
import pathlib

import hiyapyco

default_config = """
key: value
"""

home = pathlib.Path.home()
cwd = pathlib.Path.cwd()
cwd_config = cwd / "config.yml"

home_config = home / ".config" / "simphony.yml"
module_path = pathlib.Path(__file__).parent.absolute()
repo_path = module_path.parent

CONFIG = hiyapyco.load(
    str(default_config),
    str(home_config),
    str(cwd_config),
    failonmissingfiles=False,
    loglevelmissingfiles=logging.DEBUG,
)
CONFIG["module_path"] = module_path
CONFIG["repo_path"] = repo_path
CONFIG["workspace"] = pathlib.Path(CONFIG.get("workspace", repo_path / "workspace"))

if __name__ == "__main__":
    print(CONFIG["workspace"])
