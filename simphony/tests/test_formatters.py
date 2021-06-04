# Copyright © Simphony Project Contributors
# Licensed under the terms of the MIT License
# (see simphony/__init__.py for details)

import filecmp
import os

import numpy as np
import pytest

from simphony.formatters import (
    CircuitJSONFormatter,
    CircuitSiEPICFormatter,
    ModelJSONFormatter,
)
from simphony.layout import Circuit
from simphony.libraries import siepic
from simphony.models import Model
from simphony.simulators import SweepSimulator
from simphony.tools import wl2freq

waveguide_150_json = '{"freqs": [{"h": "0x1.54d316b4f0200p+47"}, {"h": "0x1.5549cc339de7ap+47"}, {"h": "0x1.55c081b24baf4p+47"}, {"h": "0x1.56373730f976ep+47"}, {"h": "0x1.56adecafa73e8p+47"}, {"h": "0x1.5724a22e55061p+47"}, {"h": "0x1.579b57ad02cdbp+47"}, {"h": "0x1.58120d2bb0955p+47"}, {"h": "0x1.5888c2aa5e5cfp+47"}, {"h": "0x1.58ff78290c249p+47"}, {"h": "0x1.59762da7b9ec3p+47"}, {"h": "0x1.59ece32667b3dp+47"}, {"h": "0x1.5a6398a5157b7p+47"}, {"h": "0x1.5ada4e23c3431p+47"}, {"h": "0x1.5b5103a2710abp+47"}, {"h": "0x1.5bc7b9211ed24p+47"}, {"h": "0x1.5c3e6e9fcc99ep+47"}, {"h": "0x1.5cb5241e7a618p+47"}, {"h": "0x1.5d2bd99d28292p+47"}, {"h": "0x1.5da28f1bd5f0cp+47"}, {"h": "0x1.5e19449a83b86p+47"}, {"h": "0x1.5e8ffa1931800p+47"}, {"h": "0x1.5f06af97df47ap+47"}, {"h": "0x1.5f7d65168d0f4p+47"}, {"h": "0x1.5ff41a953ad6ep+47"}, {"h": "0x1.606ad013e89e7p+47"}, {"h": "0x1.60e1859296661p+47"}, {"h": "0x1.61583b11442dbp+47"}, {"h": "0x1.61cef08ff1f55p+47"}, {"h": "0x1.6245a60e9fbcfp+47"}, {"h": "0x1.62bc5b8d4d849p+47"}, {"h": "0x1.6333110bfb4c3p+47"}, {"h": "0x1.63a9c68aa913dp+47"}, {"h": "0x1.64207c0956db7p+47"}, {"h": "0x1.6497318804a31p+47"}, {"h": "0x1.650de706b26aap+47"}, {"h": "0x1.65849c8560324p+47"}, {"h": "0x1.65fb52040df9ep+47"}, {"h": "0x1.66720782bbc18p+47"}, {"h": "0x1.66e8bd0169892p+47"}, {"h": "0x1.675f72801750cp+47"}, {"h": "0x1.67d627fec5186p+47"}, {"h": "0x1.684cdd7d72e00p+47"}, {"h": "0x1.68c392fc20a7ap+47"}, {"h": "0x1.693a487ace6f4p+47"}, {"h": "0x1.69b0fdf97c36dp+47"}, {"h": "0x1.6a27b37829fe7p+47"}, {"h": "0x1.6a9e68f6d7c61p+47"}, {"h": "0x1.6b151e75858dbp+47"}, {"h": "0x1.6b8bd3f433555p+47"}], "name": "Waveguide component", "pins": ["pin1", "pin2"], "s_params": [[[{"r": {"h": "0x0.0p+0"}, "i": {"h": "0x0.0p+0"}}, {"r": {"h": "0x1.f6c07bd5fe8b2p-1"}, "i": {"h": "-0x1.bf0b90de909acp-4"}}], [{"r": {"h": "0x1.f6c07bd5fe8b2p-1"}, "i": {"h": "-0x1.bf0b90de909acp-4"}}, {"r": {"h": "0x0.0p+0"}, "i": {"h": "0x0.0p+0"}}]], [[{"r": {"h": "0x0.0p+0"}, "i": {"h": "0x0.0p+0"}}, {"r": {"h": "-0x1.f6c441420b3b2p-1"}, "i": {"h": "-0x1.bdfbcee6f7debp-4"}}], [{"r": {"h": "-0x1.f6c441420b3b2p-1"}, "i": {"h": "-0x1.bdfbcee6f7debp-4"}}, {"r": {"h": "0x0.0p+0"}, "i": {"h": "0x0.0p+0"}}]], [[{"r": {"h": "0x0.0p+0"}, "i": {"h": "0x0.0p+0"}}, {"r": {"h": "0x1.de53855cb3b47p-1"}, "i": {"h": "0x1.492839c102a89p-2"}}], [{"r": {"h": "0x1.de53855cb3b47p-1"}, "i": {"h": "0x1.492839c102a89p-2"}}, {"r": {"h": "0x0.0p+0"}, "i": {"h": "0x0.0p+0"}}]], [[{"r": {"h": "0x0.0p+0"}, "i": {"h": "0x0.0p+0"}}, {"r": {"h": "-0x1.aeaccb4bb680bp-1"}, "i": {"h": "-0x1.09550ab1223aap-1"}}], [{"r": {"h": "-0x1.aeaccb4bb680bp-1"}, "i": {"h": "-0x1.09550ab1223aap-1"}}, {"r": {"h": "0x0.0p+0"}, "i": {"h": "0x0.0p+0"}}]], [[{"r": {"h": "0x0.0p+0"}, "i": {"h": "0x0.0p+0"}}, {"r": {"h": "0x1.6a2d81bb53fa5p-1"}, "i": {"h": "0x1.6124640231784p-1"}}], [{"r": {"h": "0x1.6a2d81bb53fa5p-1"}, "i": {"h": "0x1.6124640231784p-1"}}, {"r": {"h": "0x0.0p+0"}, "i": {"h": "0x0.0p+0"}}]], [[{"r": {"h": "0x0.0p+0"}, "i": {"h": "0x0.0p+0"}}, {"r": {"h": "-0x1.14331f5224bebp-1"}, "i": {"h": "-0x1.a7c998f4251d6p-1"}}], [{"r": {"h": "-0x1.14331f5224bebp-1"}, "i": {"h": "-0x1.a7c998f4251d6p-1"}}, {"r": {"h": "0x0.0p+0"}, "i": {"h": "0x0.0p+0"}}]], [[{"r": {"h": "0x0.0p+0"}, "i": {"h": "0x0.0p+0"}}, {"r": {"h": "0x1.61e029324f845p-2"}, "i": {"h": "0x1.d9e4c13d6ce81p-1"}}], [{"r": {"h": "0x1.61e029324f845p-2"}, "i": {"h": "0x1.d9e4c13d6ce81p-1"}}, {"r": {"h": "0x0.0p+0"}, "i": {"h": "0x0.0p+0"}}]], [[{"r": {"h": "0x0.0p+0"}, "i": {"h": "0x0.0p+0"}}, {"r": {"h": "-0x1.14db0a1a50f28p-3"}, "i": {"h": "-0x1.f517218bb016ap-1"}}], [{"r": {"h": "-0x1.14db0a1a50f28p-3"}, "i": {"h": "-0x1.f517218bb016ap-1"}}, {"r": {"h": "0x0.0p+0"}, "i": {"h": "0x0.0p+0"}}]], [[{"r": {"h": "0x0.0p+0"}, "i": {"h": "0x0.0p+0"}}, {"r": {"h": "-0x1.4e172c2a889e6p-4"}, "i": {"h": "0x1.f81f019a77deap-1"}}], [{"r": {"h": "-0x1.4e172c2a889e6p-4"}, "i": {"h": "0x1.f81f019a77deap-1"}}, {"r": {"h": "0x0.0p+0"}, "i": {"h": "0x0.0p+0"}}]], [[{"r": {"h": "0x0.0p+0"}, "i": {"h": "0x0.0p+0"}}, {"r": {"h": "0x1.2d483fc0a6b2ap-2"}, "i": {"h": "-0x1.e2e597c305a6bp-1"}}], [{"r": {"h": "0x1.2d483fc0a6b2ap-2"}, "i": {"h": "-0x1.e2e597c305a6bp-1"}}, {"r": {"h": "0x0.0p+0"}, "i": {"h": "0x0.0p+0"}}]], [[{"r": {"h": "0x0.0p+0"}, "i": {"h": "0x0.0p+0"}}, {"r": {"h": "-0x1.f86698a83d0f2p-2"}, "i": {"h": "0x1.b67e7d5853bb1p-1"}}], [{"r": {"h": "-0x1.f86698a83d0f2p-2"}, "i": {"h": "0x1.b67e7d5853bb1p-1"}}, {"r": {"h": "0x0.0p+0"}, "i": {"h": "0x0.0p+0"}}]], [[{"r": {"h": "0x0.0p+0"}, "i": {"h": "0x0.0p+0"}}, {"r": {"h": "0x1.5595ee361e42ap-1"}, "i": {"h": "-0x1.7518d2682850bp-1"}}], [{"r": {"h": "0x1.5595ee361e42ap-1"}, "i": {"h": "-0x1.7518d2682850bp-1"}}, {"r": {"h": "0x0.0p+0"}, "i": {"h": "0x0.0p+0"}}]], [[{"r": {"h": "0x0.0p+0"}, "i": {"h": "0x0.0p+0"}}, {"r": {"h": "-0x1.9e8ba9e9af1c1p-1"}, "i": {"h": "0x1.21e2e6cc21938p-1"}}], [{"r": {"h": "-0x1.9e8ba9e9af1c1p-1"}, "i": {"h": "0x1.21e2e6cc21938p-1"}}, {"r": {"h": "0x0.0p+0"}, "i": {"h": "0x0.0p+0"}}]], [[{"r": {"h": "0x0.0p+0"}, "i": {"h": "0x0.0p+0"}}, {"r": {"h": "0x1.d3a178373a3f0p-1"}, "i": {"h": "-0x1.81c3c2f3ea6c9p-2"}}], [{"r": {"h": "0x1.d3a178373a3f0p-1"}, "i": {"h": "-0x1.81c3c2f3ea6c9p-2"}}, {"r": {"h": "0x0.0p+0"}, "i": {"h": "0x0.0p+0"}}]], [[{"r": {"h": "0x0.0p+0"}, "i": {"h": "0x0.0p+0"}}, {"r": {"h": "-0x1.f25aae741728bp-1"}, "i": {"h": "0x1.5afdb857bbbf4p-3"}}], [{"r": {"h": "-0x1.f25aae741728bp-1"}, "i": {"h": "0x1.5afdb857bbbf4p-3"}}, {"r": {"h": "0x0.0p+0"}, "i": {"h": "0x0.0p+0"}}]], [[{"r": {"h": "0x0.0p+0"}, "i": {"h": "0x0.0p+0"}}, {"r": {"h": "0x1.f94dfce7f01c7p-1"}, "i": {"h": "0x1.77012a4140b84p-5"}}], [{"r": {"h": "0x1.f94dfce7f01c7p-1"}, "i": {"h": "0x1.77012a4140b84p-5"}}, {"r": {"h": "0x0.0p+0"}, "i": {"h": "0x0.0p+0"}}]], [[{"r": {"h": "0x0.0p+0"}, "i": {"h": "0x0.0p+0"}}, {"r": {"h": "-0x1.e8351c94f3f1cp-1"}, "i": {"h": "-0x1.08d80c66d2f85p-2"}}], [{"r": {"h": "-0x1.e8351c94f3f1cp-1"}, "i": {"h": "-0x1.08d80c66d2f85p-2"}}, {"r": {"h": "0x0.0p+0"}, "i": {"h": "0x0.0p+0"}}]], [[{"r": {"h": "0x0.0p+0"}, "i": {"h": "0x0.0p+0"}}, {"r": {"h": "0x1.bfee8b061347cp-1"}, "i": {"h": "0x1.d60f78df74dcbp-2"}}], [{"r": {"h": "0x1.bfee8b061347cp-1"}, "i": {"h": "0x1.d60f78df74dcbp-2"}}, {"r": {"h": "0x0.0p+0"}, "i": {"h": "0x0.0p+0"}}]], [[{"r": {"h": "0x0.0p+0"}, "i": {"h": "0x0.0p+0"}}, {"r": {"h": "-0x1.82715dfc8210dp-1"}, "i": {"h": "-0x1.46692d338d128p-1"}}], [{"r": {"h": "-0x1.82715dfc8210dp-1"}, "i": {"h": "-0x1.46692d338d128p-1"}}, {"r": {"h": "0x0.0p+0"}, "i": {"h": "0x0.0p+0"}}]], [[{"r": {"h": "0x0.0p+0"}, "i": {"h": "0x0.0p+0"}}, {"r": {"h": "0x1.32b3e18f584aep-1"}, "i": {"h": "0x1.9243595a2c1e1p-1"}}], [{"r": {"h": "0x1.32b3e18f584aep-1"}, "i": {"h": "0x1.9243595a2c1e1p-1"}}, {"r": {"h": "0x0.0p+0"}, "i": {"h": "0x0.0p+0"}}]], [[{"r": {"h": "0x0.0p+0"}, "i": {"h": "0x0.0p+0"}}, {"r": {"h": "-0x1.a90ca32d5f661p-2"}, "i": {"h": "-0x1.cb0995790a3a5p-1"}}], [{"r": {"h": "-0x1.a90ca32d5f661p-2"}, "i": {"h": "-0x1.cb0995790a3a5p-1"}}, {"r": {"h": "0x0.0p+0"}, "i": {"h": "0x0.0p+0"}}]], [[{"r": {"h": "0x0.0p+0"}, "i": {"h": "0x0.0p+0"}}, {"r": {"h": "0x1.b18e2ed4c0fcdp-3"}, "i": {"h": "0x1.ee1961a5f7d42p-1"}}], [{"r": {"h": "0x1.b18e2ed4c0fcdp-3"}, "i": {"h": "0x1.ee1961a5f7d42p-1"}}, {"r": {"h": "0x0.0p+0"}, "i": {"h": "0x0.0p+0"}}]], [[{"r": {"h": "0x0.0p+0"}, "i": {"h": "0x0.0p+0"}}, {"r": {"h": "0x1.8be8ccb47405dp-10"}, "i": {"h": "-0x1.f9d8e9f938c41p-1"}}], [{"r": {"h": "0x1.8be8ccb47405dp-10"}, "i": {"h": "-0x1.f9d8e9f938c41p-1"}}, {"r": {"h": "0x0.0p+0"}, "i": {"h": "0x0.0p+0"}}]], [[{"r": {"h": "0x0.0p+0"}, "i": {"h": "0x0.0p+0"}}, {"r": {"h": "-0x1.b740980ee5296p-3"}, "i": {"h": "0x1.edc8d9416273cp-1"}}], [{"r": {"h": "-0x1.b740980ee5296p-3"}, "i": {"h": "0x1.edc8d9416273cp-1"}}, {"r": {"h": "0x0.0p+0"}, "i": {"h": "0x0.0p+0"}}]], [[{"r": {"h": "0x0.0p+0"}, "i": {"h": "0x0.0p+0"}}, {"r": {"h": "0x1.ab376694351acp-2"}, "i": {"h": "-0x1.ca88c397a9af1p-1"}}], [{"r": {"h": "0x1.ab376694351acp-2"}, "i": {"h": "-0x1.ca88c397a9af1p-1"}}, {"r": {"h": "0x0.0p+0"}, "i": {"h": "0x0.0p+0"}}]], [[{"r": {"h": "0x0.0p+0"}, "i": {"h": "0x0.0p+0"}}, {"r": {"h": "-0x1.334d76fca2abap-1"}, "i": {"h": "0x1.91ce11a406923p-1"}}], [{"r": {"h": "-0x1.334d76fca2abap-1"}, "i": {"h": "0x1.91ce11a406923p-1"}}, {"r": {"h": "0x0.0p+0"}, "i": {"h": "0x0.0p+0"}}]], [[{"r": {"h": "0x0.0p+0"}, "i": {"h": "0x0.0p+0"}}, {"r": {"h": "0x1.828857d3329eap-1"}, "i": {"h": "-0x1.464df7a8bb445p-1"}}], [{"r": {"h": "0x1.828857d3329eap-1"}, "i": {"h": "-0x1.464df7a8bb445p-1"}}, {"r": {"h": "0x0.0p+0"}, "i": {"h": "0x0.0p+0"}}]], [[{"r": {"h": "0x0.0p+0"}, "i": {"h": "0x0.0p+0"}}, {"r": {"h": "-0x1.bfa0d45c37bdfp-1"}, "i": {"h": "0x1.d7373a88274c0p-2"}}], [{"r": {"h": "-0x1.bfa0d45c37bdfp-1"}, "i": {"h": "0x1.d7373a88274c0p-2"}}, {"r": {"h": "0x0.0p+0"}, "i": {"h": "0x0.0p+0"}}]], [[{"r": {"h": "0x0.0p+0"}, "i": {"h": "0x0.0p+0"}}, {"r": {"h": "0x1.e7c7e4e8b5063p-1"}, "i": {"h": "-0x1.0bf84a4e72506p-2"}}], [{"r": {"h": "0x1.e7c7e4e8b5063p-1"}, "i": {"h": "-0x1.0bf84a4e72506p-2"}}, {"r": {"h": "0x0.0p+0"}, "i": {"h": "0x0.0p+0"}}]], [[{"r": {"h": "0x0.0p+0"}, "i": {"h": "0x0.0p+0"}}, {"r": {"h": "-0x1.f92b46bf09f71p-1"}, "i": {"h": "0x1.a32b888fe3861p-5"}}], [{"r": {"h": "-0x1.f92b46bf09f71p-1"}, "i": {"h": "0x1.a32b888fe3861p-5"}}, {"r": {"h": "0x0.0p+0"}, "i": {"h": "0x0.0p+0"}}]], [[{"r": {"h": "0x0.0p+0"}, "i": {"h": "0x0.0p+0"}}, {"r": {"h": "0x1.f309c737dfcdfp-1"}, "i": {"h": "0x1.4ae3b125a9781p-3"}}], [{"r": {"h": "0x1.f309c737dfcdfp-1"}, "i": {"h": "0x1.4ae3b125a9781p-3"}}, {"r": {"h": "0x0.0p+0"}, "i": {"h": "0x0.0p+0"}}]], [[{"r": {"h": "0x0.0p+0"}, "i": {"h": "0x0.0p+0"}}, {"r": {"h": "-0x1.d5bab81134dc8p-1"}, "i": {"h": "-0x1.776d67f9e1dbcp-2"}}], [{"r": {"h": "-0x1.d5bab81134dc8p-1"}, "i": {"h": "-0x1.776d67f9e1dbcp-2"}}, {"r": {"h": "0x0.0p+0"}, "i": {"h": "0x0.0p+0"}}]], [[{"r": {"h": "0x0.0p+0"}, "i": {"h": "0x0.0p+0"}}, {"r": {"h": "0x1.a2a851275ac88p-1"}, "i": {"h": "0x1.1bea5eda25bbep-1"}}], [{"r": {"h": "0x1.a2a851275ac88p-1"}, "i": {"h": "0x1.1bea5eda25bbep-1"}}, {"r": {"h": "0x0.0p+0"}, "i": {"h": "0x0.0p+0"}}]], [[{"r": {"h": "0x0.0p+0"}, "i": {"h": "0x0.0p+0"}}, {"r": {"h": "-0x1.5c3d5f9551b50p-1"}, "i": {"h": "-0x1.6ee4dbe77ac19p-1"}}], [{"r": {"h": "-0x1.5c3d5f9551b50p-1"}, "i": {"h": "-0x1.6ee4dbe77ac19p-1"}}, {"r": {"h": "0x0.0p+0"}, "i": {"h": "0x0.0p+0"}}]], [[{"r": {"h": "0x0.0p+0"}, "i": {"h": "0x0.0p+0"}}, {"r": {"h": "0x1.05c737c104f51p-1"}, "i": {"h": "0x1.b0d82b7861e1ep-1"}}], [{"r": {"h": "0x1.05c737c104f51p-1"}, "i": {"h": "0x1.b0d82b7861e1ep-1"}}, {"r": {"h": "0x0.0p+0"}, "i": {"h": "0x0.0p+0"}}]], [[{"r": {"h": "0x0.0p+0"}, "i": {"h": "0x0.0p+0"}}, {"r": {"h": "-0x1.469ad06df2a21p-2"}, "i": {"h": "-0x1.dec371b5bdd17p-1"}}], [{"r": {"h": "-0x1.469ad06df2a21p-2"}, "i": {"h": "-0x1.dec371b5bdd17p-1"}}, {"r": {"h": "0x0.0p+0"}, "i": {"h": "0x0.0p+0"}}]], [[{"r": {"h": "0x0.0p+0"}, "i": {"h": "0x0.0p+0"}}, {"r": {"h": "0x1.cb0902a2c7759p-4"}, "i": {"h": "0x1.f69542378614cp-1"}}], [{"r": {"h": "0x1.cb0902a2c7759p-4"}, "i": {"h": "0x1.f69542378614cp-1"}}, {"r": {"h": "0x0.0p+0"}, "i": {"h": "0x0.0p+0"}}]], [[{"r": {"h": "0x0.0p+0"}, "i": {"h": "0x0.0p+0"}}, {"r": {"h": "0x1.98cb2c724c23bp-4"}, "i": {"h": "-0x1.f742a654c4786p-1"}}], [{"r": {"h": "0x1.98cb2c724c23bp-4"}, "i": {"h": "-0x1.f742a654c4786p-1"}}, {"r": {"h": "0x0.0p+0"}, "i": {"h": "0x0.0p+0"}}]], [[{"r": {"h": "0x0.0p+0"}, "i": {"h": "0x0.0p+0"}}, {"r": {"h": "-0x1.3a49ccc8bb342p-2"}, "i": {"h": "0x1.e0d1e88479f4cp-1"}}], [{"r": {"h": "-0x1.3a49ccc8bb342p-2"}, "i": {"h": "0x1.e0d1e88479f4cp-1"}}, {"r": {"h": "0x0.0p+0"}, "i": {"h": "0x0.0p+0"}}]], [[{"r": {"h": "0x0.0p+0"}, "i": {"h": "0x0.0p+0"}}, {"r": {"h": "0x1.ffcba753e771ep-2"}, "i": {"h": "-0x1.b458c185fee23p-1"}}], [{"r": {"h": "0x1.ffcba753e771ep-2"}, "i": {"h": "-0x1.b458c185fee23p-1"}}, {"r": {"h": "0x0.0p+0"}, "i": {"h": "0x0.0p+0"}}]], [[{"r": {"h": "0x0.0p+0"}, "i": {"h": "0x0.0p+0"}}, {"r": {"h": "-0x1.56db18b2d9005p-1"}, "i": {"h": "0x1.73ee1930a6549p-1"}}], [{"r": {"h": "-0x1.56db18b2d9005p-1"}, "i": {"h": "0x1.73ee1930a6549p-1"}}, {"r": {"h": "0x0.0p+0"}, "i": {"h": "0x0.0p+0"}}]], [[{"r": {"h": "0x0.0p+0"}, "i": {"h": "0x0.0p+0"}}, {"r": {"h": "0x1.9e12536d27f84p-1"}, "i": {"h": "-0x1.22901de06d365p-1"}}], [{"r": {"h": "0x1.9e12536d27f84p-1"}, "i": {"h": "-0x1.22901de06d365p-1"}}, {"r": {"h": "0x0.0p+0"}, "i": {"h": "0x0.0p+0"}}]], [[{"r": {"h": "0x0.0p+0"}, "i": {"h": "0x0.0p+0"}}, {"r": {"h": "-0x1.d2552815e0876p-1"}, "i": {"h": "0x1.87fff7c6208c5p-2"}}], [{"r": {"h": "-0x1.d2552815e0876p-1"}, "i": {"h": "0x1.87fff7c6208c5p-2"}}, {"r": {"h": "0x0.0p+0"}, "i": {"h": "0x0.0p+0"}}]], [[{"r": {"h": "0x0.0p+0"}, "i": {"h": "0x0.0p+0"}}, {"r": {"h": "0x1.f14dae2e2f7e2p-1"}, "i": {"h": "-0x1.7253d039600ffp-3"}}], [{"r": {"h": "0x1.f14dae2e2f7e2p-1"}, "i": {"h": "-0x1.7253d039600ffp-3"}}, {"r": {"h": "0x0.0p+0"}, "i": {"h": "0x0.0p+0"}}]], [[{"r": {"h": "0x0.0p+0"}, "i": {"h": "0x0.0p+0"}}, {"r": {"h": "-0x1.f9a0622f6a72fp-1"}, "i": {"h": "-0x1.dee1ebd9a8f82p-6"}}], [{"r": {"h": "-0x1.f9a0622f6a72fp-1"}, "i": {"h": "-0x1.dee1ebd9a8f82p-6"}}, {"r": {"h": "0x0.0p+0"}, "i": {"h": "0x0.0p+0"}}]], [[{"r": {"h": "0x0.0p+0"}, "i": {"h": "0x0.0p+0"}}, {"r": {"h": "0x1.eafa85c43aa4fp-1"}, "i": {"h": "0x1.e6fc0959fdc0ap-3"}}], [{"r": {"h": "0x1.eafa85c43aa4fp-1"}, "i": {"h": "0x1.e6fc0959fdc0ap-3"}}, {"r": {"h": "0x0.0p+0"}, "i": {"h": "0x0.0p+0"}}]], [[{"r": {"h": "0x0.0p+0"}, "i": {"h": "0x0.0p+0"}}, {"r": {"h": "-0x1.c61461b494b22p-1"}, "i": {"h": "-0x1.bdd798e90397bp-2"}}], [{"r": {"h": "-0x1.c61461b494b22p-1"}, "i": {"h": "-0x1.bdd798e90397bp-2"}}, {"r": {"h": "0x0.0p+0"}, "i": {"h": "0x0.0p+0"}}]], [[{"r": {"h": "0x0.0p+0"}, "i": {"h": "0x0.0p+0"}}, {"r": {"h": "0x1.8ca76ad4c6d1bp-1"}, "i": {"h": "0x1.39ec5287b8789p-1"}}], [{"r": {"h": "0x1.8ca76ad4c6d1bp-1"}, "i": {"h": "0x1.39ec5287b8789p-1"}}, {"r": {"h": "0x0.0p+0"}, "i": {"h": "0x0.0p+0"}}]], [[{"r": {"h": "0x0.0p+0"}, "i": {"h": "0x0.0p+0"}}, {"r": {"h": "-0x1.4158d7a43dc04p-1"}, "i": {"h": "-0x1.86a9ed7905291p-1"}}], [{"r": {"h": "-0x1.4158d7a43dc04p-1"}, "i": {"h": "-0x1.86a9ed7905291p-1"}}, {"r": {"h": "0x0.0p+0"}, "i": {"h": "0x0.0p+0"}}]], [[{"r": {"h": "0x0.0p+0"}, "i": {"h": "0x0.0p+0"}}, {"r": {"h": "0x1.cf337115ed30fp-2"}, "i": {"h": "0x1.c1b6f7f951a1ep-1"}}], [{"r": {"h": "0x1.cf337115ed30fp-2"}, "i": {"h": "0x1.c1b6f7f951a1ep-1"}}, {"r": {"h": "0x0.0p+0"}, "i": {"h": "0x0.0p+0"}}]]], "subcircuit": null}'


@pytest.fixture(scope="module")
def freqs():
    return np.linspace(wl2freq(1600e-9), wl2freq(1500e-9))


@pytest.fixture(scope="class")
def mzi():
    gc_input = siepic.GratingCoupler()
    y_splitter = siepic.YBranch()
    wg_long = siepic.Waveguide(length=150e-6)
    wg_short = siepic.Waveguide(length=50e-6)
    y_recombiner = siepic.YBranch()
    gc_output = siepic.GratingCoupler()

    y_splitter.multiconnect(gc_input, wg_long, wg_short)
    y_recombiner.multiconnect(gc_output, wg_short, wg_long)

    return y_splitter.circuit


@pytest.fixture(scope="class")
def mzi4(freqs):
    y1 = siepic.YBranch()
    gc1 = siepic.GratingCoupler()
    wg1 = siepic.Waveguide(length=67.730e-6)
    wg2 = siepic.Waveguide(length=297.394e-6)
    y2 = siepic.YBranch()
    gc2 = siepic.GratingCoupler()
    wg3 = siepic.Waveguide(length=256.152e-6)
    simulator = SweepSimulator(freqs[0], freqs[-1], len(freqs))

    y1.rename_pins("N$0", "N$2", "N$1")
    gc1.rename_pins("ebeam_gc_te1550_detector2", "N$0")
    wg1.rename_pins("N$1", "N$4")
    wg2.rename_pins("N$2", "N$5")
    y2.rename_pins("N$6", "N$5", "N$4")
    gc2.rename_pins("ebeam_gc_te1550_laser1", "N$3")
    wg3.rename_pins("N$6", "N$3")

    y1.multiconnect(gc1["N$0"], wg2, wg1)
    y2.multiconnect(wg3, wg2, wg1)
    wg3.connect(gc2["N$3"])

    simulator.multiconnect(gc2, gc1)

    return simulator.circuit


@pytest.fixture(scope="class")
def waveguide():
    return siepic.Waveguide(length=150e-6)


class TestModelJSONFormatter:
    def test_format(self, freqs, waveguide):
        assert waveguide_150_json == waveguide.to_string(
            freqs, formatter=ModelJSONFormatter()
        )

    def test_parse(self, freqs, waveguide):
        waveguide2 = Model.from_string(waveguide_150_json)
        assert np.allclose(
            waveguide.s_parameters(freqs), waveguide2.s_parameters(freqs)
        )


class TestCircuitJSONFormatter:
    def test_format(self, freqs, mzi):
        json = os.path.join(os.path.dirname(__file__), "mzi.json")
        temp = os.path.join(os.path.dirname(__file__), "mzi.temp.json")

        mzi.to_file(temp, freqs, formatter=CircuitJSONFormatter())

        assert filecmp.cmp(json, temp)
        os.unlink(temp)

    def test_parse(self, freqs, mzi):
        json = os.path.join(os.path.dirname(__file__), "mzi.json")
        mzi2 = Circuit.from_file(json, formatter=CircuitJSONFormatter())

        assert np.allclose(mzi.s_parameters(freqs), mzi2.s_parameters(freqs))


class TestCircuitSiEPICFormatter:
    def test_parse(self, freqs, mzi4):
        spi = os.path.join(
            os.path.dirname(__file__),
            "..",
            "plugins",
            "siepic",
            "tests",
            "spice",
            "MZI4",
            "MZI4_main.spi",
        )

        mzi42 = Circuit.from_file(spi, formatter=CircuitSiEPICFormatter())

        assert np.allclose(mzi4.s_parameters(freqs), mzi42.s_parameters(freqs))
