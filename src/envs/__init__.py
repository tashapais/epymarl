import os
import sys

import warnings

from .multiagentenv import MultiAgentEnv
from .gymma import GymmaWrapper

try:
    from .smaclite_wrapper import SMACliteWrapper
except (ImportError, OSError) as e:
    SMACliteWrapper = None
    warnings.warn(
        f"SMAClite is not available ({e}), so these environments will not be available!"
    )


if sys.platform == "linux":
    os.environ.setdefault(
        "SC2PATH", os.path.join(os.getcwd(), "3rdparty", "StarCraftII")
    )


def __check_and_prepare_smac_kwargs(kwargs):
    assert "common_reward" in kwargs and "reward_scalarisation" in kwargs
    assert kwargs[
        "common_reward"
    ], "SMAC only supports common reward. Please set `common_reward=True` or choose a different environment that supports general sum rewards."
    del kwargs["common_reward"]
    del kwargs["reward_scalarisation"]
    assert "map_name" in kwargs, "Please specify the map_name in the env_args"
    return kwargs


def smaclite_fn(**kwargs) -> MultiAgentEnv:
    kwargs = __check_and_prepare_smac_kwargs(kwargs)
    return SMACliteWrapper(**kwargs)


def gymma_fn(**kwargs) -> MultiAgentEnv:
    assert "common_reward" in kwargs and "reward_scalarisation" in kwargs
    return GymmaWrapper(**kwargs)


REGISTRY = {}
if SMACliteWrapper is not None:
    REGISTRY["smaclite"] = smaclite_fn
REGISTRY["gymma"] = gymma_fn


# registering both smac and smacv2 causes a pysc2 error
# --> dynamically register the needed env
def register_smac():
    from .smac_wrapper import SMACWrapper

    def smac_fn(**kwargs) -> MultiAgentEnv:
        kwargs = __check_and_prepare_smac_kwargs(kwargs)
        return SMACWrapper(**kwargs)

    REGISTRY["sc2"] = smac_fn


def register_smacv2():
    from .smacv2_wrapper import SMACv2Wrapper

    def smacv2_fn(**kwargs) -> MultiAgentEnv:
        # sc2v2 supports general (per-agent) rewards: keep common_reward and
        # reward_scalarisation and pass them through to the wrapper, which returns
        # a scalar team reward when common_reward=True and a per-agent vector
        # (damage-share redistribution of the team reward) when False.
        assert "common_reward" in kwargs and "reward_scalarisation" in kwargs
        assert "map_name" in kwargs, "Please specify the map_name in the env_args"
        return SMACv2Wrapper(**kwargs)

    REGISTRY["sc2v2"] = smacv2_fn
