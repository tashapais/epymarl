from pathlib import Path
import numpy as np
import yaml

from smacv2.env.starcraft2.wrapper import StarCraftCapabilityEnvWrapper

from .multiagentenv import MultiAgentEnv


SMACv2_CONFIG_DIR = Path(__file__).parent.parent / "config" / "envs" / "smacv2_configs"


def get_scenario_names():
    return [p.name for p in SMACv2_CONFIG_DIR.iterdir()]


def load_scenario(map_name, **kwargs):
    scenario_path = SMACv2_CONFIG_DIR / f"{map_name}.yaml"
    with open(scenario_path, "r") as f:
        scenario_args = yaml.load(f, Loader=yaml.FullLoader)
    scenario_args.update(kwargs)
    return StarCraftCapabilityEnvWrapper(**scenario_args["env_args"])


class SMACv2Wrapper(MultiAgentEnv):
    def __init__(self, map_name, seed, common_reward=True, reward_scalarisation="sum", **kwargs):
        # common_reward=False -> individual rewards: redistribute the SMAC team
        # reward across agents by per-agent damage share (attribution granularity
        # is varied while the team-level signal is conserved). The common_reward=True
        # path is unchanged (returns the original scalar team reward).
        self.common_reward = common_reward
        self.reward_scalarisation = reward_scalarisation
        # Ablation: obs_mask_unit_type=True zeros each agent's OWN unit-type one-hot
        # in the observation, so role identity is not handed to the encoder and must
        # be inferred from dynamics / the reward signal. SMACv2 draws unit types
        # i.i.d. per slot, so an agent's own type is independent of its allies' (no
        # leak by elimination), and ally/enemy features are left intact to preserve
        # competence. The own-type bits are the final `unit_type_bits` entries of each
        # agent's obs (validated below; no trailing timestep feature in this config).
        self.obs_mask_unit_type = bool(kwargs.pop("obs_mask_unit_type", False))
        self.env = load_scenario(map_name, seed=seed, **kwargs)
        self.episode_limit = self.env.episode_limit
        env_info = self.env.get_env_info()
        self.n_agents = env_info["n_agents"]
        self.n_actions = env_info["n_actions"]
        self._sc = self._find_sc_env()
        n_enemies = getattr(self._sc, "n_enemies", self.n_agents)
        # action layout: [no-op, stop, move x4, attack-enemy x n_enemies]
        self.n_no_attack = self.n_actions - n_enemies
        self._unit_type_bits = int(getattr(self._sc, "unit_type_bits", 0))
        if self.obs_mask_unit_type:
            assert self._unit_type_bits > 0, "obs_mask_unit_type=True but unit_type_bits==0"
            assert not getattr(self._sc, "obs_timestep_number", False), \
                "obs_timestep_number=True: own unit-type is not the final obs feature"
            self.env.reset()
            tail = np.asarray(self.env.get_obs())[:, -self._unit_type_bits:]
            assert np.allclose(tail.sum(axis=1), 1.0), \
                f"own unit-type tail not one-hot (sums={tail.sum(axis=1)}); obs layout assumption wrong"

    def _find_sc_env(self):
        """Locate the underlying StarCraft2Env (the object exposing `enemies`)."""
        o = self.env
        for _ in range(4):
            if hasattr(o, "enemies"):
                return o
            o = getattr(o, "env", None)
            if o is None:
                break
        return self.env

    def _enemy_hp(self):
        enemies = getattr(self._sc, "enemies", {})
        return {eid: (u.health + getattr(u, "shield", 0.0)) for eid, u in enemies.items()}

    def _distribute_reward(self, team_reward, actions, pre_hp, post_hp):
        """Split team_reward across agents by per-agent damage share. Conserves the
        team total: sum(per_agent) == team_reward."""
        n = self.n_agents
        acts = [int(a) for a in (actions.tolist() if hasattr(actions, "tolist") else actions)]
        attackers = {}
        for a_id, act in enumerate(acts):
            if act >= self.n_no_attack:
                attackers.setdefault(act - self.n_no_attack, []).append(a_id)
        dmg_i = np.zeros(n, dtype=np.float64)
        for eid, hp0 in pre_hp.items():
            dmg = max(0.0, hp0 - post_hp.get(eid, 0.0))
            atk = attackers.get(eid, [])
            if dmg > 0.0 and atk:
                for a in atk:
                    dmg_i[a] += dmg / len(atk)
        if dmg_i.sum() > 0.0:
            per = float(team_reward) * dmg_i / dmg_i.sum()
        else:
            # no attributable damage (e.g. pure shaping/penalty step) -> equal split
            per = np.full(n, float(team_reward) / n, dtype=np.float64)
        return per.astype(np.float32)

    def step(self, actions):
        """Returns obss, reward, terminated, truncated, info"""
        if not self.common_reward:
            pre_hp = self._enemy_hp()
        rews, terminated, info = self.env.step(actions)
        if not self.common_reward:
            rews = self._distribute_reward(rews, actions, pre_hp, self._enemy_hp())
        obss = self.get_obs()
        truncated = False
        return obss, rews, terminated, truncated, info

    def _mask_obs_row(self, o):
        o = np.array(o, dtype=np.float32, copy=True)
        if self.obs_mask_unit_type and self._unit_type_bits > 0:
            o[-self._unit_type_bits:] = 0.0  # zero own unit-type one-hot (dead agents already 0)
        return o

    def get_obs(self):
        """Returns all agent observations in a list"""
        obss = self.env.get_obs()
        if self.obs_mask_unit_type and self._unit_type_bits > 0:
            return [self._mask_obs_row(o) for o in obss]
        return obss

    def get_obs_agent(self, agent_id):
        """Returns observation for agent_id"""
        o = self.env.get_obs_agent(agent_id)
        if self.obs_mask_unit_type and self._unit_type_bits > 0:
            return self._mask_obs_row(o)
        return o

    def get_obs_size(self):
        """Returns the shape of the observation"""
        return self.env.get_obs_size()

    def get_state(self):
        return self.env.get_state()

    def get_state_size(self):
        """Returns the shape of the state"""
        return self.env.get_state_size()

    def get_avail_actions(self):
        return self.env.get_avail_actions()

    def get_avail_agent_actions(self, agent_id):
        """Returns the available actions for agent_id"""
        return self.env.get_avail_agent_actions(agent_id)

    def get_total_actions(self):
        """Returns the total number of actions an agent could ever take"""
        return self.env.get_total_actions()

    def reset(self, seed=None, options=None):
        """Returns initial observations and info"""
        if seed is not None:
            self.env.seed(seed)
        self.env.reset()
        return self.get_obs(), {}

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()

    def seed(self, seed=None):
        self.env.seed(seed)

    def save_replay(self):
        self.env.save_replay()

    def get_env_info(self):
        return self.env.get_env_info()

    def get_stats(self):
        return self.env.get_stats()


if __name__ == "__main__":
    for scenario in get_scenario_names():
        env = load_scenario(scenario)
        env_info = env.get_env_info()
        # print name of config, number of agents, state shape, observation shape, action shape
        print(
            scenario,
            env_info["n_agents"],
            env_info["state_shape"],
            env_info["obs_shape"],
            env_info["n_actions"],
        )
        print()
