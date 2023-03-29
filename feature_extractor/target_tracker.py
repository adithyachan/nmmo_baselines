from team_helper import TeamHelper


class TargetTracker():
  def __init__(self, team_helper: TeamHelper) -> None:
    self._team_helper = team_helper

    self._target_entity_id = None
    self.target_entity_pop = None

  def reset(self, init_obs):
    self._target_entity_id = [None] * self.TEAM_SIZE
    self.target_entity_pop = [None] * self.TEAM_SIZE

  def update(self, obs):
      pass
