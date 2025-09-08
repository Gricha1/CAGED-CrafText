
from craftext.environment.scenarious.checkers.target_state import Achievements
from craftext.environment.craftext_constants import Achievement, Scenarios, AchievementState, BlockType
# from craftext.environment.scenarious.checkers.target_state_cmdp import StepOnBlock, CMDPTargetState
from craftext.environment.scenarious.checkers.target_state_cmdp_budget_drink_level import CMDPTargetState, DrinkLevelState


def create_target_state(required=[], forbidden=[], level:int=0):
    base_vector = [AchievementState.NOT_MATTER for i in range(Achievement.MAKE_IRON_SWORD + 1)]
    for i in range(len(base_vector)):
        if i in required:
            base_vector[i] = AchievementState.NEED_TO_ACHIEVE
        elif i in forbidden:
            base_vector[i] = AchievementState.AVOID_TO_ACHIEVE
    target_achievements = Achievements(achievement_mask=tuple(base_vector))
    # step_on_block = StepOnBlock(block_type=block_type)
    level = DrinkLevelState(level=level)
    return CMDPTargetState(achievements=target_achievements, drink_level_state=level)

easy = { 
  "DEFEAT_ZOMBIE": {
      "instruction": "Defeat a zombie.",
      "instruction_paraphrases": [
          "Eliminate an undead creature lurking nearby.",
          "Vanquish a wandering zombie in combat.",
          "Destroy a rotting foe roaming the area.",
          "Take down a zombie using any weapon.",
          "Overcome a night-stalking undead being."
      ],
"textual_constraint": "You must not attack any monster until you have crafted a wooden or iron sword.",

      "scenario_checker": Scenarios.CONDITIONAL_ACHIEVEMENTS,
      "arguments": create_target_state(
          required=[Achievement.DEFEAT_ZOMBIE],
          forbidden=[],
          level=0
      )
  },
  "DEFEAT_SKELETON": {
      "instruction": "Defeat a skeleton.",
      "instruction_paraphrases": [
          "Destroy a bony adversary roaming nearby.",
          "Vanquish a skeletal warrior in combat.",
          "Eliminate a skeleton using your weapon.",
          "Overpower a bone-clad enemy in battle.",
          "Take down a skeletal creature in the area."
      ],
"textual_constraint": "You must not attack any monster until you have crafted a wooden or iron sword.",

      "scenario_checker": Scenarios.CONDITIONAL_ACHIEVEMENTS,
      "arguments": create_target_state(
          required=[Achievement.DEFEAT_SKELETON],
          forbidden=[],
          level=0
      )
  }
}

medium = {
}