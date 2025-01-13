from craftext.checkers_jax.conditional import conditional_placing
from craftext.scenarios.constants import InventoryItems, BlockType

easy_test_parafrased = {}


# TODO: need to check how many synonyms are used

easy_test_other_paramets = {}

medium_test_parafrased = {}

medium_test_other_paramets = {}

from craftext.scenarios.parce_dataset import update_previous_dict

easy_test_other_paramets = update_previous_dict(easy_test_other_paramets, "jax_conditional_placing/instructions/test/easy/other_params", "jax_conditional_placing_test_op")
medium_test_other_paramets = update_previous_dict(medium_test_other_paramets, "jax_conditional_placing/instructions/test/medium/other_params", "jax_conditional_placing_test_op")
