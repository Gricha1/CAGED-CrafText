from craftext.checkers_jax.relevant import place_object_relevant_to
from craftext.scenarios.constants import BlockType

easy_test_parafrased = {}


easy_test_other_paramets = {}



medium_test_parafrased = {}

### TODO: check the complexity
medium_test_other_paramets = {}

from craftext.scenarios.constants import base_path
from craftext.scenarios.parce_dataset import update_previous_dict
import os

easy_test_other_paramets = update_previous_dict(
    easy_test_other_paramets, 
    os.path.join(base_path, "jax_localization_place/instructions/test/easy/other_params"), 
    "localization_place_test_op"
)

medium_test_other_paramets = update_previous_dict(
    medium_test_other_paramets, 
    os.path.join(base_path, "jax_localization_place/instructions/test/medium/other_params"), 
    "localization_place_test_op"
)
