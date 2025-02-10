from craftext.checkers_jax.building import is_line_formed, is_square_formed
from craftext.scenarios.constants import BlockType

easy_test_parafrased = {}

easy_test_other_paramets = {}



# TODO: check complexity
medium_test_parafrased = {}


medium_test_other_paramets = {}

from craftext.scenarios.parce_dataset import update_previous_dict

from craftext.scenarios.constants import base_path
import os

easy_test_other_paramets = update_previous_dict(
    easy_test_other_paramets, 
    os.path.join(base_path, "jax_build_line/instructions/test/easy/other_params"), 
    "jax_build_line_test_op"
)



medium_test_other_paramets = update_previous_dict(
    medium_test_other_paramets, 
    os.path.join(base_path, "jax_build_line/instructions/test/medium/other_params"), 
    "jax_build_line_test_op"
)


easy_test_parafrased = update_previous_dict(
    easy_test_parafrased, 
    os.path.join(base_path, "jax_build_line/instructions/test/easy/paraphrases"), 
    "jax_build_line_test_paraphrases"
)


medium_test_parafrased = update_previous_dict(
    medium_test_parafrased, 
    os.path.join(base_path, "jax_build_line/instructions/test/medium/paraphrases"), 
    "jax_build_line_test_paraphrases"
)
