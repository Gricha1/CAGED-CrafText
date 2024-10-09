from craftext.checkers_jax.building import is_line_formed, is_square_formed
from craftext.scenarios.build_line import test as build_line_instructions
from craftext.scenarios.build_squere import test as build_squere_instructions
from craftext.scenarios.localization_place import test as localization_place_instructions

# Merging 'easy' dictionaries
easy_test_parafrased = {**build_line_instructions.easy_test_parafrased, **build_squere_instructions.easy_test_parafrased, **localization_place_instructions.easy_test_parafrased}


# Merging 'easy' dictionaries
easy_test_other_paramets = {**build_line_instructions.easy_test_other_paramets, **build_squere_instructions.easy_test_other_paramets, **localization_place_instructions.easy_test_other_paramets}
