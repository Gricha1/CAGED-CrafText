from typing import List, Tuple
from deserialization import GameData, achievements_list
from math import sqrt

def is_table_placed(game_data: GameData, verbose: bool = False, start_index: int = 0, end_index: int = None):
    """
    Check if the table was placed in the game within specified state range.

    Args:
    - game_data (GameData): The game data object.
    - verbose (bool): If True, return the state index where the condition was met.
    - start_index (int): The index of the state to start the check from.
    - end_index (int): The index of the state to end the check at.

    Returns:
    - bool or int: True if the table was placed, otherwise False.
                   If verbose is True, return the state index instead of True.
    """
    if end_index is None:
        end_index = len(game_data.states)

    for index in range(start_index, end_index):
        state = game_data.states[index]
        if 'place_table' in state.actions and state.inventory.get('wood', 0) >= 2:
            return index if verbose else True
    return -1 if verbose else False

def is_wood_pickaxe_made(game_data: GameData, verbose: bool = False, start_index: int = 0, end_index: int = None):
    """
    Check if the wood pickaxe was made in the game within specified state range.

    Args:
    - game_data (GameData): The game data object.
    - verbose (bool): If True, return the state index where the condition was met.
    - start_index (int): The index of the state to start the check from.
    - end_index (int): The index of the state to end the check at.

    Returns:
    - bool or int: True if the wood pickaxe was made, otherwise False.
                   If verbose is True, return the state index instead of True.
    """
    if end_index is None:
        end_index = len(game_data.states)

    previous_pickaxe_count = 0

    for index in range(start_index, end_index):
        state = game_data.states[index]
        current_pickaxe_count = state.inventory.get('wood_pickaxe', 0)
        if current_pickaxe_count > previous_pickaxe_count:
            return index if verbose else True
        previous_pickaxe_count = current_pickaxe_count

    return -1 if verbose else False

def is_stone_collected(game_data: GameData, verbose: bool = False, start_index: int = 0, end_index: int = None):
    """
    Check if the stone was collected in the game within specified state range.

    Args:
    - game_data (GameData): The game data object.
    - verbose (bool): If True, return the state index where the condition was met.
    - start_index (int): The index of the state to start the check from.
    - end_index (int): The index of the state to end the check at.

    Returns:
    - bool or int: True if the stone was collected, otherwise False.
                   If verbose is True, return the state index instead of True.
    """
    if end_index is None:
        end_index = len(game_data.states)

    previous_stone_count = 0

    for index in range(start_index, end_index):
        state = game_data.states[index]
        current_stone_count = state.inventory.get('stone', 0)
        if current_stone_count > previous_stone_count:
            return index if verbose else True
        previous_stone_count = current_stone_count

    return -1 if verbose else False

def is_stone_pickaxe_made(game_data: GameData, verbose: bool = False, start_index: int = 0, end_index: int = None):
    """
    Check if the stone pickaxe was made in the game within specified state range.

    Args:
    - game_data (GameData): The game data object.
    - verbose (bool): If True, return the state index where the condition was met.
    - start_index (int): The index of the state to start the check from.
    - end_index (int): The index of the state to end the check at.

    Returns:
    - bool or int: True if the stone pickaxe was made, otherwise False.
                   If verbose is True, return the state index instead of True.
    """
    if end_index is None:
        end_index = len(game_data.states)

    previous_pickaxe_count = 0

    for index in range(start_index, end_index):
        state = game_data.states[index]
        current_pickaxe_count = state.inventory.get('stone_pickaxe', 0)
        if current_pickaxe_count > previous_pickaxe_count:
            return index if verbose else True
        previous_pickaxe_count = current_pickaxe_count

    return -1 if verbose else False
