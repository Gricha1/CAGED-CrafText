import jax.numpy as jnp
import jax
from typing import Tuple

def check_square_2x2(center: Tuple[int, int], region: jax.Array, stone_index: int) -> jax.Array:
    """
    Проверка наличия квадрата 2x2 из блоков stone_index.
    """
    i, j = center

    return jnp.all(jnp.array([
        # Проверка центрального квадрата 2x2
        region[i, j] == stone_index,
        region[i+1, j] == stone_index,
        region[i, j+1] == stone_index,
        region[i+1, j+1] == stone_index,

        # Проверка внешней границы 4x4 без блока stone_index
        region[i-1, j-1] != stone_index,
        region[i-1, j] != stone_index,
        region[i-1, j+1] != stone_index,
        region[i-1, j+2] != stone_index,
        region[i, j-1] != stone_index,
        region[i+1, j-1] != stone_index,
        region[i+2, j-1] != stone_index,
        region[i+2, j] != stone_index,
        region[i+2, j+1] != stone_index,
        region[i+2, j+2] != stone_index,
        region[i, j+2] != stone_index,
        region[i+1, j+2] != stone_index
    ]))


def check_square_3x3(center: Tuple[int, int], region: jax.Array, stone_index: int) -> jax.Array:
    """
    Проверка наличия квадрата 3x3 из блоков stone_index.
    """
    i, j = center

    return jnp.all(jnp.array([
        # Проверка центрального квадрата 3x3
        region[i, j] == stone_index,
        region[i-1, j] == stone_index,
        region[i+1, j] == stone_index,
        region[i, j-1] == stone_index,
        region[i, j+1] == stone_index,
        region[i-1, j-1] == stone_index,
        region[i-1, j+1] == stone_index,
        region[i+1, j-1] == stone_index,
        region[i+1, j+1] == stone_index,

        # Проверка внешней границы 5x5 без блока stone_index
        region[i-2, j-2] != stone_index,
        region[i-2, j-1] != stone_index,
        region[i-2, j] != stone_index,
        region[i-2, j+1] != stone_index,
        region[i-2, j+2] != stone_index,
        region[i-1, j-2] != stone_index,
        region[i, j-2] != stone_index,
        region[i+1, j-2] != stone_index,
        region[i+2, j-2] != stone_index,
        region[i+2, j-1] != stone_index,
        region[i+2, j] != stone_index,
        region[i+2, j+1] != stone_index,
        region[i+2, j+2] != stone_index,
        region[i-1, j+2] != stone_index,
        region[i, j+2] != stone_index,
        region[i+1, j+2] != stone_index
    ]))


def check_square_4x4(center: Tuple[int, int], region: jax.Array, stone_index: int) -> jax.Array:
    """
    Проверка наличия квадрата 4x4 из блоков stone_index.
    """
    i, j = center

    return jnp.all(jnp.array([
        # Проверка центрального квадрата 4x4
        region[i, j] == stone_index,
        region[i+1, j] == stone_index,
        region[i+2, j] == stone_index,
        region[i+3, j] == stone_index,
        region[i, j+1] == stone_index,
        region[i+1, j+1] == stone_index,
        region[i+2, j+1] == stone_index,
        region[i+3, j+1] == stone_index,
        region[i, j+2] == stone_index,
        region[i+1, j+2] == stone_index,
        region[i+2, j+2] == stone_index,
        region[i+3, j+2] == stone_index,
        region[i, j+3] == stone_index,
        region[i+1, j+3] == stone_index,
        region[i+2, j+3] == stone_index,
        region[i+3, j+3] == stone_index,

        # Проверка внешней границы 6x6 без блока stone_index
        region[i-1, j-1] != stone_index,
        region[i-1, j] != stone_index,
        region[i-1, j+1] != stone_index,
        region[i-1, j+2] != stone_index,
        region[i-1, j+3] != stone_index,
        region[i-1, j+4] != stone_index,
        region[i, j-1] != stone_index,
        region[i+1, j-1] != stone_index,
        region[i+2, j-1] != stone_index,
        region[i+3, j-1] != stone_index,
        region[i+4, j-1] != stone_index,
        region[i+4, j] != stone_index,
        region[i+4, j+1] != stone_index,
        region[i+4, j+2] != stone_index,
        region[i+4, j+3] != stone_index,
        region[i+4, j+4] != stone_index,
        region[i, j+4] != stone_index,
        region[i+1, j+4] != stone_index,
        region[i+2, j+4] != stone_index,
        region[i+3, j+4] != stone_index
    ]))
