import jax
import jax.numpy as jnp

def check_line_2(center, region, check_diagonal=False):
    """
    Проверка наличия линии размера 2.
    Если check_diagonal=True, проверяется диагональная линия.
    """
    i, j = center

    def check_diagonal_lines(_):
        diagonal_1 = jnp.array([
            region[i, j],
            region[i + 1, j + 1]
        ])
        diagonal_2 = jnp.array([
            region[i, j],
            region[i + 1, j - 1]
        ])

        return (jnp.sum(diagonal_1) == 2) & jnp.all(diagonal_1 == 1) | \
               (jnp.sum(diagonal_2) == 2) & jnp.all(diagonal_2 == 1)

    def check_straight_lines(_):
        vertical = jnp.array([
            region[i, j],
            region[i + 1, j]
        ])
        horizontal = jnp.array([
            region[i, j],
            region[i, j + 1]
        ])

        return (jnp.sum(vertical) == 2) & jnp.all(vertical == 1) | \
               (jnp.sum(horizontal) == 2) & jnp.all(horizontal == 1)

    return jax.lax.cond(check_diagonal, check_diagonal_lines, check_straight_lines, None)


def check_line_3(center, region, check_diagonal=False):
    """
    Проверка наличия линии размера 3.
    Если check_diagonal=True, проверяется диагональная линия.
    """
    i, j = center

    def check_diagonal_lines(_):
        diagonal_1 = jnp.array([
            region[i, j],
            region[i + 1, j + 1],
            region[i + 2, j + 2]
        ])
        diagonal_2 = jnp.array([
            region[i, j],
            region[i + 1, j - 1],
            region[i + 2, j - 2]
        ])

        return (jnp.sum(diagonal_1) == 3) & jnp.all(diagonal_1 == 1) | \
               (jnp.sum(diagonal_2) == 3) & jnp.all(diagonal_2 == 1)

    def check_straight_lines(_):
        vertical = jnp.array([
            region[i, j],
            region[i + 1, j],
            region[i + 2, j]
        ])
        horizontal = jnp.array([
            region[i, j],
            region[i, j + 1],
            region[i, j + 2]
        ])

        return (jnp.sum(vertical) == 3) & jnp.all(vertical == 1) | \
               (jnp.sum(horizontal) == 3) & jnp.all(horizontal == 1)

    return jax.lax.cond(check_diagonal, check_diagonal_lines, check_straight_lines, None)


def check_line_4(center, region, check_diagonal=False):
    """
    Проверка наличия линии размера 4.
    Если check_diagonal=True, проверяется диагональная линия.
    """
    i, j = center

    def check_diagonal_lines(_):
        diagonal_1 = jnp.array([
            region[i, j],
            region[i + 1, j + 1],
            region[i + 2, j + 2],
            region[i + 3, j + 3]
        ])
        diagonal_2 = jnp.array([
            region[i, j],
            region[i + 1, j - 1],
            region[i + 2, j - 2],
            region[i + 3, j - 3]
        ])

        return (jnp.sum(diagonal_1) == 4) & jnp.all(diagonal_1 == 1) | \
               (jnp.sum(diagonal_2) == 4) & jnp.all(diagonal_2 == 1)

    def check_straight_lines(_):
        vertical = jnp.array([
            region[i, j],
            region[i + 1, j],
            region[i + 2, j],
            region[i + 3, j]
        ])
        horizontal = jnp.array([
            region[i, j],
            region[i, j + 1],
            region[i, j + 2],
            region[i, j + 3]
        ])

        return (jnp.sum(vertical) == 4) & jnp.all(vertical == 1) | \
               (jnp.sum(horizontal) == 4) & jnp.all(horizontal == 1)

    return jax.lax.cond(check_diagonal, check_diagonal_lines, check_straight_lines, None)
