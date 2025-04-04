from typing import List, TypeVar, Type
import jax.numpy as jnp

T = TypeVar("T")

def list_to_array(lst: List[T]) -> T:
    """Convert a list of dataclass instances to a batched version with jnp.arrays."""
    if not lst:
        raise ValueError("Input list is empty.")

    cls: Type[T] = type(lst[0])  # Определяем класс элементов списка
    converted_data = {}

    for k, field in cls.__dataclass_fields__.items():
        values = [getattr(v, k) for v in lst]

        # Если поле уже является jnp.ndarray, то стекуем его вдоль первой оси
        if isinstance(values[0], jnp.ndarray):
            converted_data[k] = jnp.stack(values, axis=0)  # Собираем массив массивов
        elif isinstance(values[0], (int, float, bool)):  
            converted_data[k] = jnp.array(values)  # Просто массив скаляров
        else:
            converted_data[k] = list_to_array(values)  # Рекурсивный вызов для вложенных датаклассов

    return cls(**converted_data)
