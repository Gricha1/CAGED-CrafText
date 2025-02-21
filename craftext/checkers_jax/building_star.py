import jax.numpy as jnp
import jax.lax as lax

# Blocks list as an example
blocks_list = [
    "INVALID", "OUT_OF_BOUNDS", "GRASS", "WATER", "STONE", "TREE", 
    "WOOD", "PATH", "COAL", "IRON", "DIAMOND", "CRAFTING_TABLE", 
    "FURNACE", "SAND", "LAVA", "PLANT", "RIPE_PLANT", "WALL", 
    "DARKNESS", "WALL_MOSS", "STALAGMITE", "SAPPHIRE", "RUBY", 
    "CHEST", "FOUNTAIN", "FIRE_GRASS", "ICE_GRASS", "GRAVEL", 
    "FIRE_TREE", "ICE_SHRUB", "ENCHANTMENT_TABLE_FIRE", 
    "ENCHANTMENT_TABLE_ICE", "NECROMANCER", "GRAVE", "GRAVE2", 
    "GRAVE3", "NECROMANCER_VULNERABLE"
]



def check_cross(region, stone_index, size):

    center = size // 2
    
    if size == 3:
        straight = (jnp.all(region[center, :] == stone_index) &
                    jnp.all(region[:, center] == stone_index))

        diagonal = (jnp.all(jnp.diag(region) == stone_index) &
                    jnp.all(jnp.diag(jnp.fliplr(region)) == stone_index))
        
        return straight | diagonal
    
    elif size in (5, 7):  
        straight = (jnp.all(region[center, :] == stone_index) &
                    jnp.all(region[:, center] == stone_index))
        
        diagonal = (jnp.all(jnp.diag(region) == stone_index) &
                    jnp.all(jnp.diag(jnp.fliplr(region)) == stone_index))
        
        return straight | diagonal | (straight & diagonal)
    
    return False

def scan_cross_function(carry, x):
    region, stone_index, region_size, size = carry
    i, j = x // region_size, x % region_size
    
    if i + size > region_size or j + size > region_size:
        return carry, False 
    
    sub_region = region[i:i+size, j:j+size]
    is_cross = check_cross(sub_region, stone_index, size)
    
    return carry, is_cross

def is_cross_formed(game_data, block_name, size=3, radius=5):

    stone_index = block_name.value
    
    if game_data is None or game_data.states is None:
        return jnp.array(False)
    
    game_map = game_data.states[0].map.game_map
    if game_map is None:
        return jnp.array(False)
    
    player_position = game_data.states[0].variables.player_position
    if player_position is None:
        return jnp.array(False)
    
    x, y = player_position
    region_size = 2 * radius + 1
    
    region = lax.dynamic_slice(
        game_map,
        start_indices=(x - radius, y - radius),
        slice_sizes=(region_size, region_size)
    )
    
    indices = jnp.arange(region_size * region_size)
    carry = (region, stone_index, region_size, size)
    _, crosses = lax.scan(scan_cross_function, carry, indices)
    
    return jnp.any(crosses)
