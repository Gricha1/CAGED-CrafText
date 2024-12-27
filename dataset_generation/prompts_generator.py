
def generate_prompt(template, **kwargs):
    """Generates a single prompt using a template and keyword arguments."""
    return template.format(**kwargs)


def generate_prompts(count_goals, instructions_class):
    """Generates a list of prompts based on the instruction class and count."""
    # Read base prompt
    with open("prompts/instruction_generation.txt", 'r') as f:
        base_prompt = f.read()
    
    # Load goals, code template, and example based on the instruction class
    if instructions_class == "achivments":
        from dataset_generation.prompts.achivments.goal_generator import generate_example_goals
        goals = generate_example_goals(num_goals=count_goals)
        with open("prompts/achivments/template.txt", 'r') as f:
            code = f.read()
        with open("prompts/achivments/example.txt", 'r') as f:
            example = f.read()
    
    elif instructions_class == "building_line":
        from dataset_generation.prompts.building_line.goal_generator import generate_example_goals
        goals = generate_example_goals(num_goals=count_goals)
        with open("prompts/building_line/template.txt", 'r') as f:
            code = f.read()
        with open("prompts/building_line/example.txt", 'r') as f:
            example = f.read()
    else:
        raise ValueError(f"Unknown instruction class: {instructions_class}")
    
    # Generate prompts
    prompts = [
        generate_prompt(base_prompt, code=code, instruction=goal, example=example)
        for goal in goals
    ]
    
    return prompts
