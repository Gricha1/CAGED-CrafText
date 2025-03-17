
class PromptTemplate:
    def __init__(self, template: str):
        self.template = template

    def render(self, **kwargs) -> str:
        rendered_prompt = self.template
        for key, value in kwargs.items():
            placeholder = f"${key.upper()}$"
            rendered_prompt = rendered_prompt.replace(placeholder, value)
        return rendered_prompt

def promt_instruction(instruction):
    prompt_template = PromptTemplate("""
Craftax is a virtual environment designed for exploration, crafting, and task completion. 
The agent can move, collect resources, craft items, place objects, and interact with its surroundings. 
Tasks often require gathering resources and crafting items before placing or utilizing them.

Task: Create a step-by-step action plan (minimum 2, maximum 3 steps) for the agent in Craftax to achieve the instruction.

Response Format:  
- Provide the plan as a numbered list.  
- Each step must describe a specific action or logical task for the agent, such as gathering resources, crafting items, or placing objects.  
- Keep each step concise (maximum 5 words), clear, and focused on efficiently fulfilling the instruction.  
- End the list with "6. Finish!"

For instruction: "Collect wood and Place Table"  
Plan:  
1. Find tree  
2. Collect some wood  
3. Craft Table  
4. Place Table  
5. Repeat if needed  
6. Finish!  

For Instruction: $INSTRUCTION$  
Plan:""")


    instruction = instruction
    
    full_prompt = prompt_template.render(INSTRUCTION=instruction)
    return full_prompt
