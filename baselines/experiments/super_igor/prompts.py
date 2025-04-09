class PlanFormatError(Exception):
    """Raised when the model answer format is invalid or cannot be parsed."""
    pass

class PlanExtractor():
    @staticmethod
    def extract(model_answer, prompt_template):
        if prompt_template == 0: #base prompt
            return BasePlanExtractor.extract(model_answer)
        else:
            return FunctionPlanExtractor.extract(model_answer)
        
class BasePlanExtractor():
    @staticmethod
    def extract(model_answer):
        try:
            formeted_r = ""
            if "Plan:" in model_answer:
                formeted_r = model_answer.split("Plan:")[2]
            elif "1" in model_answer:
                formeted_r = model_answer.split("1")[2]
            else:
                formeted_r = model_answer
            if "Finish!" in model_answer:
                formeted_r = formeted_r.split("Finish!")[0] + "Finish!"
            return formeted_r
        except:
            raise PlanFormatError(f"Incorrect plan format, model answer was {model_answer}")
    @staticmethod
    def extract_prompt(prompt_with_answer):
        return "Plan: ".join(prompt_with_answer.split("Plan: ")[:2])+"Plan: "
    
class FunctionPlanExtractor():
    @staticmethod
    def extract(model_answer):
        try:
            print(model_answer)
            list_starts = model_answer.split("[",)[2]
            list_end = list_starts.split("]",)[0]
            return "\n".join(list_end.split("\n"))
        except Exception as e:
            raise PlanFormatError(f"Incorrect plan format: {e}")
    @staticmethod
    def extract_prompt(prompt_with_answer):
        return "Answer: ".join(prompt_with_answer.split("Answer: ")[:2])+"Answer: "
        
    
BASE_PROMPT = """
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
        Plan:
"""



PROMPT_WITH_FUNCTIONS = f"""
        You control an agent in a 2D game with siplified Minecraft environment. You will need to provide a detailed step-by-step plan for following the user's instructions. 
        You must include all the preliminary steps that it needs to complete.

        Send your answer as a python list.
        Instruction: Make a pickaxe from wood
        Answer: ["gather_resource(resource_type = wood, count = 2)", "create_item(table)", "gather_resource(resource_type = wood, count = 2)", "create_item(wooden pickaxe)"]

        Send your answer as a python list.
        Instruction: $INSTRUCTION$  
        Answer: 
        """
        
PROMPT_WITH_FUCNTIONS_HINTS = f"""
        You control an agent in a 2D game with siplified Minecraft environment. You will need to provide a detailed step-by-step plan for following the user's instructions. 
        You must include all the preliminary steps that it needs to complete.

        For doing this each step descrbe in format of subgoals:
        {{'gather_resource': "'resource_type'", "count"}}, 
        {{'place_item': "'item_type'}},
        {{construct_figure: 'block_name', 'figure_type', 'side_size'}},
        {{'create_item': "'item_type'"}},
        {{'defeat_enemy': "'enemy_type'"}},
        {{'place_item_relative_to_another': "'item_type_to_place', 'item_type_reference', 'direction', 'distance'"}},

        Possible arguments:
        resource_type: wood, stone, coal, diamond, iron, plant, water
        item_type: stone, table, plant, furnace
        block_name: stone, table, plant, furnace
        figure_type: line, diagonal_line, square
        item_type: table, wooden sword, wooden pickaxe, stone sword, stone pickaxe, iron sword, iron pickaxe
        enemy_type: zombie, skeleton, cow
        item_type_to_place: stone, table, plant, furnace
        item_type_reference: stone, table, plant, furnace, water, tree, stone, coal, diamond, iron, plant
        direction: left, right, bottom, top

        Send your answer as a python list.
        Instruction: Make a pickaxe from wood
        Answer: 
        ["gather_resource(resource_type = wood, count = 2)",
        "create_item(table)", 
        "gather_resource(resource_type = wood, count = 2)", 
        "create_item(wooden pickaxe)"]

        Send your answer as a python list.
        Instruction: $INSTRUCTION$  
        Answer: 
        """

PROMPTS = [BASE_PROMPT, PROMPT_WITH_FUNCTIONS, PROMPT_WITH_FUCNTIONS_HINTS]
class PromptTemplate:
    def __init__(self, template: str):
        self.template = template

    def render(self, **kwargs) -> str:
        rendered_prompt = self.template
        for key, value in kwargs.items():
            placeholder = f"${key.upper()}$"
            rendered_prompt = rendered_prompt.replace(placeholder, value)
        return rendered_prompt

def promt_instruction(instruction, prompt=BASE_PROMPT):
    prompt_template = PromptTemplate(prompt)
    instruction = instruction
    full_prompt = prompt_template.render(INSTRUCTION=instruction)
    return full_prompt
