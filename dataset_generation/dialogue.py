import openai
import os
from tqdm import tqdm
import argparse
from prompts_generator import generate_prompts


def check_and_format(content):
    """Extracts content after the first '=' if present."""
    if "=" in content:
        return content.split("=", 1)[1]
    return content


def main(count_goals, instructions_class, output_file):
    """Generates instructions based on prompts."""
    instructions = []
    prompts = generate_prompts(count_goals, instructions_class)
    
    for prompt in tqdm(prompts, desc='Generating Instructions'):
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
        )
        content = response["choices"][0]["message"]["content"]
        content = check_and_format(content)
        instructions.append(content)
        
        with open(output_file, "a") as f:
            f.write(content + "\n----\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate Instructions from Prompts')
    parser.add_argument('--count_goals', type=int, required=True, help='Number of goals to generate instructions for')
    parser.add_argument('--instructions_class', type=str, required=True, help='Class of instructions to generate')
    parser.add_argument('--output_file', type=str, required=True, help='Output file to save instructions')
    
    args = parser.parse_args()
    
    # Fetch API key from environment variable
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set.")
    
    openai.api_key = api_key.strip()
    
    main(args.count_goals, args.instructions_class, args.output_file)
