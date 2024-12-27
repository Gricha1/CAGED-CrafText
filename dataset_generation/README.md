# Instruction Generator

## Overview
This script generates instructions based on user-defined goals, instruction classes, and difficulty levels using the OpenAI GPT-4 API. The generated instructions are written to an output file.

## Requirements
- Python 3.8+
- OpenAI API key
- Required libraries: `openai`, `tqdm`, `argparse`

## Installation
 Set up your OpenAI API key as an environment variable:
   ```bash
   export OPENAI_API_KEY='your_api_key_here'
   ```

## Usage
Run the script with the following arguments:

```bash
python script.py --count_goals <number_of_goals> \
                 --instructions_class <class_of_instructions> \
                 --output_file <output_file> \
                 [--difficulty EASY|MEDIUM|HARD]
```

### Arguments:
- `--count_goals` *(int, required)*: Number of goals to generate instructions for.
- `--instructions_class` *(str, required)*: Class of instructions to generate.
  - Options: `achivments`, `building_line`, `building_squere`, `conditonal_placing`, `localization_placing`
- `--output_file` *(str, required)*: File to save the generated instructions.
- `--difficulty` *(str, optional, default='EASY')*: Difficulty level (`EASY`, `MEDIUM`, `HARD`).

## Example
```bash
python script.py --count_goals 5 \
                 --instructions_class building_line \
                 --output_file instructions.txt \
                 --difficulty MEDIUM
```

## Script Workflow
1. Prompts are generated based on the given parameters (`count_goals`, `instructions_class`, `difficulty`).
2. For each prompt, a request is sent to the OpenAI GPT-4 API.
3. The content from the response is extracted and formatted.
4. The formatted content is written to the specified output file.

## Environment Variables
- `OPENAI_API_KEY`: Required for authenticating with the OpenAI API.