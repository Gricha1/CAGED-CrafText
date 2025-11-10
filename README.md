# CAGED - CrafText

## Installation

1. Clone the repository.
2. Create a virtual environment and install the dependencies from `requirements.txt`:

   ```bash
   cd docker
   sh build_12cuda.sh
   sh start_12cuda.sh
   ```
   
### Run the PPO Lagrangian in CMDP
   ```bash
   cd baselines
   ```

   ```bash
   python ppo_lag_with_instruction.py --craftext_settings {setting} --env_name="Craftax-Classic-Pixels-v1-Text" --num_envs=512
   ```

        'check_lambda': lambda ...: scenario_function(...): ...  # Example usage of the function
    }
}
```

Replace `instruction_id` with a unique identifier for each instruction, and complete the `check_lambda` to demonstrate how you would verify the given instruction using the function.
