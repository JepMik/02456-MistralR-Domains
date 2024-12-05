import os
import json
import matplotlib.pyplot as plt

# Define the directory containing checkpoints grouped by task
checkpoints_dir = {
    "Linguistic": [
        "FineTuned_Linguistic_R1_Test",
        "FineTuned_Linguistic_R4_Test",
        "FineTuned_Linguistic_R8_Test",
        "FineTuned_Linguistic_R16_Test"
    ],
    "Math": [
        "FineTuned_Math_R1_Test",
        "FineTuned_Math_R4_Test",
        "FineTuned_Math_R8_Test",
        "FineTuned_Math_R16_Test"
    ]
}

# Directory to save the plots
output_plots_dir = "plots"

# Ensure the output directory exists
os.makedirs(output_plots_dir, exist_ok=True)

# Function to find trainer_state.json recursively in a directory
def find_trainer_state(checkpoint_path):
    for root, dirs, files in os.walk(checkpoint_path):
        if "trainer_state.json" in files:
            return os.path.join(root, "trainer_state.json")
    return None

# Iterate over the tasks (Linguistic and Math)
for task, checkpoint_list in checkpoints_dir.items():
    print(f"Processing task: {task}")
    
    for checkpoint_dir in checkpoint_list:
        checkpoint_path = checkpoint_dir  # Adjust this path as needed
        print(f"Checking {checkpoint_path} for checkpoint directories...")
        
        # Extract R value (e.g., R1, R4, R8, R16)
        r_value = checkpoint_dir.split("_")[-2]  # R1, R4, R8, R16
        
        # Create a subfolder for the R value within the task folder
        task_r_output_dir = os.path.join(output_plots_dir, task, f"R{r_value}")
        os.makedirs(task_r_output_dir, exist_ok=True)
        
        # Iterate through subdirectories to handle multiple checkpoints (e.g., checkpoint-3000)
        for checkpoint in sorted(os.listdir(checkpoint_path)):
            checkpoint_full_path = os.path.join(checkpoint_path, checkpoint)
            
            # Check if it's a directory that starts with "checkpoint"
            if os.path.isdir(checkpoint_full_path) and checkpoint.startswith("checkpoint"):
                print(f"Processing {checkpoint_full_path}...")
                
                # Check if trainer_state.json exists in the checkpoint directory
                trainer_state_path = find_trainer_state(checkpoint_full_path)
                
                if not trainer_state_path:
                    print(f"Skipping {checkpoint}: trainer_state.json not found.")
                    continue
                
                # Load trainer_state.json
                with open(trainer_state_path, "r") as f:
                    trainer_state = json.load(f)
                
                # Extract log history
                log_history = trainer_state.get("log_history", [])
                train_steps = [entry["step"] for entry in log_history if "loss" in entry]
                train_losses = [entry["loss"] for entry in log_history if "loss" in entry]
                eval_steps = [entry["step"] for entry in log_history if "eval_loss" in entry]
                eval_losses = [entry["eval_loss"] for entry in log_history if "eval_loss" in entry]
                
                # Plot training and evaluation loss
                plt.figure(figsize=(8, 6))
                
                # Plot training loss (frequent points)
                if train_losses:
                    plt.plot(train_steps, train_losses, label="Training Loss (frequent)", marker="o", linestyle="-", alpha=0.7)
                
                # Plot evaluation loss (epoch-level points)
                if eval_losses:
                    plt.scatter(eval_steps, eval_losses, label="Evaluation Loss (per epoch)", color="red", marker="x", s=50)
                
                plt.title(f"Loss vs Steps for {checkpoint}")
                plt.xlabel("Steps")
                plt.ylabel("Loss")
                plt.legend()
                plt.grid()
                
                # Save the plot in R value subfolder within task folder
                # Create the file name using task, checkpoint directory, and checkpoint number
                checkpoint_num = checkpoint.split("-")[-1]  # Extract the checkpoint number (e.g., 3000, 5000)
                plot_filename = f"{checkpoint_dir}_{checkpoint_num}.png"
                
                # Save the plot with the generated filename
                plot_path = os.path.join(task_r_output_dir, plot_filename)
                plt.savefig(plot_path)
                plt.close()
                print(f"Saved plot for {checkpoint} at {plot_path}.")
