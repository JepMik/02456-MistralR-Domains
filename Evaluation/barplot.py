# Script that generates bar plots for the evaluation of the generated responses by the model
#

import pandas as pd
import sys
import seaborn as sns
import matplotlib.pyplot as plt

data_dict = {
                "Baseline": "Baseline/Results/baseline_scores.csv",
            }


# Define CLI arguments
arg = sys.argv[1]
data = data_dict[arg]

#print(data)
# Load the results
results = pd.read_csv(data)

# Generate bar plots for the evaluation of the generated responses
math_data = sum(results["Math"])/len(results["Math"])
ling_data = sum(results["Linguistic"])/ len(results["Linguistic"])


# Set the theme for the plot
sns.set_theme(style="whitegrid")

# Create the figure and axes for subplots
fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)

# Data for each plot
categories = ["Math", "Linguistic"]
scores = [math_data, ling_data]

# Plot for "Math"
sns.barplot(ax=axes[0], x=["Math"], y=[math_data], palette="Blues")
axes[0].set_ylim(0, 100)
axes[0].set_yticks(range(0, 101, 10))
axes[0].set_title("Math with MathPromptEval")
axes[0].set_ylabel("Mean value")
axes[0].text(0, math_data + 1, f"{math_data:.1f}", ha='center', va='bottom', fontsize=10)

# Plot for "Linguistic"
sns.barplot(ax=axes[1], x=["Linguistic"], y=[ling_data], palette="Greens")
axes[1].set_ylim(0, 100)
axes[1].set_title("Linguistic with LingPromptEval")
axes[1].text(0, ling_data + 1, f"{ling_data:.1f}", ha='center', va='bottom', fontsize=10)

# Adjust layout
plt.tight_layout()
plt.savefig(f"Evaluation/Plots/{arg}_barplot.png")





