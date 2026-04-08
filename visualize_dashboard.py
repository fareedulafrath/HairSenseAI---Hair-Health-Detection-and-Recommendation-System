import pandas as pd
import matplotlib.pyplot as plt

# Load your CSV file
df = pd.read_csv("prediction_log.csv")

# Count how many predictions for each final_stage (0, 1, 2)
stage_counts = df['final_stage'].value_counts().sort_index()

# Define colors for each stage
colors = ['#8BC34A', '#FFC107', '#F44336']  # Green, Orange, Red

# Plot bar chart
plt.figure(figsize=(8,5))
plt.bar(stage_counts.index, stage_counts.values, color=colors)

# Add labels and title
plt.xticks([0,1,2], ["Stage 0 (Normal)", "Stage 1 (Mild Loss)", "Stage 2 (Severe Loss)"])
plt.ylabel("Number of Users")
plt.title("📊 Hair Loss Stage Distribution")
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Show chart
plt.tight_layout()
plt.show()
