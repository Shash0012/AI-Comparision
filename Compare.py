import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

#  Load the dataset
file_path = r"C:\Users\shash\Downloads\AI_Comparision.csv"  # Update your file path
df = pd.read_csv(file_path)

#  Drop rows with missing AI_System or ratings
df = df.dropna(subset=["AI_System", "Accuracy Rating (1-5)", "Creativity Rating (1-5)", "Efficiency_Rating (1-5)"])

#  Convert rating columns to numeric
rating_columns = ["Accuracy Rating (1-5)", "Creativity Rating (1-5)", "Efficiency_Rating (1-5)"]
df[rating_columns] = df[rating_columns].apply(pd.to_numeric, errors="coerce")

# Group data by AI system and calculate mean scores
grouped_df = df.groupby("AI_System")[rating_columns].mean()

# 1. Create a figure with subplots (2 charts in one slide)
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

#Bar Chart: AI Model Performance Comparison
grouped_df.plot(kind="bar", ax=axes[0], colormap="viridis")
axes[0].set_title("AI Model Performance by Metric")
axes[0].set_xlabel("AI Models")
axes[0].set_ylabel("Average Rating (1-5)")
axes[0].set_xticklabels(grouped_df.index, rotation=45)
axes[0].legend(title="Metric Type")
axes[0].grid(axis="y", linestyle="--", alpha=0.7)

#Line Chart: Trends in AI Model Performance
for model in grouped_df.index:
    axes[1].plot(rating_columns, grouped_df.loc[model], marker="o", label=model)
    
axes[1].set_title("Trends in AI Model Performance")
axes[1].set_xlabel("Evaluation Metrics")
axes[1].set_ylabel("Average Rating (1-5)")
axes[1].legend(title="AI Models")
axes[1].grid(True)

#Adjust layout to prevent overlap
plt.tight_layout()
plt.show()

#Interactive Chart - Plotly (Optional)
fig = px.bar(
    grouped_df.reset_index().melt(id_vars=["AI_System"], var_name="Metric", value_name="Average Score"),
    x="AI_System",
    y="Average Score",
    color="Metric",
    barmode="group",
    title="Average AI Model Performance by Metric Type",
    labels={"AI_System": "AI Models", "Average Score": "Average Rating (1-5)", "Metric": "Evaluation Metric"},
)

fig.show()
