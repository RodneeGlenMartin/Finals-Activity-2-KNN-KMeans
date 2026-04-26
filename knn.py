import csv
import math
from collections import Counter
import matplotlib.pyplot as plt

# File path to our dataset
csv_file = r"d:\Desktop\kn\knn_dataset.csv"

# 1. Read our dataset
dataset = []
with open(csv_file, mode='r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        dataset.append({
            "CustomerID": int(row["CustomerID"]),
            "Annual_Income_k": float(row["Annual_Income_k"]),
            "Store_Visits_Per_Month": float(row["Store_Visits_Per_Month"]),
            "Customer_Tier": row["Customer_Tier"]
        })

# The new entry to classify
# Let's say a new customer has an Annual Income of $65k and visits 5 times a month
new_entry = {
    "Annual_Income_k": 65.0, 
    "Store_Visits_Per_Month": 5.0
}

print("--- New Entry ---")
print(f"Income: {new_entry['Annual_Income_k']}k, Visits: {new_entry['Store_Visits_Per_Month']}, Tier: ?\n")

# 2. Compute the distance individually for each row
print("--- Computing Distances ---")
for i, row in enumerate(dataset):
    x1 = row["Annual_Income_k"]
    y1 = row["Store_Visits_Per_Month"]
    x2 = new_entry["Annual_Income_k"]
    y2 = new_entry["Store_Visits_Per_Month"]
    
    # Distance formula: sqrt((X2 - X1)^2 + (Y2 - Y1)^2)
    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    row["distance"] = round(distance, 2)
    
    # Printing the step-by-step computation
    print(f"d{i+1} (ID {row['CustomerID']}) = sqrt(({x2} - {x1})^2 + ({y2} - {y1})^2)")
    print(f"   = {row['distance']}")

# 3. Sort in Ascending Order
print("\n--- Sorted Dataset by Distance (Ascending) ---")
dataset.sort(key=lambda x: x["distance"])

print(f"{'ID':<5} {'INCOME(k)':<12} {'VISITS':<10} {'TIER':<10} {'DISTANCE':<8}")
for row in dataset:
    print(f"{row['CustomerID']:<5} {row['Annual_Income_k']:<12} {row['Store_Visits_Per_Month']:<10} {row['Customer_Tier']:<10} {row['distance']:<8}")

# 4. Choose K? Let's choose K=3
K = 3
print(f"\n--- Choosing K = {K} ---")
top_k = dataset[:K]
print(f"Top {K} nearest neighbors:")
for row in top_k:
    print(f"- Tier: {row['Customer_Tier']} (Distance: {row['distance']})")

# Count the classes in the top K to determine the majority
classes = [row["Customer_Tier"] for row in top_k]
class_counts = Counter(classes)
predicted_tier = class_counts.most_common(1)[0][0]

print(f"\nThe most common tier among the top {K} is: {predicted_tier}")
print(f"Therefore, the new entry is classified as: {predicted_tier}")

# 5. Updated Dataset: insert the new point
print("\n--- Updated Dataset (Inserting the new point) ---")
new_customer_id = max(row["CustomerID"] for row in dataset) + 1
new_entry_full = {
    "CustomerID": new_customer_id,
    "Annual_Income_k": new_entry["Annual_Income_k"],
    "Store_Visits_Per_Month": new_entry["Store_Visits_Per_Month"],
    "Customer_Tier": predicted_tier
}

# Clean up distance from dataset and append the new entry
updated_dataset = []
for row in dataset:
    updated_dataset.append({
        "CustomerID": row["CustomerID"],
        "Annual_Income_k": row["Annual_Income_k"],
        "Store_Visits_Per_Month": row["Store_Visits_Per_Month"],
        "Customer_Tier": row["Customer_Tier"]
    })

updated_dataset.append(new_entry_full)

# Sort back by ID for clean chronological output
updated_dataset.sort(key=lambda x: x["CustomerID"])

print(f"{'ID':<5} {'INCOME(k)':<12} {'VISITS':<10} {'TIER':<10}")
for row in updated_dataset:
    print(f"{row['CustomerID']:<5} {row['Annual_Income_k']:<12} {row['Store_Visits_Per_Month']:<10} {row['Customer_Tier']:<10}")

# 6. Visualization
print("\n--- Visualizing the KNN Process ---")
plt.figure(figsize=(10, 6))

# Define colors for our tiers
colors = {'Basic': 'green', 'Silver': 'orange', 'Gold': 'purple'}

# Plot existing data
for row in dataset:
    plt.scatter(row["Annual_Income_k"], row["Store_Visits_Per_Month"], 
                color=colors.get(row["Customer_Tier"], 'blue'), label=row["Customer_Tier"], s=100, zorder=2)

# Handle duplicate labels in legend
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))

# Plot the new entry
new_point_plot = plt.scatter(new_entry["Annual_Income_k"], new_entry["Store_Visits_Per_Month"], 
            color='red', marker='*', s=300, label='New Entry (To Classify)', edgecolor='black', zorder=3)
by_label['New Entry (To Classify)'] = new_point_plot

# Draw dashed lines to the K nearest neighbors
line_handle = None
for i, row in enumerate(top_k):
    line, = plt.plot([new_entry["Annual_Income_k"], row["Annual_Income_k"]], 
             [new_entry["Store_Visits_Per_Month"], row["Store_Visits_Per_Month"]], 
             'k--', alpha=0.5, zorder=1)
    if i == 0:
        line_handle = line
if line_handle:
    by_label['Distance to NN'] = line_handle

# Draw a circle around the K nearest neighbors
if len(top_k) > 0:
    kth_distance = top_k[-1]["distance"]
    circle = plt.Circle((new_entry["Annual_Income_k"], new_entry["Store_Visits_Per_Month"]), 
                        kth_distance, color='gray', fill=False, linestyle='--', linewidth=1.5)
    plt.gca().add_patch(circle)
    # For legend, create a proxy artist
    import matplotlib.lines as mlines
    circle_legend = mlines.Line2D([], [], color='gray', marker='o', linestyle='None',
                          markersize=10, markerfacecolor='none', markeredgecolor='gray', markeredgewidth=1.5)
    by_label[f'k={K} Boundary'] = circle_legend

plt.title(f"K-Nearest Neighbors Visualization (k={K})\nPredicted Class: {predicted_tier}")
plt.xlabel("Annual Income ($k)")
plt.ylabel("Store Visits Per Month")
plt.legend(by_label.values(), by_label.keys(), loc='center left', bbox_to_anchor=(1, 0.5))
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(r"d:\Desktop\kn\knn_visualization.png", dpi=300, bbox_inches='tight')
print(f"\nVisualization saved to: d:\\Desktop\\kn\\knn_visualization.png")
plt.show()
