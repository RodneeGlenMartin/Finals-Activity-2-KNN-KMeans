import csv
import math
import random
import matplotlib.pyplot as plt

# File paths
csv_file = r"d:\Desktop\kn\kmeans_dataset.csv"
output_csv = r"d:\Desktop\kn\kmeans_steps_output.csv"

# 1. Transform to table
dataset = []
with open(csv_file, mode='r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        dataset.append({
            "id": f"P{row['CustomerID']}",
            "x": float(row['Annual_Income_k']),
            "y": float(row['Spending_Score']) # Fixed the bug!
        })

# Open output CSV to save the tables directly to a file
with open(output_csv, mode='w', newline='') as out_file:
    writer = csv.writer(out_file)
    
    print("--- 1. Transform to table ---")
    writer.writerow(["--- 1. Transform to table ---"])
    
    header = ['Data Points', 'X (Income)', 'Y (Spending)']
    print(f"{header[0]:<12} | {header[1]:<10} | {header[2]:<10}")
    writer.writerow(header)
    
    for pt in dataset:
        print(f"{pt['id']:<12} | {pt['x']:<10} | {pt['y']:<10}")
        writer.writerow([pt['id'], pt['x'], pt['y']])
    
    writer.writerow([])

    # 2. Initialize Centroids (Fixing Initialization and Confirmation Bias)
    # Using K=5 because this dataset naturally forms 5 clusters.
    # Randomly selecting initial points instead of hardcoding.
    random.seed(42) # Seed for reproducibility in the walkthrough
    initial_pts = random.sample(dataset, 5)
    centroids = []
    for i, pt in enumerate(initial_pts):
        centroids.append({"id": i+1, "x": pt["x"], "y": pt["y"]})

    def compute_distance(p1, p2):
        return math.sqrt((p2['x'] - p1['x'])**2 + (p2['y'] - p1['y'])**2)

    iteration = 1
    while True:
        print(f"\n================ Iteration {iteration} ================")
        writer.writerow([f"================ Iteration {iteration} ================"])
        
        print("\n--- 2. Initialize Centroids (Current) ---")
        writer.writerow(["--- 2. Initialize Centroids (Current) ---"])
        for c in centroids:
            print(f"C{c['id']}: ({c['x']}, {c['y']})")
            writer.writerow([f"C{c['id']}", c['x'], c['y']])
        writer.writerow([])
        
        # 3. Compute the distance and determine cluster
        print("\n--- 3. Compute distance and determine cluster ---")
        writer.writerow(["--- 3. Compute distance and determine cluster ---"])
        
        headers = [f"C{c['id']}({c['x']}, {c['y']})" for c in centroids]
        
        # Terminal Header
        print(f"{'Data Points':<10} | {'Distance to':^75} | {'Cluster'}")
        print(f"{'':<10} | {headers[0]:<13} | {headers[1]:<13} | {headers[2]:<13} | {headers[3]:<13} | {headers[4]:<13} |")
        print("-" * 105)
        
        # CSV Header
        csv_headers = ['Data Points'] + [f'Distance to {h}' for h in headers] + ['Cluster']
        writer.writerow(csv_headers)
        
        clusters_changed = False
        
        for pt in dataset:
            dists = [round(compute_distance(pt, c), 2) for c in centroids]
            min_dist = min(dists)
            cluster = dists.index(min_dist) + 1
            
            old_cluster = pt.get("cluster", -1)
            if old_cluster != cluster:
                clusters_changed = True
                
            pt["cluster"] = cluster
            
            point_str = f"{pt['id']} ({pt['x']},{pt['y']})"
            
            # Terminal Row
            print(f"{point_str:<10} | {dists[0]:<13.2f} | {dists[1]:<13.2f} | {dists[2]:<13.2f} | {dists[3]:<13.2f} | {dists[4]:<13.2f} | {cluster}")
            # CSV Row
            writer.writerow([point_str] + dists + [cluster])
            
        writer.writerow([])
            
        if not clusters_changed:
            msg = "Clusters did not change. K-Means has converged!"
            print(f"\n{msg}")
            writer.writerow([msg])
            writer.writerow([])
            break
            
        # 4. Compute again the distance and determine new centroids
        print("\n--- 4. Compute again and determine new centroids ---")
        writer.writerow(["--- 4. Compute again and determine new centroids ---"])
        writer.writerow(["Centroid ID", "New X", "New Y"])
        
        new_centroids = []
        for i in range(1, 6):
            cluster_points = [p for p in dataset if p["cluster"] == i]
            if cluster_points:
                mean_x = sum(p["x"] for p in cluster_points) / len(cluster_points)
                mean_y = sum(p["y"] for p in cluster_points) / len(cluster_points)
                new_c = {"id": i, "x": round(mean_x, 2), "y": round(mean_y, 2)}
            else:
                new_c = centroids[i-1]
            
            new_centroids.append(new_c)
            print(f"New Centroid C{i}: ({new_c['x']}, {new_c['y']})")
            writer.writerow([f"C{i}", new_c['x'], new_c['y']])
            
        writer.writerow([])
        centroids = new_centroids
        iteration += 1

# Final Clusters Visualization
print("\n--- Final Clusters K-Means in action ---")
plt.figure(figsize=(10, 6))
colors = {1: 'green', 2: 'orange', 3: 'purple', 4: 'blue', 5: 'red'}

for pt in dataset:
    plt.scatter(pt["x"], pt["y"], color=colors[pt["cluster"]], s=100, alpha=0.7)

for c in centroids:
    plt.scatter(c["x"], c["y"], color='yellow', marker='o', s=400, edgecolor='black', zorder=3, label=f"Centroid {c['id']}")

plt.title("K-Means Clustering (K=5, Unbiased)")
plt.xlabel("Annual Income ($k)")
plt.ylabel("Spending Score (1-100)")
plt.grid(True, linestyle='--', alpha=0.7)

handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())

plt.tight_layout()
plt.savefig(r"d:\Desktop\kn\kmeans_visualization.png", dpi=300, bbox_inches='tight')
print("\nVisualization saved to: d:\\Desktop\\kn\\kmeans_visualization.png")
print("Tables and steps saved to: d:\\Desktop\\kn\\kmeans_steps_output.csv")
plt.show()
