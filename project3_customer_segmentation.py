# PROJECT 3: Unsupervised Learning, Customer Segmentation
# DecodeLabs Data Science Industrial Training, 2026 Batch

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# STEP 1: Load and prepare the data
df = pd.read_csv("customer_segmentation_data.csv")

df_features = df.drop(columns=["CustomerID"])  # ID is a label, not a real characteristic

# One-hot encode the two text columns so K-Means can use them numerically
df_encoded = pd.get_dummies(df_features, columns=["Education", "Marital_Status"])
print(df_encoded.shape)   # (1000, 33)


# STEP 2: Scale, then reduce with PCA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_encoded)
df_scaled = pd.DataFrame(X_scaled, columns=df_encoded.columns)

pca = PCA(n_components=2, random_state=42)
pca_result = pca.fit_transform(df_scaled)

for i, var in enumerate(pca.explained_variance_ratio_, start=1):
    print(f"Component {i}: {var*100:.2f}%")
print(f"Total variance retained: {pca.explained_variance_ratio_.sum()*100:.2f}%")
# Component 1: 19.59% | Component 2: 4.55% | Total: 24.14%


# STEP 3: Find the right K (Elbow Method + Silhouette Score)
wcss = []
silhouette_scores = []
k_range = range(2, 11)

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(pca_result)
    wcss.append(km.inertia_)
    silhouette_scores.append(silhouette_score(pca_result, labels))
    print(f"K={k}: WCSS={km.inertia_:.1f}, Silhouette={silhouette_scores[-1]:.4f}")

# RESULT: Silhouette Score mathematically favors K=2 (0.6449), the cleanest
# possible split. But 2 clusters is not actionable for a marketing team.
# The Elbow curve shows the steepest gains flattening out around K=4, and
# K=4 aligns with 4 genuinely distinct, actionable customer personas.
# DECISION: K=4 chosen over the statistically "cleanest" K=2, prioritizing
# business usefulness over the single best silhouette number.


# STEP 4: Fit final model with K=4
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(pca_result)
print("Final Silhouette Score (K=4):", silhouette_score(pca_result, cluster_labels))

df["Cluster"] = cluster_labels


# STEP 5: Translate clusters into personas
# using REAL, original-unit columns, not PCA components
persona_cols = ["Age","Income","Recency","MntWines","MntFruits","MntMeatProducts",
                 "MntFishProducts","MntSweetProducts","MntGoldProds",
                 "NumWebPurchases","NumCatalogPurchases","NumStorePurchases",
                 "NumWebVisitsMonth","Complain"]

persona_table = df.groupby("Cluster")[persona_cols].mean().round(1)
persona_table["Count"] = df["Cluster"].value_counts().sort_index()
print(persona_table.to_string())

df.to_csv("customer_segmentation_with_clusters.csv", index=False)
print("Saved final labeled dataset.")


# FINAL PERSONAS
# ===
# Cluster 0, "Young Budget Shoppers": youngest, lowest income, modest
#   spending, zero complaints, store-preferred over web.
# Cluster 1, "Young Affluent High-Engagers": high income, big spenders,
#   heaviest web AND store activity, most valuable young segment.
# Cluster 2, "Older Budget-Conscious": lowest engagement across the
#   board, and the only group with a real complaint rate (10%),
#   a retention risk worth flagging.
# Cluster 3, "Established Affluent Spenders": oldest, highest income,
#   highest spend on wine and meat, less web-active than Cluster 1,
#   a traditional in-store shopper.
