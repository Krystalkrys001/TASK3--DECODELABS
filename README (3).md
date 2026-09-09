# Project 3: Unsupervised Learning, Customer Segmentation
**DecodeLabs Data Science Industrial Training, 2026 Batch**

## Overview

This project groups 1,000 retail customers into distinct, actionable segments using no labels at all, just raw demographic and purchasing behavior data. The pipeline covers scaling, dimensionality reduction (PCA), clustering (K-Means), mathematically justifying the number of clusters, and translating the result into real business personas.

**Note on the dataset:** DecodeLabs did not provide a dataset for this project. A synthetic 1,000-row, 26-column retail customer dataset was generated instead, built with 4 realistic underlying customer patterns so the clustering step would have genuine, discoverable structure, rather than random noise. This is disclosed here for transparency; it is not scraped or real company data.

## Objectives

- Reduce 33 encoded features down to 2 dimensions using PCA
- Mathematically justify the number of clusters using the Elbow Method and Silhouette Score, rather than guessing
- Translate the resulting clusters into named, actionable business personas

## 1. Preparing the Data

`CustomerID` was dropped (a label, not a real characteristic), and the two text columns (`Education`, `Marital_Status`) were one-hot encoded, expanding the dataset to 33 numeric columns. All 33 were then scaled with `StandardScaler`, since K-Means and PCA both rely on distance calculations that break down when features sit on wildly different scales (e.g. `Income` in the tens of thousands vs. a 0-2 `Kidhome` count).

## 2. Dimensionality Reduction (PCA)

| Component | Variance Explained |
|---|---|
| Component 1 | 19.59% |
| Component 2 | 4.55% |
| **Total (2 components)** | **24.14%** |

Only 24% of the original variance is retained in 2 dimensions, this is an expected trade-off, not a failure. Across 33 spread-out features, no 2 components were ever going to capture the majority of the variety. The 2D reduction exists purely to make clustering and visualization tractable; the final personas are built from the real, original-unit columns, not the compressed PCA values.

## 3. Choosing K: Elbow Method vs. Silhouette Score

| K | WCSS | Silhouette Score |
|---|---|---|
| 2 | 2097.2 | **0.6449** |
| 3 | 1470.3 | 0.5339 |
| 4 | 1085.3 | 0.3945 |
| 5 | 908.3 | 0.3639 |
| 6 | 786.7 | 0.3769 |

**The honest tension:** Silhouette Score mathematically favors K=2, the most cleanly separated split possible. But two giant customer buckets aren't an actionable marketing strategy. The Elbow curve's steepest gains flatten out around K=4, and K=4 produces four genuinely distinct, actionable personas.

**Decision: K=4.** The most statistically "clean" answer isn't automatically the most useful one. Four segments give a marketing team enough granularity to design different campaigns per group; two segments would not.

Final model's Silhouette Score at K=4: **0.3945**, a reasonable, moderate separation, consistent with real-world customer data rarely falling into perfectly distinct boxes.

## 4. The Four Personas

| Persona | Age | Income | Wine Spend | Web Purchases | Store Purchases | Complaint Rate |
|---|---|---|---|---|---|---|
| **Young Budget Shoppers** | 29.7 | $26,936 | $265.70 | 2.9 | 4.4 | 0% |
| **Young Affluent High-Engagers** | 33.5 | $84,940 | $673.90 | 7.1 | 10.7 | 0% |
| **Older Budget-Conscious** | 39.8 | $26,423 | $214.00 | 2.2 | 3.4 | 10% |
| **Established Affluent Spenders** | 40.7 | $88,440 | $820.80 | 5.4 | 7.9 | 0% |

- **Young Budget Shoppers**: youngest, lowest income, modest spending, zero complaints, prefers store over web.
- **Young Affluent High-Engagers**: high income, big spenders, heaviest activity across both web and store, the most valuable young segment.
- **Older Budget-Conscious**: lowest engagement across every metric, and the only segment with a meaningful complaint rate (10%), a real retention risk worth flagging to the business.
- **Established Affluent Spenders**: oldest, highest income, highest spend on wine and meat, but noticeably less web-active than the younger affluent segment, more of a traditional in-store shopper.

## Tech Stack

Python, Pandas, Scikit-learn (StandardScaler, PCA, KMeans, silhouette_score), Matplotlib

## Files

- `project3_customer_segmentation.py` — full pipeline: load, encode, scale, PCA, K selection, clustering, personas
- `customer_segmentation_data.csv` — synthetic input dataset
- `customer_segmentation_with_clusters.csv` — final output with cluster labels

## Key Takeaway

The Silhouette Score gave a clear, unambiguous mathematical answer, and it wasn't the one that was actually useful. Recognizing when to prioritize business actionability over the single "best" statistical metric was the real decision point of this project, not the clustering code itself.
