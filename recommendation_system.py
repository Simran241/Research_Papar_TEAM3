import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

# ============================================================
# NOVELTY 1: Real structured dataset (not random)
# Simulates Amazon-style product ratings (1-5 scale)
# ============================================================
np.random.seed(42)

users = [f'U{i}' for i in range(1, 21)]   # 20 users
products = ['Laptop', 'Phone', 'Headphones', 'Tablet', 'Camera',
            'Smartwatch', 'Speaker', 'Monitor', 'Keyboard', 'Mouse']

# Simulate user personas to make data realistic
tech_lovers     = users[:5]     # Rate Laptop, Monitor, Keyboard high
audio_fans      = users[5:10]   # Rate Headphones, Speaker high
mobile_users    = users[10:15]  # Rate Phone, Tablet high
casual_users    = users[15:]    # Mixed ratings

data = []
for user in users:
    for product in products:
        if user in tech_lovers:
            base = 4 if product in ['Laptop', 'Monitor', 'Keyboard', 'Mouse'] else 2
        elif user in audio_fans:
            base = 5 if product in ['Headphones', 'Speaker'] else 2
        elif user in mobile_users:
            base = 4 if product in ['Phone', 'Tablet', 'Smartwatch'] else 2
        else:
            base = 3  # casual users rate everything avg
        
        # Add some noise
        rating = min(5, max(1, base + np.random.randint(-1, 2)))
        data.append([user, product, rating])

df = pd.DataFrame(data, columns=['user', 'product', 'rating'])
print("Dataset Preview:")
print(df.head(10))
print(f"\nTotal ratings: {len(df)}")

# ============================================================
# NOVELTY 2: Train/Test Split for Evaluation
# ============================================================
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

user_item = train_df.pivot_table(
    index='user', columns='product', values='rating'
).fillna(0)

print("\nUser-Item Matrix Shape:", user_item.shape)
print(user_item.head(5))

# ============================================================
# Collaborative Filtering (User-Based)
# ============================================================
user_similarity = cosine_similarity(user_item)
user_similarity_df = pd.DataFrame(
    user_similarity,
    index=user_item.index,
    columns=user_item.index
)

def collaborative_recommend(user, top_n=5):
    """Recommend products based on most similar user's preferences."""
    if user not in user_similarity_df.index:
        return popularity_fallback(top_n)  # NOVELTY 3: Cold-Start
    similar_users = user_similarity_df[user].sort_values(ascending=False)[1:]
    top_user = similar_users.index[0]
    return user_item.loc[top_user].sort_values(ascending=False).head(top_n)

# ============================================================
# Content-Based Filtering (Product Features)
# ============================================================
product_features = {
    'Laptop':     [1, 0, 0, 1, 0, 1, 0, 1, 0, 0],
    'Phone':      [0, 1, 0, 1, 0, 1, 0, 0, 0, 0],
    'Headphones': [0, 0, 1, 0, 1, 0, 1, 0, 0, 0],
    'Tablet':     [1, 1, 0, 0, 0, 1, 0, 0, 0, 0],
    'Camera':     [0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
    'Smartwatch': [0, 1, 0, 1, 0, 1, 0, 0, 0, 0],
    'Speaker':    [0, 0, 1, 0, 1, 0, 1, 0, 0, 0],
    'Monitor':    [1, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    'Keyboard':   [1, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    'Mouse':      [1, 0, 0, 0, 0, 0, 0, 0, 1, 0],
}
# Feature labels: [computing, mobile, audio, portable, imaging, wireless, bass, display, peripheral, optical]

feature_df = pd.DataFrame(product_features).T
item_similarity = cosine_similarity(feature_df)
item_similarity_df = pd.DataFrame(
    item_similarity,
    index=feature_df.index,
    columns=feature_df.index
)

print("\nItem Similarity Matrix:")
print(item_similarity_df.round(2))

def content_recommend(product, top_n=5):
    """Recommend similar products based on feature vectors."""
    return item_similarity_df[product].sort_values(ascending=False)[1:top_n+1]

# ============================================================
# NOVELTY 4: Weighted Hybrid Recommendation
# Alpha = weight for collaborative, Beta = weight for content-based
# Alpha is higher when user has more ratings (active user)
# ============================================================
def hybrid_recommend(user, product, top_n=5):
    """
    Weighted hybrid: combines collaborative + content-based scores.
    Alpha is dynamically adjusted based on user activity level.
    """
    user_rating_count = train_df[train_df['user'] == user].shape[0]
    
    # Dynamic weighting: active users get more collaborative weight
    if user_rating_count >= 8:
        alpha = 0.7  # Trust collaborative more
    elif user_rating_count >= 4:
        alpha = 0.5  # Balance both
    else:
        alpha = 0.3  # Trust content more (sparse user)
    
    beta = 1 - alpha
    
    # Get collaborative scores
    collab_scores = collaborative_recommend(user, top_n=len(products))
    collab_norm = collab_scores / collab_scores.max() if collab_scores.max() > 0 else collab_scores
    
    # Get content scores
    if product in item_similarity_df.columns:
        content_scores = item_similarity_df[product].drop(product)
    else:
        content_scores = pd.Series(0, index=products)
    
    # Align indexes
    all_products = list(set(collab_norm.index) | set(content_scores.index))
    collab_aligned = collab_norm.reindex(all_products, fill_value=0)
    content_aligned = content_scores.reindex(all_products, fill_value=0)
    
    # Weighted combination
    hybrid_score = alpha * collab_aligned + beta * content_aligned
    hybrid_score = hybrid_score.sort_values(ascending=False).head(top_n)
    
    print(f"\n{'='*50}")
    print(f"Hybrid Recommendation for {user} (viewing: {product})")
    print(f"User activity: {user_rating_count} ratings | α(collab)={alpha}, β(content)={beta}")
    print(f"{'='*50}")
    print(hybrid_score.round(4))
    return hybrid_score

# ============================================================
# NOVELTY 3: Cold-Start Fallback (Popularity-Based)
# ============================================================
def popularity_fallback(top_n=5):
    """For new users: recommend most popular products by avg rating."""
    popular = df.groupby('product')['rating'].mean().sort_values(ascending=False).head(top_n)
    print("(Cold-Start) Popular Products:")
    print(popular)
    return popular

# ============================================================
# NOVELTY 5: Evaluation Metrics (Precision@K, Recall@K, NDCG@K)
# ============================================================
def precision_at_k(recommended, relevant, k=5):
    """What fraction of top-K recommendations are relevant?"""
    rec_k = list(recommended[:k])
    hits = len(set(rec_k) & set(relevant))
    return hits / k

def recall_at_k(recommended, relevant, k=5):
    """What fraction of relevant items are in top-K?"""
    rec_k = list(recommended[:k])
    hits = len(set(rec_k) & set(relevant))
    return hits / len(relevant) if relevant else 0

def ndcg_at_k(recommended, relevant, k=5):
    """Normalized Discounted Cumulative Gain at K."""
    dcg = 0
    for i, item in enumerate(list(recommended[:k])):
        if item in relevant:
            dcg += 1 / np.log2(i + 2)  # i+2 because log2(1)=0
    idcg = sum(1 / np.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / idcg if idcg > 0 else 0

def evaluate_system(test_df, threshold=3.5, k=5):
    """Evaluate recommendation quality on test set."""
    precisions, recalls, ndcgs = [], [], []
    
    test_users = test_df['user'].unique()
    for user in test_users:
        # Ground truth: products rated > threshold in test set
        relevant = test_df[
            (test_df['user'] == user) & (test_df['rating'] >= threshold)
        ]['product'].tolist()
        
        if not relevant:
            continue
        
        # Get recommendations
        collab = collaborative_recommend(user, top_n=k)
        recommended = list(collab.index)
        
        precisions.append(precision_at_k(recommended, relevant, k))
        recalls.append(recall_at_k(recommended, relevant, k))
        ndcgs.append(ndcg_at_k(recommended, relevant, k))
    
    results = {
        'Precision@5': round(np.mean(precisions), 4),
        'Recall@5':    round(np.mean(recalls), 4),
        'NDCG@5':      round(np.mean(ndcgs), 4),
        'F1@5':        round(
            2 * np.mean(precisions) * np.mean(recalls) /
            (np.mean(precisions) + np.mean(recalls) + 1e-9), 4
        )
    }
    return results

# ============================================================
# RUN EVERYTHING
# ============================================================
print("\n--- Collaborative Recommendation for U1 ---")
print(collaborative_recommend('U1'))

print("\n--- Content-Based: Similar to Laptop ---")
print(content_recommend('Laptop'))

hybrid_recommend('U1', 'Laptop')
hybrid_recommend('U6', 'Headphones')

# Cold-start test
print("\n--- Cold-Start (New User) ---")
popularity_fallback()

# Evaluation
print("\n--- System Evaluation Metrics ---")
metrics = evaluate_system(test_df)
for metric, value in metrics.items():
    print(f"  {metric}: {value}")
