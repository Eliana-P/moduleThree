import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns

cleaned_directory = '/Users/elianapritchard/Documents/INST414 WORK/cleaned_reddit_data' 
cleaned_file_list = [f for f in os.listdir(cleaned_directory) if f.endswith('.csv')]

custom_stop_words = [
    'just', 'like', 'time', 'people', 'know', 'says', 'said', 
    'man', 'new', 'years', 'don', 'youre', 'dont', 'oc', 
    'game', 'made', 'get', 'see', 'using', 'way', 'make'
]

combined_stop_words = list(ENGLISH_STOP_WORDS.union(custom_stop_words))

query_subreddits = ['technology', 'science', 'dataisbeautiful'] 

all_texts = []
subreddit_names = []  
subreddit_data = {}

for file in cleaned_file_list:
    subreddit_name = file.split('.')[0]  
    
    data = pd.read_csv(os.path.join(cleaned_directory, file))
    
    data['text_combined'] = data['title'].fillna('') + " " + data['body'].fillna('')
    
    all_texts.append(data['text_combined'].str.cat(sep=' '))  
    subreddit_names.append(subreddit_name) 
    
    subreddit_data[subreddit_name] = data

vectorizer = TfidfVectorizer(stop_words=combined_stop_words)
tfidf_matrix = vectorizer.fit_transform(all_texts)

cosine_sim = cosine_similarity(tfidf_matrix)

cosine_sim_df = pd.DataFrame(cosine_sim, index=subreddit_names, columns=subreddit_names)

top_subreddit_names = []
top_similarities = []
query_labels = []

for query in query_subreddits:
    print(f"\n### Top 10 Most Similar Subreddits to {query} ###")
    
    query_idx = subreddit_names.index(query)
    
    similarity_scores = cosine_sim[query_idx]
    
    sorted_indices = similarity_scores.argsort()[::-1]
    
    top_10_subreddits = [(subreddit_names[i], similarity_scores[i]) for i in sorted_indices if subreddit_names[i] != query][:10]
    
    for idx, (subreddit, score) in enumerate(top_10_subreddits):
        print(f"{idx+1}. {subreddit}: {score:.4f}")
    
    for subreddit, score in top_10_subreddits:
        top_subreddit_names.append(subreddit)
        top_similarities.append(score)
        query_labels.append(query)  

top_data = pd.DataFrame({
    'Subreddit': top_subreddit_names,
    'Cosine Similarity': top_similarities,
    'Query Subreddit': query_labels
})

plt.figure(figsize=(12, 8))
sns.barplot(x='Subreddit', y='Cosine Similarity', hue='Query Subreddit', data=top_data, palette="Set2")

plt.title("Top 10 Most Similar Subreddits (for Each Query Subreddit)", fontsize=16)
plt.xlabel('Subreddit Name', fontsize=12)
plt.ylabel('Cosine Similarity', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
