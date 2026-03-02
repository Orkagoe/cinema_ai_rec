import pandas as pd
from sqlalchemy import create_engine
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from database import DATABASE_URL

def get_recommendations_model():
    engine = create_engine(DATABASE_URL)
    query = "SELECT id, title, genre_text, description, plot, imdb_rating FROM movies"
    df = pd.read_sql(query, engine)
    df['content'] = (df['genre_text'].fillna('') + ' ' + 
                     df['plot'].fillna('') + ' ' + 
                     df['description'].fillna(''))
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['content'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return df, cosine_sim

def get_content_recommendations(movie_id, df, cosine_sim, top_n=5):
    try:
        if movie_id not in df['id'].values:
            return None
        idx = df.index[df['id'] == movie_id][0]
        sim_scores = sorted(list(enumerate(cosine_sim[idx])), key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:top_n+1]
        return [
            {"movie_id": int(df['id'].iloc[i]), "score": float(s), "title": df['title'].iloc[i]}
            for i, s in sim_scores
        ]
    except:
        return None

def get_popular_fallback(df, top_n=5):
    df['rating_num'] = pd.to_numeric(df['imdb_rating'], errors='coerce').fillna(0)
    popular = df.sort_values(by='rating_num', ascending=False).head(top_n)
    return [
        {"movie_id": int(row['id']), "score": float(row['rating_num']), "title": row['title']}
        for _, row in popular.iterrows()
    ]