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

def get_collaborative_recommendations(user_id, df, cosine_sim, top_n=5):
    import numpy as np
    engine = create_engine(DATABASE_URL)
    query = f"SELECT movie_id, seconds_watched, completed FROM watch_history WHERE user_id = {user_id}"
    history_df = pd.read_sql(query, engine)
    
    if history_df.empty:
        return None
        
    user_profile_scores = np.zeros(len(df))
    watched_movie_ids = history_df['movie_id'].tolist()
    
    for _, row in history_df.iterrows():
        movie_id = row['movie_id']
        seconds = row['seconds_watched'] if pd.notna(row['seconds_watched']) else 0
        completed = row['completed']
        
        weight = 0.5 # Baseline weight for opening a movie
        if completed:
            weight = 2.0
        elif seconds > 0:
            # Scale seconds watched (assume max reasonable seconds is ~7200 for 2 hours)
            # Normalizing so 7200 seconds -> ~1.5 weight
            weight = 0.5 + min((seconds / 7200.0), 1.0)
            
        try:
            # Find index of movie in the dataframe
            idx = df.index[df['id'] == movie_id][0]
            # Add weighted similarity scores
            user_profile_scores += weight * cosine_sim[idx]
        except IndexError:
            continue
            
    # Zero out scores for already watched movies so we don't recommend them again
    for movie_id in watched_movie_ids:
        try:
            idx = df.index[df['id'] == movie_id][0]
            user_profile_scores[idx] = 0.0
        except IndexError:
            pass
            
    # Get top N indices with highest score
    if user_profile_scores.max() == 0:
        return None
        
    top_indices = user_profile_scores.argsort()[::-1][:top_n]
    
    recommendations = []
    for i in top_indices:
        score = float(user_profile_scores[i])
        if score > 0:
            recommendations.append({
                "movie_id": int(df['id'].iloc[i]),
                "score": round(score, 3), # Round score for readability
                "title": df['title'].iloc[i]
            })
            
    return recommendations

def get_youtube_like_feed(user_id, df, cosine_sim):
    import numpy as np
    engine = create_engine(DATABASE_URL)
    # Order by id DESC assumes higher ID means more recently added to history
    query = f"SELECT movie_id, seconds_watched, completed, id as watch_id FROM watch_history WHERE user_id = {user_id} ORDER BY id DESC"
    history_df = pd.read_sql(query, engine)
    
    feed = {
        "continue_watching": [],
        "up_next": None,
        "because_you_watched": None,
        "top_picks_for_you": [],
        "trending": get_popular_fallback(df, top_n=10)
    }
    
    if history_df.empty:
        feed["top_picks_for_you"] = feed["trending"][:5]
        return feed
        
    watched_movie_ids = history_df['movie_id'].tolist()
    
    # 1. Continue Watching (Недосмотренные фильмы)
    in_progress = history_df[history_df['completed'] == False]
    for _, row in in_progress.head(5).iterrows():
        try:
            m_idx = df.index[df['id'] == row['movie_id']][0]
            feed["continue_watching"].append({
                "movie_id": int(row['movie_id']),
                "title": df['title'].iloc[m_idx],
                "seconds_watched": int(row['seconds_watched']) if pd.notna(row['seconds_watched']) else 0
            })
        except IndexError:
            pass

    # 2. Up Next (Смотреть дальше - продолжение последнего просмотренного кино)
    # Consider "completed" OR watched more than 90% of a standard movie length (let's say > 6000 seconds as proxy for now without actual duration)
    # Or simply: order by ID and get the latest one that is completed OR has a high watch time
    latest_watched = history_df.head(1)
    if not latest_watched.empty:
        last_row = latest_watched.iloc[0]
        # Treat as completed if flag is True OR if watched more than 12 minutes (720 seconds)
        is_essentially_completed = last_row['completed'] or (pd.notna(last_row['seconds_watched']) and last_row['seconds_watched'] > 720)
        
        if is_essentially_completed:
            last_completed_id = last_row['movie_id']
            try:
                last_idx = df.index[df['id'] == last_completed_id][0]
                last_title = df['title'].iloc[last_idx]
                
                # Используем косинусное сходство + бонус за совпадение префикса (например, Властелин Колец 1 -> Властелин Колец 2)
                sim_scores = cosine_sim[last_idx].copy()
                for w_id in watched_movie_ids:
                    try:
                        w_idx = df.index[df['id'] == w_id][0]
                        sim_scores[w_idx] = -1.0 # Исключаем уже просмотренные
                    except IndexError:
                        pass
                    
                # Ищем сиквелы по первой части названия (до двоеточия или первого слова, если длинное)
                base_prefix = last_title.split(":")[0] if ":" in last_title else " ".join(last_title.split(" ")[:2])
                
                for i in range(len(df)):
                    title = df['title'].iloc[i]
                    if title != last_title and title.startswith(base_prefix):
                        sim_scores[i] += 0.5 # Огромный бонус за то же самое начало названия (потенциальный сиквел)
                        
                top_next_idx = sim_scores.argmax()
                if sim_scores[top_next_idx] > 0:
                    feed["up_next"] = {
                        "reason": f"Продолжить просмотр франшизы / логический следующий шаг после «{last_title}»",
                        "recommendation": {
                            "movie_id": int(df['id'].iloc[top_next_idx]),
                            "title": df['title'].iloc[top_next_idx],
                            "score": round(float(sim_scores[top_next_idx]), 3)
                        }
                    }
            except IndexError:
                pass

    # 3. Because you watched (Content-based рекомендация на основе случайного ранее просмотренного фильма)
    # Берем случайный фильм из истории, чтобы лента была разнообразной
    random_history_movie = history_df.sample(n=1).iloc[0]['movie_id']
    try:
        r_idx = df.index[df['id'] == random_history_movie][0]
        r_title = df['title'].iloc[r_idx]
        
        # Получаем рекомендации, исключая уже просмотренное
        raw_recs = get_content_recommendations(random_history_movie, df, cosine_sim, top_n=15)
        if raw_recs:
            filtered_recs = [r for r in raw_recs if r['movie_id'] not in watched_movie_ids][:5]
            if filtered_recs:
                feed["because_you_watched"] = {
                    "reason": f"Похожее на то, что вы смотрели: «{r_title}»",
                    "recommendations": filtered_recs
                }
    except Exception:
        pass
        
    # 4. Top Picks For You (Глубокий анализ по всей истории - коллаборативный/исторический метод)
    top_picks = get_collaborative_recommendations(user_id, df, cosine_sim, top_n=10)
    if top_picks:
        # Убираем дубликаты, если они уже есть в up_next или trending
        used_ids = set()
        if feed["up_next"]:
            used_ids.add(feed["up_next"]["recommendation"]["movie_id"])
            
        clean_top_picks = []
        for pick in top_picks:
            if pick["movie_id"] not in used_ids:
                clean_top_picks.append(pick)
                used_ids.add(pick["movie_id"])
                
        feed["top_picks_for_you"] = clean_top_picks[:5]
        
    return feed