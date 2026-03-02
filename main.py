from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from typing import List, Dict, Any

from database import SessionLocal
from models import Movie, User, Rating, WatchHistory
import recommender

app = FastAPI(title="Cinema AI Service", version="2.0.0")

ML_MODEL = {
    "df": None,
    "similarity": None
}

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def load_models():
    try:
        df, sim = recommender.get_recommendations_model()
        ML_MODEL["df"] = df
        ML_MODEL["similarity"] = sim
    except Exception:
        pass

@app.on_event("startup")
async def startup_event():
    load_models()

@app.post("/api/v1/ml/retrain")
async def retrain_models(background_tasks: BackgroundTasks):
    background_tasks.add_task(load_models)
    return {
        "_comment_action": "Процесс переобучения запущен в фоновом режиме", 
        "status": "ok"
    }

@app.get("/api/v1/stats")
async def get_ml_stats(db: Session = Depends(get_db)):
    return {
        "_comment_stats": "Общая статистика по всем таблицам базы данных",
        "movies": db.query(Movie).count(),
        "users": db.query(User).count(),
        "ratings": db.query(Rating).count(),
        "history": db.query(WatchHistory).count()
    }

@app.get("/api/v1/data/movies")
async def get_all_movies(db: Session = Depends(get_db)):
    return {
        "_comment_movies": "Таблица: movies. Содержит метаданные для Content-Based рекомендаций.",
        "movies": db.query(Movie).all()
    }

@app.get("/api/v1/data/ratings")
async def get_all_ratings(db: Session = Depends(get_db)):
    return {
        "_comment_ratings": "Таблица: ratings. Оценки юзеров (user_id -> movie_id). Основа для Collaborative Filtering.",
        "ratings": db.query(Rating).all()
    }

@app.get("/api/v1/data/history")
async def get_all_history(db: Session = Depends(get_db)):
    return {
        "_comment_history": "Таблица: watch_history. Сюда будут падать секунды просмотра.",
        "history": db.query(WatchHistory).all()
    }

@app.get("/api/v1/recommend/content/{movie_id}")
async def recommend_content(movie_id: int, limit: int = 5):
    if ML_MODEL["df"] is None:
        raise HTTPException(status_code=503, detail="Модель обучается")
    
    recs = recommender.get_content_recommendations(
        movie_id, ML_MODEL["df"], ML_MODEL["similarity"], limit
    )
    
    if recs is None:
        return {
            "_comment_response": "Фильм не найден, возвращаем популярные",
            "movie_id": movie_id,
            "recommendations": recommender.get_popular_fallback(ML_MODEL["df"], limit),
            "method": "popular_fallback"
        }
    
    return {
        "_comment_response": "Успешные рекомендации на основе TF-IDF",
        "movie_id": movie_id,
        "recommendations": recs,
        "method": "content_based_similarity"
    }

@app.get("/api/v1/recommend/collaborative/{user_id}")
async def recommend_collaborative(user_id: int, limit: int = 5):
    if ML_MODEL["df"] is None:
        raise HTTPException(status_code=503, detail="Модель обучается")
        
    try:
        recs = recommender.get_collaborative_recommendations(user_id, limit)
        if not recs:
            return {
                "_comment_response": "У юзера нет оценок, возвращаем популярные (Cold Start)",
                "user_id": user_id,
                "recommendations": recommender.get_popular_fallback(ML_MODEL["df"], limit),
                "method": "popular_fallback_cold_start"
            }
        return {
            "_comment_response": "Персональные рекомендации на основе оценок похожих юзеров",
            "user_id": user_id,
            "recommendations": recs,
            "method": "collaborative_filtering"
        }
    except AttributeError:
        return {
            "_comment_response": "Метод коллаборативной фильтрации еще не реализован в recommender.py",
            "user_id": user_id,
            "recommendations": recommender.get_popular_fallback(ML_MODEL["df"], limit),
            "method": "collaborative_not_implemented"
        }

@app.get("/api/v1/movie/details/{movie_id}")
async def get_movie_details(movie_id: int, db: Session = Depends(get_db)):
    movie = db.query(Movie).filter(Movie.id == movie_id).first()
    if not movie:
        raise HTTPException(status_code=404, detail="Фильм не найден")
    
    return {
        "_comment_details": f"Расширенная информация для фильма с ID {movie_id}",
        "id": movie.id,
        "title": movie.title,
        "description": movie.description,
        "genres": movie.genre_text,
        "imdb_rating": movie.imdb_rating,
        "stats": {
            "ratings_count": len(movie.ratings),
            "avg_score": sum([r.score for r in movie.ratings]) / len(movie.ratings) if movie.ratings else 0
        }
    }