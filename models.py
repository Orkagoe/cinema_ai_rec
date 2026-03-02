from sqlalchemy import Column, BigInteger, String, Text, ForeignKey, SmallInteger, Integer, Table, Boolean
from sqlalchemy.orm import relationship
from database import Base

movie_genres = Table(
    "movie_genres",
    Base.metadata,
    Column("movie_id", BigInteger, ForeignKey("movies.id"), primary_key=True),
    Column("genre_id", BigInteger, ForeignKey("genres.id"), primary_key=True),
)

class User(Base):
    __tablename__ = "users"
    id = Column(BigInteger, primary_key=True)
    username = Column(String(50))
    email = Column(String(255))
    ratings = relationship("Rating", back_populates="user")
    watch_history = relationship("WatchHistory", back_populates="user")

class Movie(Base):
    __tablename__ = "movies"
    id = Column(BigInteger, primary_key=True)
    title = Column(String(255))
    genre_text = Column(String(255))
    plot = Column(Text)
    description = Column(String(2000))
    imdb_rating = Column(String(255))
    genres = relationship("Genre", secondary=movie_genres)
    ratings = relationship("Rating", back_populates="movie")

class Genre(Base):
    __tablename__ = "genres"
    id = Column(BigInteger, primary_key=True)
    name = Column(String(255))

class Rating(Base):
    __tablename__ = "ratings"
    id = Column(BigInteger, primary_key=True)
    user_id = Column(BigInteger, ForeignKey("users.id"))
    movie_id = Column(BigInteger, ForeignKey("movies.id"))
    score = Column(SmallInteger)
    user = relationship("User", back_populates="ratings")
    movie = relationship("Movie", back_populates="ratings")

class WatchHistory(Base):
    __tablename__ = "watch_history"
    id = Column(BigInteger, primary_key=True)
    user_id = Column(BigInteger, ForeignKey("users.id"))
    movie_id = Column(BigInteger, ForeignKey("movies.id"))
    seconds_watched = Column(Integer)
    completed = Column(Boolean)
    user = relationship("User", back_populates="watch_history")