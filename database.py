from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, declarative_base
import logging

DB_USER, DB_PASS, DB_HOST, DB_PORT, DB_NAME = "test_user", "pass1", "localhost", "5432", "testdb"
DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def check_db_connection():
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
            return True
    except:
        return False