import os
import sys
import pandas as pd
from pathlib import Path
from sqlalchemy.orm import Session

# Thêm thư mục gốc backend vào PYTHONPATH
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.db import engine, SessionLocal, Base
from app.models.movie import Movie

def seed_movies():
    print("Starting to seed movies...")
    # Tạo bảng nếu chưa có
    Base.metadata.create_all(bind=engine)
    
    csv_path = Path(__file__).parent.parent / "CinematchAi" / "data" / "processed" / "movies.csv"
    if not csv_path.exists():
        print(f"Error: Could not find file {csv_path}")
        print("Please run CinematchAi data pipeline first.")
        return

    df = pd.read_csv(csv_path)
    
    db: Session = SessionLocal()
    
    # Kiểm tra xem đã có dữ liệu chưa
    existing_count = db.query(Movie).count()
    if existing_count > 0:
        print(f"Database already has {existing_count} movies. Skipping...")
        # db.query(Movie).delete()
        # db.commit()
        db.close()
        return

    movies_to_insert = []
    
    # Xử lý các cột thể loại
    genre_columns = [
        "Action", "Adventure", "Animation", "Children", "Comedy", 
        "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", 
        "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", 
        "Thriller", "War", "Western", "unknown"
    ]
    
    for _, row in df.iterrows():
        # Tạo chuỗi genres ví dụ "Action|Comedy"
        genres_list = [g for g in genre_columns if g in df.columns and row.get(g, 0) == 1]
        genres_str = "|".join(genres_list) if genres_list else None
        
        # Xử lý nullable values
        release_year = int(row["release_year"]) if pd.notna(row.get("release_year")) else None
        imdb_url = str(row["imdb_url"]) if pd.notna(row.get("imdb_url")) else None

        movie = Movie(
            movielens_id=int(row["movie_id"]),
            title=str(row["title"]),
            genres=genres_str,
            release_year=release_year,
            imdb_url=imdb_url
        )
        movies_to_insert.append(movie)
        
    db.bulk_save_objects(movies_to_insert)
    db.commit()
    print(f"Successfully seeded {len(movies_to_insert)} movies into database.")
    db.close()

if __name__ == "__main__":
    seed_movies()
