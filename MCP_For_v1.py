from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict
import os
from neo4j import GraphDatabase
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = FastAPI()

NEO4J_URI = os.environ.get("NEO4J_URI")
NEO4J_USER = os.environ.get("NEO4J_USER")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD")
DATABASE_MOVIE = "moviesandseries"
DATABASE_PLACE = "neo4j"

# ---------------- Neo4j Connector ----------------
class Neo4jConnector:
    def __init__(self, db_name="neo4j"):
        self.uri = NEO4J_URI
        self.user = NEO4J_USER
        self.password = NEO4J_PASSWORD
        self.database = db_name
        self.driver = None

    def connect(self):
        if self.driver is None:
            try:
                self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
                self.driver.verify_connectivity()
            except Exception as e:
                raise RuntimeError(f"Neo4j veritabanına bağlanılamadı. Hata: {e}")

    def fetch(self, query):
        self.connect()
        with self.driver.session(database=self.database) as session:
            result = session.run(query)
            return [dict(record) for record in result]
            
    def count_nodes(self, label):
        self.connect()
        query = f"MATCH (n:{label}) RETURN count(n) AS count"
        with self.driver.session(database=self.database) as session:
            result = session.run(query).single()
            return result["count"] if result else 0

    def close(self):
        if self.driver:
            self.driver.close()
            self.driver = None

# Pydantic Modelleri
class Movie(BaseModel):
    title: str
    genres: List[str]
    imdb_rating: Optional[float]
    poster: Optional[str]
    overview: Optional[str]

class Place(BaseModel):
    name: str
    address: Optional[str]
    rating: Optional[float]
    maps_link: Optional[str]

class ReservationRequest(BaseModel):
    place_name: str
    date: str
    time: str

class ReservationStatus(BaseModel):
    is_available: bool
    message: str

class MovieTicket(BaseModel):
    cinema: str
    time: str
    link: str

class StreamingLink(BaseModel):
    platform: str
    link: str

# ----------------- Tool Fonksiyonları -----------------

def get_movies_from_db():
    query = """
    MATCH (m:Movie)
    OPTIONAL MATCH (m)-[:IN_GENRE]->(g:Genre)
    RETURN 
        m.title AS title,
        m.imdb_rating AS imdb_rating,
        m.poster AS poster,
        m.overview AS overview,
        collect(g.name) AS genres
    LIMIT 200
    """
    neo = Neo4jConnector(DATABASE_MOVIE)
    data = neo.fetch(query)
    neo.close()
    return data

def get_places_from_db(semt: Optional[str] = None):
    query = """
    MATCH (p:Place)
    RETURN 
        p.name AS name,
        p.google_adres AS address,
        p.google_ortalama_puan AS rating,
        p.maps_linki AS maps_link
    """
    neo = Neo4jConnector(DATABASE_PLACE)
    data = neo.fetch(query)
    neo.close()
    if semt:
        return [p for p in data if (p.get("address") or "").lower().find(semt.lower()) >= 0]
    return data

import time
from typing import List, Dict, Optional

# Mekan müsaitlik verisini simüle eden bir sözlük
RESERVATION_DATA = {
    "Mekan A": {
        "2024-12-25": {
            "18:00": True,
            "19:00": False,  # Dolu
            "20:00": False,  # Dolu
            "21:00": True
        }
    },
    "Mekan B": {
        "2024-12-25": {
            "19:30": True,
            "20:30": True,
        }
    },
    "Mekan C": {
        "2024-12-26": {
            "18:00": True,
            "19:00": True,
        }
    }
}

def check_reservation_status(place_name: str, date: str, time: str) -> Dict:
    """
    Belirtilen mekan, tarih ve saat için rezervasyon durumunu kontrol eder.
    Gerçek bir API çağrısını simüle etmek için küçük bir gecikme ekler.
    
    Args:
        place_name (str): Rezervasyon yapılacak mekanın adı.
        date (str): Rezervasyon tarihi (örn: "2024-12-25").
        time (str): Rezervasyon saati (örn: "20:00").
        
    Returns:
        dict: Müsaitlik durumu ve mesajı içeren bir sözlük.
    """
    # API gecikmesini simüle et
    time.sleep(0.5)

    place_data = RESERVATION_DATA.get(place_name)
    if not place_data:
        return {"is_available": False, "message": f"{place_name} için rezervasyon bilgisi bulunamadı."}

    date_data = place_data.get(date)
    if not date_data:
        # Belirtilen tarihte bilgi yoksa, müsaitlik durumu kontrol edilebilir
        return {"is_available": True, "message": f"{place_name} için {date} tarihinde müsaitlik bulunmaktadır."}

    is_available = date_data.get(time)
    if is_available is None:
        return {"is_available": True, "message": f"{place_name} için {date} tarihinde {time} saatine rezervasyon yapılabilir."}

    if not is_available:
        # Doluysa, alternatif saatleri öner
        available_times = [t for t, avail in date_data.items() if avail]
        if available_times:
            suggestion = ", ".join(available_times)
            return {"is_available": False, "message": f"Maalesef, {time} dolu. Alternatif saatler: {suggestion}."}
        else:
            return {"is_available": False, "message": f"Maalesef, {date} tarihi için hiç müsait saat bulunmamaktadır."}
    
    return {"is_available": True, "message": f"{place_name} için {date} tarihinde {time} saatine rezervasyon yapılabilir."}



# Sinema bileti verisini simüle eden bir sözlük
MOVIE_TICKET_DATA = {
    "Inception": [
        {"cinema": "CityMall Sineması", "time": "19:30", "link": "http://cinema.com/tickets/inception"},
        {"cinema": "MegaPlex", "time": "21:00", "link": "http://megaplex.com/tickets/inception"},
    ],
    "Interstellar": [
        {"cinema": "Grand Sineması", "time": "18:00", "link": "http://grandcinema.com/interstellar"}
    ]
}

def search_movie_tickets(movie_name: str) -> List[Dict]:
    """
    Vizyondaki filmler için sinema bileti arar.
    
    Args:
        movie_name (str): Bilet aranacak filmin adı.
        
    Returns:
        List[Dict]: Sinema, saat ve link bilgilerini içeren bir liste.
    """
    time.sleep(0.5)
    
    movie_info = MOVIE_TICKET_DATA.get(movie_name)
    if movie_info:
        return movie_info
    
    return []

# Streaming link verisini simüle eden bir sözlük
STREAMING_DATA = {
    "The Matrix": [
        {"platform": "Netflix", "link": "http://netflix.com/thematrix"},
        {"platform": "Amazon Prime Video", "link": "http://primevideo.com/thematrix"},
        {"platform": "Apple TV", "link": "http://appletv.com/thematrix"}
    ],
    "Interstellar": [
        {"platform": "Netflix", "link": "http://netflix.com/interstellar"}
    ],
    "Inception": [] # Henüz hiçbir platformda değil
}

def search_streaming_links(movie_name: str) -> List[Dict]:
    """
    Vizyonda olmayan filmler için yayın platformu linklerini bulur.
    
    Args:
        movie_name (str): Yayın platformu aranan filmin adı.
        
    Returns:
        List[Dict]: Platform ve link bilgilerini içeren bir liste.
    """
    time.sleep(0.5)
    
    streaming_info = STREAMING_DATA.get(movie_name)
    if streaming_info is None:
        return []
        
    return streaming_info

# ----------------- API Uç Noktaları -----------------

@app.get("/tools/get_movies", response_model=List[Movie])
def get_movies_tool():
    """Film veya dizi verilerini arar."""
    return get_movies_from_db()

@app.get("/tools/get_places", response_model=List[Place])
def get_places_tool(semt: Optional[str] = None):
    """İstanbul'daki mekanları semte göre arar."""
    return get_places_from_db(semt)

@app.get("/tools/check_reservation_status", response_model=ReservationStatus)
def check_reservation(place_name: str, date: str, time: str):
    """Belirtilen mekan, tarih ve saat için rezervasyon durumunu kontrol eder."""
    return check_reservation_status(place_name, date, time)

@app.get("/tools/search_movie_tickets", response_model=List[MovieTicket])
def search_tickets(movie_name: str):
    """Vizyondaki filmler için sinema bileti arar."""
    return search_movie_tickets(movie_name)

@app.get("/tools/search_streaming_links", response_model=List[StreamingLink])
def search_streaming(movie_name: str):
    """Vizyonda olmayan filmler için yayın platformu linklerini bulur."""
    return search_streaming_links(movie_name)

@app.get("/tools")
def get_tools_list():
    """Modelin kullanabileceği araçların listesini döndürür."""
    return [
        {"name": "get_movies", "description": "Film ve dizi verilerini arar."},
        {"name": "get_places", "description": "İstanbul'daki mekanları semte göre arar.", "parameters": {"semt": {"type": "string"}}},
        {"name": "check_reservation_status", "description": "Belirtilen mekan için rezervasyon durumu ve saatlerini kontrol eder.", "parameters": {"place_name": {"type": "string"}, "date": {"type": "string"}, "time": {"type": "string"}}},
        {"name": "search_movie_tickets", "description": "Vizyondaki filmler için sinema bileti arar.", "parameters": {"movie_name": {"type": "string"}}},
        {"name": "search_streaming_links", "description": "Vizyonda olmayan filmler için yayın platformu linklerini bulur.", "parameters": {"movie_name": {"type": "string"}}},
    ]
    
@app.get("/stats")
def get_database_stats():
    """Veritabanındaki film, dizi ve mekan sayılarını döndürür."""
    try:
        neo_movie = Neo4jConnector(DATABASE_MOVIE)
        movies_and_series_count = neo_movie.count_nodes("Movie") # "Movie" etiketini kullanarak film ve dizi sayısını alın.
        neo_movie.close()
        
        neo_place = Neo4jConnector(DATABASE_PLACE)
        locations_count = neo_place.count_nodes("Place") # "Place" etiketini kullanarak mekan sayısını alın.
        neo_place.close()
        
        return {
            "movies_and_series_count": movies_and_series_count,
            "locations_count": locations_count
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Veritabanı istatistikleri alınırken hata oluştu: {e}")