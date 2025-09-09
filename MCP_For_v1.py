from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict
import os
from neo4j import GraphDatabase
from dotenv import load_dotenv

# ---------------- Load Env ----------------
load_dotenv()

def get_env(key, default=None):
    return os.getenv(key, default)

NEO4J_URI = get_env("NEO4J_URI")
NEO4J_USER = get_env("NEO4J_USER")
NEO4J_PASSWORD = get_env("NEO4J_PASSWORD")

DATABASE_MOVIE = "moviesandseries"
DATABASE_PLACE = "neo4j"

# ---------------- Neo4j Connector ----------------
class Neo4jConnector:
    def __init__(self, uri=None, user=None, password=None, database=DATABASE_PLACE):
        self.uri = uri or NEO4J_URI
        self.user = user or NEO4J_USER
        self.password = password or NEO4J_PASSWORD
        self.database = database
        self.driver = None

    def connect(self):
        if self.driver:
            return
        try:
            self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
            self.driver.verify_connectivity()
        except Exception as e:
            raise RuntimeError(f"Neo4j veritabanına bağlanılamadı: {e}")

    def fetch(self, query: str):
        self.connect()
        with self.driver.session(database=self.database) as session:
            result = session.run(query)
            return [dict(record) for record in result]

    def close(self):
        if self.driver:
            self.driver.close()
            self.driver = None

# ---------------- Models ----------------
class MovieTicket(BaseModel):
    cinema: str
    time: str
    link: str

class StreamingLink(BaseModel):
    platform: str
    link: str

# ---------------- Tool Fonksiyonları ----------------
def get_movies_from_db():
    query = """
    MATCH (m:Movie)
    RETURN m.title AS title, m.year AS year, m.rottenTomatoesScore AS rt
    ORDER BY m.rottenTomatoesScore DESC
    LIMIT 50
    """
    nc = Neo4jConnector()
    try:
        return nc.fetch(query)
    except Exception as e:
        raise RuntimeError(f"Neo4j sorgu hatası: {e}")

def get_total_movies_and_series():
    query = """
    MATCH (m:Movie) RETURN count(m) AS movies
    UNION ALL
    MATCH (s:Series) RETURN count(s) AS series
    """
    nc = Neo4jConnector()
    try:
        rows = nc.fetch(query)
        return sum([list(r.values())[0] for r in rows])
    except Exception as e:
        raise RuntimeError(f"Toplam film & dizi sayısı alınamadı: {e}")

def get_total_meyhane():
    query = """MATCH (p:Meyhane) RETURN count(p) AS meyhane_count"""
    nc = Neo4jConnector()
    try:
        rows = nc.fetch(query)
        return rows[0]["meyhane_count"] if rows else 0
    except Exception as e:
        raise RuntimeError(f"Meyhane sayısı alınamadı: {e}")

# ---------------- FastAPI App ----------------
app = FastAPI(title="MCP For V1")

@app.get("/tools/get_movies")
def api_get_movies(rotten_threshold: float = 90.0):
    try:
        movies = get_movies_from_db()
        return [m for m in movies if ('rt' in m and (m['rt'] or 0) >= rotten_threshold)]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/tools/metrics")
def api_metrics():
    try:
        total_movies_series = get_total_movies_and_series()
        total_meyhane = get_total_meyhane()
        return {
            "movies_and_series": total_movies_series,
            "meyhane_count": total_meyhane
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
