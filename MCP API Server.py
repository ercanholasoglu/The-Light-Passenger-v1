from flask import Flask, request, jsonify
from typing import Dict, Any
import requests
import json
import os
import re
import unicodedata
from cachetools import cached, TTLCache
from datetime import datetime
from neo4j import GraphDatabase
import sys
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document

app = Flask(__name__)

# Ortam değişkenlerini al
try:
    OPENWEATHER_API_KEY = os.environ.get("OPENWEATHER_API_KEY")
    TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")
    NEO4J_URI = os.environ.get("NEO4J_URI")
    NEO4J_USER = os.environ.get("NEO4J_USER")
    NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD")
    OLLAMA_MODEL_NAME = "llama3-8b-tr"
except KeyError as e:
    print(f"Eksik ortam değişkeni: {e}. Lütfen yapılandırın.")
    sys.exit(1)

# Neo4j bağlantısı
class Neo4jConnector:
    def __init__(self):
        self.uri = NEO4J_URI
        self.user = NEO4J_USER
        self.password = NEO4J_PASSWORD
        self.database = "neo4j"
        self.driver = None

    def connect(self):
        if self.driver is None:
            try:
                self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
                self.driver.verify_connectivity()
            except Exception as exc:
                print(f"Neo4j bağlantı hatası: {exc}")
                raise ConnectionError(f"Neo4j bağlantı hatası: {exc}") from exc
    
    def get_meyhaneler(self, limit: int = 10000) -> List[Dict[str, Any]]:
        self.connect()
        query = """
        MATCH (m:Meyhane)
        RETURN
            m.name                  AS name,
            m.google_adres          AS address,
            m.google_ortalama_puan AS rating,
            m.google_toplam_yorum   AS review_count,
            m.maps_linki            AS map_link,
            m.google_telefon        AS phone,
            m.fiyat_seviyesi_simge AS price_level,
            elementId(m)            AS neo4j_element_id
        ORDER BY m.google_ortalama_puan DESC
        LIMIT $limit
        """
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(query, limit=limit)
                records = [self._clean_record(record) for record in result]
                return records
        except Exception as exc:
            print(f"Sorgu hatası (get_meyhaneler): {exc}")
            return []
        
    @staticmethod
    def _clean_record(record) -> Dict[str, Any]:
        name = record.get("name") or "Bilinmiyor"
        address = record.get("address") or "Adres yok"
        rating_value = record.get("rating")
        rating = float(rating_value) if rating_value is not None else 0.0
        review_count_value = record.get("review_count")
        review_count = int(review_count_value) if review_count_value is not None else 0
        map_link = record.get("map_link") or ""
        phone = record.get("phone") or ""
        price_level = record.get("price_level") or ""
        neo4j_element_id = record["neo4j_element_id"]
        return {
            "name": name,
            "address": address,
            "rating": rating,
            "review_count": review_count,
            "map_link": map_link,
            "phone": phone,
            "price_level": price_level,
            "neo4j_element_id": neo4j_element_id,
        }

    def close(self):
        if self.driver:
            self.driver.close()
            self.driver = None

neo4j_conn = Neo4jConnector()

# Diğer yardımcı fonksiyonlar (clean_location_query, get_openweather_forecast, format_weather_response) buraya taşınacak
def clean_location_query(query: str) -> str:
    normalized_query = unicodedata.normalize('NFKD', query.lower()).encode('ascii', 'ignore').decode('utf-8')
    istanbul_locations = [
        r'etiler', r'levent', r'maslak', r'nisantasi', r'nisantaşi',
        r'bebek', r'arnavutkoy', r'arnavutköy', r'ortakoy', r'ortaköy', r'cihangir',
        r'taksim', r'karakoy', r'karaköy', r'galata', r'fatih',
        r'sultanahmet', r'eminonu', r'eminönü', r'kadikoy', r'kadıköy', r'moda',
        r'bagdat caddesi', r'bağdat caddesi', r'suadiye', r'bostanci', r'bostancı',
        r'maltepe', r'kartal', r'pendik', r'uskudar', r'üsküdar',
        r'camlica', r'çamlıca', r'beykoz', r'atasehir', r'ataşehir', r'cekmekoy', r'çekmeköy',
        r'sariyer', r'sarıyer', r'istinye', r'tarabya', r'yenikoy', r'yeniköy',
        r'bahcekoy', r'bahçeköy', r'buyukdere', r'büyükdere', r'zumrutevler', r'zümrutevler',
        r'florya', r'yesilkoy', r'yeşilköy', r'yesilyurt', 'yeşilyurt', r'bakirkoy', r'bakırköy',
        r'atakoy', r'ataköy', r'zeytinburnu', r'gungoren', r'güngören', r'esenler',
        r'bayrampasa', r'bayrampaşa', r'gaziosmanpasa', r'gaziosmanpaşa', r'eyup', r'eyüp', r'kagithane', r'kağıthane',
        r'sisli', r'şişli', r'besiktas', r'beşiktaş', r'avcilar', r'avcılar', r'beylikduzu', 'beylikdüzü',
        r'esenyurt', r'buyukcekmece', r'büyükçekmece', r'silivri', r'catalca', r'çatalca',
        r'sile', r'şile', r'agva', r'ağva', r'adalar', r'basaksehir', 'başakşehir',
        r'bahcelievler', r'bahçelievler', 'kucukcekmece', 'küçükçekmece', 'cankurtaran'
    ]
    for loc_regex in istanbul_locations:
        match = re.search(r'\b' + loc_regex + r'\b', normalized_query)
        if match:
            return match.group(0)
    general_cities = [
        r'istanbul', r'ankara', r'izmir', r'adana',
        r'bursa', r'antalya', r'konya', r'kayseri',
        r'gaziantep', r'samsun', r'eskisehir', r'eskişehir', r'duzce', r'düzce'
    ]
    for city_regex in general_cities:
        match = re.search(r'\b' + city_regex + r'\b', normalized_query)
        if match:
            return match.group(0)
    return "istanbul"

weather_cache = TTLCache(maxsize=100, ttl=300)

@cached(weather_cache)
def get_openweather_forecast(location: str) -> Dict:
    api_key = OPENWEATHER_API_KEY
    if not api_key:
        return {"error": "API anahtarı bulunamadı."}
    try:
        geo_response = requests.get(
            f"http://api.openweathermap.org/geo/1.0/direct?q={location},TR&limit=1&appid={api_key}",
            timeout=10,
        )
        geo_response.raise_for_status()
        geo = geo_response.json()
        if not geo:
            return {"error": f"'{location}' konumu bulunamadı."}
        lat, lon = geo[0]["lat"], geo[0]["lon"] 
        weather_response = requests.get(
            f"https://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={api_key}&units=metric&lang=tr",
            timeout=10,
        )
        weather_response.raise_for_status()
        weather = weather_response.json()
        return weather
    except requests.exceptions.RequestException as e:
        return {"error": f"API hatası: {e}"}
    except Exception as e:
        return {"error": f"Beklenmedik bir hata oluştu: {e}"}

def format_weather_response(location: str, data: Dict) -> str:
    if "error" in data:
        return f"❌ {data['error']}"
    try:
        lines = [f"🌤️ **{location.capitalize()} Hava Durumu Tahmini:**"]
        if "list" not in data or not data["list"]:
            return f"❌ {location} için hava durumu verisi bulunamadı."
        today = datetime.now().date()
        daily_forecasts = {}
        for item in data["list"]:
            timestamp = item["dt"]
            forecast_time = datetime.fromtimestamp(timestamp)
            forecast_date = forecast_time.date()
            if forecast_date >= today and len(daily_forecasts) < 5:
                date_str = "Bugün" if forecast_date == today else forecast_time.strftime("%d %B")
                temp = item["main"]["temp"]
                description = item["weather"][0]["description"]
                icon_code = item["weather"][0]["icon"]
                icon_url = f"http://openweathermap.org/img/wn/{icon_code}.png"
                if forecast_date not in daily_forecasts:
                    daily_forecasts[forecast_date] = {
                        "date_str": date_str,
                        "temps": [],
                        "descriptions": set(),
                        "icons": set()
                    }
                daily_forecasts[forecast_date]["temps"].append(temp)
                daily_forecasts[forecast_date]["descriptions"].add(description)
                daily_forecasts[forecast_date]["icons"].add(icon_url)
        for date, forecast in daily_forecasts.items():
            avg_temp = sum(forecast["temps"]) / len(forecast["temps"])
            descriptions = ", ".join(list(forecast["descriptions"]))
            lines.append(
                f"- **{forecast['date_str']}**: Sıcaklık: {avg_temp:.1f}°C, Durum: {descriptions.capitalize()}."
            )
        return "\n".join(lines)
    except Exception as e:
        return f"❌ Hava durumu bilgisi formatlanırken bir hata oluştu: {e}"

@cached(TTLCache(maxsize=100, ttl=3600))
def get_fun_fact() -> str:
    try:
        response = requests.get("https://uselessfacts.jsph.pl/api/v2/facts/random?language=tr", timeout=5)
        response.raise_for_status()
        fact = response.json().get("text", "İlginç bir bilgi bulunamadı.")
        return fact
    except requests.exceptions.Timeout:
        return "İlginç bilgi servisi şu an çok yavaş veya çalışmıyor."
    except requests.exceptions.RequestException as e:
        return f"İlginç bilgi servisi şu an çalışmıyor. Hata: {e}"
    except Exception as e:
        return f"İlginç bilgi alınırken beklenmedik bir hata oluştu: {e}"

def process_documents(docs: List[Any]) -> List[Document]:
    processed = []
    for doc in docs:
        if isinstance(doc, dict):
            metadata = {
                "Mekan Adı": doc.get("name", "Bilinmeyen Mekan"),
                "Adres": doc.get("address", "Bilinmeyen Adres"),
                "Google Puanı": str(doc.get("rating", 0.0)),
                "Google Yorum Sayısı": str(doc.get("review_count", 0)),
                "Maps Linki": doc.get("map_link", "Harita linki yok"),
                "Telefon": doc.get("phone", "Yok"),
                "Fiyat Seviyesi": str(doc.get("price_level", "Yok"))
            }
            main_content = (
                f"Mekan Adı: {metadata['Mekan Adı']}, "
                f"Adres: {metadata['Adres']}, "
                f"Google Puanı: {metadata['Google Puanı']}, "
                f"Google Yorum Sayısı: {metadata['Google Yorum Sayısı']}, "
                f"Fiyat Seviyesi: {metadata['Fiyat Seviyesi']}"
            )
            processed.append(Document(
                page_content=main_content,
                metadata=metadata
            ))
    return processed

@app.route("/get_weather_forecast", methods=["POST"])
def get_weather_forecast_route():
    data = request.json
    location = data.get("location", "istanbul")
    weather_data = get_openweather_forecast(location)
    formatted_weather = format_weather_response(location, weather_data)
    return jsonify({"result": formatted_weather})

@app.route("/provide_fun_fact", methods=["POST"])
def provide_fun_fact_route():
    fun_fact = get_fun_fact()
    return jsonify({"result": fun_fact})

@app.route("/search_places", methods=["POST"])
def search_places_route():
    data = request.json
    query = data.get("query")
    if not query:
        return jsonify({"result": "Sorgu belirtilmedi."}), 400
    
    meyhaneler_listesi = []
    try:
        meyhaneler_listesi = neo4j_conn.get_meyhaneler(limit=10000)
        if not meyhaneler_listesi:
            print("Uyarı: Neo4j'den hiç mekan verisi çekilemedi.")
            return jsonify({"result": "Veritabanında mekan bulunamadı."})
    except Exception as e:
        print(f"Neo4j'den veri çekerken hata oluştu: {e}")
        return jsonify({"result": f"Veritabanı hatası: {e}"})

    # Basit bir filtreleme
    filtered_places = [
        place for place in meyhaneler_listesi
        if query.lower() in place['name'].lower() or query.lower() in place['address'].lower()
    ]
    
    if not filtered_places:
        return jsonify({"result": f"'{query}' için mekan bulunamadı."})
    
    result_str = "İşte bulduğum mekanlar:\n"
    for place in filtered_places[:5]: # İlk 5'i döndür
        result_str += f"- Mekan Adı: {place['name']}, Adres: {place['address']}, Puan: {place['rating']}\n"
        
    return jsonify({"result": result_str})

if __name__ == "__main__":
    app.run(port=5000)
