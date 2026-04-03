from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import logging
import warnings
import asyncio
from concurrent.futures import ThreadPoolExecutor
import time
from contextlib import asynccontextmanager
import requests
import pandas as pd
import os

# ========================
# IMPORTS
# ========================
from yield_predictor import predict_yield
from irrigation_planner import predict_irrigation
from disease_model import predict_disease_from_file
from gemini_free import get_response
from database import init_db, close_db
from route_auth import router as auth_router
from crop_recommender import router as crop_router
from database_mongo import init_mongodb
import sys
import os
# Add database directory to path to import simple_community
db_path = os.path.join(os.path.dirname(__file__), 'database')
if db_path not in sys.path:
    sys.path.insert(0, db_path)
from simple_community import get_posts as get_community_posts_from_db, init_posts_db, create_post as create_post_in_db


# ========================
# LOGGING & WARNINGS SETUP
# ========================
# Suppress noisy model unpickle warnings (inconsistent sklearn/xgboost versions)
warnings.filterwarnings(
    "ignore",
    message=r"Trying to unpickle estimator.*",
    category=UserWarning
)
warnings.filterwarnings(
    "ignore",
    message=r"If you are loading a serialized model \(like pickle in Python, RDS in R\) or",
    category=UserWarning
)

# Setup logging - Show INFO to see request flow
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
# Suppress only verbose uvicorn access logs
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# Thread pool for blocking operations
executor = ThreadPoolExecutor(max_workers=5)

# ========================
# LIFESPAN CONTEXT MANAGER (modern FastAPI 0.93+)
# ========================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application startup and shutdown lifecycle.
    Manages database initialization and cleanup.
    """
    # ================ STARTUP ================
    logger.info("=" * 60)
    logger.info("[STARTUP] Application starting...")
    logger.info("=" * 60)
    
    # Initialize MongoDB with retries
    logger.info("[STARTUP] Initializing MongoDB with retries...")
    mongo_ok = init_mongodb(retries=5, delay=2)
    if mongo_ok:
        logger.info("[STARTUP] ✅ MongoDB initialized successfully")
        # Only initialize MongoDB collections if connection is successful
        try:
            await asyncio.wait_for(init_db(), timeout=5.0)
            logger.info("[STARTUP] ✅ MongoDB collections initialized")
        except asyncio.TimeoutError:
            logger.warning("[STARTUP] ⚠️ MongoDB collection initialization timed out - continuing")
        except Exception as e:
            logger.warning(f"[STARTUP] ⚠️ MongoDB collection error: {str(e)} - continuing")
    else:
        logger.warning("[STARTUP] ⚠️ MongoDB unavailable - app will work without MongoDB persistence")
    
    # Initialize community posts database (SQLite - always works, doesn't need MongoDB)
    logger.info("[STARTUP] Initializing community posts database...")
    try:
        init_posts_db()
        logger.info("[STARTUP] ✅ Community ready")
    except Exception as e:
        logger.error(f"[STARTUP] ❌ Community DB error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
    
    logger.info("[STARTUP] ✅ Application ready")
    logger.info("=" * 60)
    
    yield  # Application runs here
    
    # ================ SHUTDOWN ================
    logger.info("=" * 60)
    logger.info("[SHUTDOWN] Application shutting down...")
    
    try:
        await close_db()
        logger.info("[SHUTDOWN] ✅ Database closed")
    except Exception as e:
        logger.warning(f"[SHUTDOWN] ⚠️ Database close error: {str(e)}")
    
    logger.info("[SHUTDOWN] ✅ Shutdown complete")
    logger.info("=" * 60)

# ========================
# FASTAPI APP SETUP
# ========================
app = FastAPI(
    title="Birsakisan Backend",
    description="Crop recommendation, irrigation planning, yield prediction, disease detection",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware (allow all origins for frontend compatibility)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(auth_router)
app.include_router(crop_router)

# ========================
# SCHEMAS
# ========================
class YieldInput(BaseModel):
    latitude: float
    longitude: float
    district_encoded: int
    N: float
    P: float
    K: float
    pH: float
    soil_moisture: float
    temperature: float
    humidity: float
    rainfall: float
    solar_radiation: float
    irrigation_encoded: int
    season_encoded: int
    soil_health_score: float
    water_availability: float
    crop_label: str
    harvest_price_rs_per_kg: float = 15

class IrrigationInput(BaseModel):
    latitude: float
    longitude: float
    soil_moisture: float
    temperature: float
    humidity: float
    rainfall: float
    crop_label: str
    season_encoded: int = 0
    clay_percent: float = 30
    sand_percent: float = 50

class ChatRequest(BaseModel):
    user_id: str = "demo"
    message: str
    language: str = "hindi"
    district: str = None  # Optional: user's district for context

class CreatePostRequest(BaseModel):
    """
    Community post creation payload.

    This maps directly to the frontend Community tab:
    - user_id: ID of the user creating the post
    - caption: Short title for the post
    - body: Full content / description
    - thumbnail_url: Optional image/thumbnail URL (handled by frontend)
    """
    user_id: str
    caption: str
    body: str
    thumbnail_url: Optional[str] = None


@app.get("/community/posts")
async def get_community_posts(limit: int = 5, offset: int = 0):
    """
    Get community posts with pagination
    
    Returns: JSON array of posts directly (not wrapped in object), e.g.:
    [
        {
            "id": 1,
            "user_id": "farmer_123",
            "caption": "पaddy में कीट नियंत्रण के उपाय",
            "body": "मेरे खेत में स्टेम बोरर...",
            "thumbnail_url": "https://example.com/thumb1.jpg",
            "created_at": "2025-12-09T01:00:00"
        }
    ]
    """
    try:
        posts = get_community_posts_from_db(limit, offset)
        # Return array directly (Flutter expects List<Map>)
        return posts
    except Exception as e:
        logger.error(f"[COMMUNITY] Error fetching posts: {str(e)}")
        return []

@app.post("/community/posts")
async def create_community_post(post: CreatePostRequest):
    """
    Create a new community post
    
    Request Body:
    {
        "user_id": "farmer_123",
        "caption": "पaddy में कीट नियंत्रण",
        "body": "मेरे खेत में समस्या है...",
        "thumbnail_url": "https://example.com/thumb1.jpg"   // Optional
    }
    
    Returns: Created post with id
    {
        "id": 8,
        "user_id": "farmer_123",
        "caption": "पaddy में कीट नियंत्रण",
        "body": "मेरे खेत में समस्या है...",
        "thumbnail_url": "https://example.com/thumb1.jpg",
        "created_at": "2025-12-09T02:45:00"
    }
    """
    try:
        # Validate input
        if not post.user_id or not post.user_id.strip():
            raise HTTPException(status_code=400, detail="user_id is required")
        if not post.caption or not post.caption.strip():
            raise HTTPException(status_code=400, detail="caption is required")
        if not post.body or not post.body.strip():
            raise HTTPException(status_code=400, detail="body is required")
        
        # Create the post
        new_post = create_post_in_db(
            user_id=post.user_id.strip(),
            caption=post.caption.strip(),
            body=post.body.strip(),
            thumbnail_url=post.thumbnail_url.strip() if post.thumbnail_url else None,
        )
        
        logger.info(f"[COMMUNITY] ✅ New post created: ID={new_post['id']}, Caption={post.caption[:50]}")
        return new_post
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[COMMUNITY] ❌ Error creating post: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Failed to create post: {str(e)}")

# ========================
# MARKET PRICE ENDPOINT
# ========================

@app.get("/market-price")
async def get_market_price():
    """
    Return market price and arrival data from CSV as JSON list of records.
    - Reads data/market_price.csv
    - Replaces '-' with None
    - Preserves column names exactly as in CSV
    """
    csv_path = os.path.join(os.path.dirname(__file__), "data", "market_price.csv")

    try:
        df = pd.read_csv(csv_path)
        # Replace '-' with None
        df = df.replace("-", None)
        # Convert to list of dicts
        data = df.to_dict(orient="records")
        return data
    except Exception as e:
        logger.error(f"[MARKET-PRICE] Error reading CSV: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Could not read market price CSV file"
        )

# ========================
# HELPER FUNCTIONS - LOCATION & WEATHER
# ========================

OPENWEATHER_API_KEY = "a0393389518004aa8edc50e205258a20"

def get_reverse_geocode(lat: float, lon: float) -> dict:
    """
    Reverse geocode latitude/longitude to get location details
    Uses OpenStreetMap Nominatim API
    """
    try:
        logger.info(f"[LOCATION] 🌍 Reverse geocoding ({lat}, {lon})...")
        
        url = f"https://nominatim.openstreetmap.org/reverse?lat={lat}&lon={lon}&format=json&addressdetails=1"
        response = requests.get(url, headers={"User-Agent": "birsakisan_app"}, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        address = data.get("address", {})
        
        location_data = {
            "city": address.get("city") or address.get("town") or address.get("village") or "",
            "district": address.get("county") or address.get("district") or "",
            "state": address.get("state") or "",
            "country": address.get("country", ""),
            "pincode": address.get("postcode", "")
        }
        
        logger.info(f"[LOCATION] ✅ Location found: {location_data['city']}, {location_data['district']}, {location_data['state']}")
        return location_data
    
    except requests.exceptions.Timeout:
        logger.warning("[LOCATION] ⏱️ Nominatim API timeout")
        return {"error": "Location API timeout", "city": "", "district": "", "state": ""}
    except requests.exceptions.RequestException as e:
        logger.warning(f"[LOCATION] ⚠️ Location API error: {str(e)}")
        return {"error": str(e), "city": "", "district": "", "state": ""}
    except Exception as e:
        logger.error(f"[LOCATION] ❌ Unexpected error: {str(e)}")
        return {"error": str(e), "city": "", "district": "", "state": ""}

def get_weather_data(lat: float, lon: float) -> dict:
    """
    Get weather data for a location
    Uses OpenWeatherMap API
    """
    try:
        logger.info(f"[WEATHER] 🌤️ Fetching weather for ({lat}, {lon})...")
        
        url = (
            f"https://api.openweathermap.org/data/2.5/weather"
            f"?lat={lat}&lon={lon}&appid={OPENWEATHER_API_KEY}&units=metric"
        )
        
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        # Safely extract weather fields
        temperature = data.get("main", {}).get("temp")
        humidity = data.get("main", {}).get("humidity")
        wind_speed = data.get("wind", {}).get("speed", 0)
        
        # Rainfall in last hour
        rainfall = 0
        if "rain" in data:
            rainfall = data["rain"].get("1h", data["rain"].get("3h", 0))
        
        # Weather description
        weather_desc = data.get("weather", [{}])[0].get("main", "Unknown")
        
        if temperature is None:
            logger.warning("[WEATHER] ⚠️ Invalid response from OpenWeatherMap")
            return {"error": "Invalid weather data", "temperature": None}
        
        weather_data = {
            "temperature": round(temperature, 2),
            "humidity": humidity,
            "rainfall": round(rainfall, 2),
            "wind_speed": round(wind_speed, 2),
            "weather": weather_desc
        }
        
        logger.info(f"[WEATHER] ✅ Weather data: {temperature}°C, {humidity}% humidity, {rainfall}mm rain")
        return weather_data
    
    except requests.exceptions.Timeout:
        logger.warning("[WEATHER] ⏱️ OpenWeatherMap API timeout")
        return {"error": "Weather API timeout", "temperature": None}
    except requests.exceptions.RequestException as e:
        logger.warning(f"[WEATHER] ⚠️ Weather API error: {str(e)}")
        return {"error": str(e), "temperature": None}
    except Exception as e:
        logger.error(f"[WEATHER] ❌ Unexpected error: {str(e)}")
        return {"error": str(e), "temperature": None}

# ========================
# ENDPOINTS
# ========================

@app.get("/")
async def root():
    """Root endpoint - check if backend is running"""
    return {"status": "ok", "message": "Birsakisan Backend is running"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok", "message": "Backend is healthy"}

@app.get("/location/details")
async def get_location_details(lat: float, lon: float):
    """
    Get location details (city, district, state, pincode) from coordinates
    
    Query Parameters:
        lat: Latitude
        lon: Longitude
    
    Returns:
        {
            "status": "success",
            "location": {
                "city": str,
                "district": str,
                "state": str,
                "country": str,
                "pincode": str
            }
        }
    """
    try:
        logger.info(f"[API] 📍 /location/details request for ({lat}, {lon})")
        
        location = get_reverse_geocode(lat, lon)
        
        if "error" in location:
            logger.warning(f"[API] ⚠️ Location fetch failed: {location['error']}")
            return {
                "status": "partial",
                "message": "Location data unavailable, check coordinates",
                "location": location
            }
        
        logger.info(f"[API] ✅ Location details returned: {location['city']}, {location['district']}")
        return {
            "status": "success",
            "location": location
        }
    
    except Exception as e:
        logger.error(f"[API] ❌ Error: {str(e)}")
        return {
            "status": "failed",
            "error": str(e)
        }

@app.get("/weather/current")
async def get_current_weather(lat: float, lon: float):
    """
    Get current weather data for a location
    
    Query Parameters:
        lat: Latitude
        lon: Longitude
    
    Returns:
        {
            "status": "success",
            "weather": {
                "temperature": float (°C),
                "humidity": int (%),
                "rainfall": float (mm),
                "wind_speed": float (m/s),
                "weather": str (e.g., "Sunny", "Rainy")
            }
        }
    """
    try:
        logger.info(f"[API] 🌤️ /weather/current request for ({lat}, {lon})")
        
        weather = get_weather_data(lat, lon)
        
        if "error" in weather:
            logger.warning(f"[API] ⚠️ Weather fetch failed: {weather['error']}")
            return {
                "status": "partial",
                "message": "Weather data unavailable, check coordinates",
                "weather": weather
            }
        
        logger.info(f"[API] ✅ Weather data returned: {weather['temperature']}°C, {weather['humidity']}% humidity")
        return {
            "status": "success",
            "weather": weather
        }
    
    except Exception as e:
        logger.error(f"[API] ❌ Error: {str(e)}")
        return {
            "status": "failed",
            "error": str(e)
        }

@app.get("/location-weather")
async def get_location_and_weather(lat: float, lon: float):
    """
    Get both location AND weather data in one call
    
    Query Parameters:
        lat: Latitude
        lon: Longitude
    
    Returns:
        {
            "status": "success",
            "location": {...},
            "weather": {...}
        }
    """
    try:
        logger.info(f"[API] 📍🌤️ /location-weather request for ({lat}, {lon})")
        
        location = get_reverse_geocode(lat, lon)
        weather = get_weather_data(lat, lon)
        
        logger.info(f"[API] ✅ Combined data returned for {location.get('city', 'Unknown')}")
        return {
            "status": "success",
            "location": location,
            "weather": weather
        }
    
    except Exception as e:
        logger.error(f"[API] ❌ Error: {str(e)}")
        return {
            "status": "failed",
            "error": str(e)
        }

@app.post("/predict_yield")
def predict_yield_endpoint(data: YieldInput):
    """Predict crop yield based on farm conditions"""
    try:
        logger.info(f"[YIELD] Request received from user")
        result = predict_yield(data.dict())
        logger.info(f"[YIELD] Prediction complete")
        return result
    except Exception as e:
        logger.error(f"[YIELD] Error: {str(e)}")
        return {"error": str(e), "status": "failed"}

@app.post("/predict_irrigation")
def predict_irrigation_endpoint(data: IrrigationInput):
    """Predict irrigation requirements for crop"""
    try:
        logger.info(f"[IRRIGATION] Request received")
        result = predict_irrigation(data.dict())
        logger.info(f"[IRRIGATION] Prediction complete")
        return result
    except Exception as e:
        logger.error(f"[IRRIGATION] Error: {str(e)}")
        return {"error": str(e), "status": "failed"}

@app.post("/disease_detection")
async def disease_detection_endpoint(file: UploadFile = File(...)):
    """Detect disease in crop leaf image"""
    try:
        logger.info(f"[DISEASE] File received: {file.filename}")
        
        # Read file
        contents = await file.read()
        
        # Save temporarily
        with open("temp_disease_image.jpg", "wb") as f:
            f.write(contents)
        
        # Predict
        result = predict_disease_from_file("temp_disease_image.jpg")
        logger.info(f"[DISEASE] Detection complete")
        return result
    except Exception as e:
        logger.error(f"[DISEASE] Error: {str(e)}")
        return {"error": str(e), "status": "failed"}

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """Enhanced KrishiMitra Chatbot endpoint with safety validation and intent classification"""
    try:
        logger.info(f"[CHAT] 🌾 Request from {request.user_id} (lang: {request.language}, district: {request.district})")
        start = time.time()
        
        # Call enhanced get_response with all parameters
        result = get_response(
            user_message=request.message,
            language=request.language,
            user_id=request.user_id,
            district=request.district
        )
        
        elapsed = time.time() - start
        logger.info(f"[CHAT] ✅ Response ready in {elapsed:.2f}s | Intent: {result['intent']} | Safe: {result['is_safe']}")
        
        return {
            "status": "success",
            "message": request.message,
            "response": result['reply'],
            "language": result['language'],
            "confidence": result['confidence'],
            "intent": result['intent'],
            "is_safe": result['is_safe'],
            "warnings": result['warnings'],
            "user_id": request.user_id,
            "district": request.district or "unknown",
            "response_time_seconds": elapsed
        }
    except Exception as e:
        logger.error(f"[CHAT] ❌ Error: {str(e)}")
        return {
            "status": "failed",
            "error": str(e),
            "user_id": request.user_id
        }
    
    

# ========================
# RUN
# ========================
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting server with uvicorn...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
