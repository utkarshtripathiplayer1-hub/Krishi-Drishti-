from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from pymongo import MongoClient
from bson import ObjectId
from datetime import datetime
from dotenv import load_dotenv
from email_service import send_email_alert
import os


# APP  
app = FastAPI()

# Mongodb connection

MONGO_URL = "mongodb+srv://tanishabhatt06_db_user:tanishabhatt@beehive.wxh44zt.mongodb.net/?appName=beehive"
load_dotenv()
MONGO_URL=os.getenv("MONGO_URL")
client = MongoClient(MONGO_URL)
db = client["beehive_db"]

users_col = db["users"]
hives_col = db["hives"]
sensor_col = db["sensor_data"]
audio_col = db["audio_data"]
alerts_col = db["alerts"]

# -----------------------------
# CLEAN FUNCTION (PASTE HERE)
# -----------------------------
def clean_data(doc):
    return {
        "id": str(doc["_id"]),
        "hive_id": str(doc.get("hive_id")),
        "temperature": doc.get("temperature"),
        "humidity": doc.get("humidity"),
        "weight": doc.get("weight"),
        "prediction": doc.get("prediction"),
        "timestamp": str(doc.get("timestamp"))
    }

def predict_health(temp, humidity, co2,voc,pressure,light,vibration,microphone_db,rain):
    # Dummy logic (for now)
    if temp > 35:
        return "overheat"
    elif humidity < 50:
        return "dry"
    elif co2 > 1500 :
        return "co2_overload"
    elif voc < 80:
        return "VOC_Spike"
    elif pressure > 5:
        return "Weather_change"
    elif light > 200:
        return "Hive_open or damage"
    elif vibration >3:
        return " High Disturbance"
    elif microphone_db > 80:
        return "Swarming"
    elif rain==1:
        return "Foraging_reduced"
    else:
        return "healthy"

# -----------------------------
# CREATE AUDIO FOLDER
# -----------------------------
if not os.path.exists("audio"):
    os.makedirs("audio")

# -----------------------------
# HELPER
# -----------------------------
def serialize(doc):
    doc["_id"] = str(doc["_id"])
    return doc

# -----------------------------
# SCHEMAS
# -----------------------------
class User(BaseModel):
    name: str
    email: str
    password: str

class Hive(BaseModel):
    user_id: str
    location: str

from pydantic import BaseModel

class SensorData(BaseModel):
    user_id: str
    hive_id: str

    temperature: float
    humidity: float
    co2: float
    voc: float
    pressure: float
    light: float
    vibration: float
    microphone_db: float
    rain: float

# -----------------------------
# ROOT
# -----------------------------
@app.get("/")
def home():
    return {"status": "Backend Running "}

# -----------------------------
# USER API
# -----------------------------
@app.post("/register")
def register(user: User):
    if users_col.find_one({"email": user.email}):
        raise HTTPException(status_code=400, detail="Email exists")

    result = users_col.insert_one(user.dict())
    return {"user_id": str(result.inserted_id)}

# -----------------------------
# HIVE API
# -----------------------------
@app.post("/create-hive")
def create_hive(hive: Hive):
    try:
        new_hive = {
            "user_id": hive.user_id,
            "location": hive.location
        }

        result = hives_col.insert_one(new_hive)

        return {
            "message": "Hive created",
            "hive_id": str(result.inserted_id)
        }

    except Exception as e:
        return {"error": str(e)}


# Sensor api

@app.post("/sensor-data")
def add_sensor(data: SensorData):

    # STEP 1: prediction
    prediction = predict_health(
        data.temperature,
        data.humidity,
        data.co2,
        data.voc,
        data.pressure,
        data.light,
        data.vibration,
        data.microphone_db,
        data.rain
    )

    print("Prediction:", prediction)

    # STEP 2: alerts based on prediction
    try:
        if prediction != "healthy":
            send_email_alert(
                f"ALERT!\nHive: {data.hive_id}\nStatus: {prediction}"
            )
            print("Email sent successfully")
    except Exception as e:
        print("Email error:", e)

    # STEP 3: rule-based alerts
    alerts = []

    if data.temperature > 38:
        alerts.append("OVERHEAT")

    if data.co2 > 1500:
        alerts.append("CO2_HIGH")

    if data.vibration > 3:
        alerts.append("DISTURBANCE")

    if data.microphone_db > 80:
        alerts.append("SWARMING")

    if alerts:
        message = "Beehive Alert!\n\n" + "\n".join(alerts)
        try:
            send_email_alert(message)
            print("Rule-based email sent")
        except Exception as e:
            print("Rule-based email error:", e)

    # STEP 4: store data
    new_data = {
        "user_id": data.user_id,
        "hive_id": data.hive_id,
        "temperature": data.temperature,
        "humidity": data.humidity,
        "co2": data.co2,
        "voc": data.voc,
        "pressure": data.pressure,
        "light": data.light,
        "vibration": data.vibration,
        "microphone_db": data.microphone_db,
        "rain": data.rain,
        "prediction": prediction
    }

    sensor_col.insert_one(new_data)

    return {
        "message": "Data stored",
        "prediction": prediction,
        "alerts": alerts
    }




# Alert API

@app.post("/alerts")
def create_alert(hive_id: str, message: str, severity: str):
    alert = {
        "hive_id": hive_id,
        "message": message,
        "severity": severity,
        "timestamp": datetime.utcnow()
    }

    alerts_col.insert_one(alert)
    return {"message": "Alert created"}

@app.get("/alerts/{hive_id}")
def get_alerts(hive_id: str):
    data = alerts_col.find({"hive_id": hive_id})
    return [serialize(d) for d in data]

# Health check

@app.get("/health")
def health():
    return {"status": "OK"}
