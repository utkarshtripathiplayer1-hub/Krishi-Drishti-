 # BirsaKisan-Drishti Backend 
FASTAPI backend for Birsakisan-Drishti agricultural assitance

## QUICK START

### PREREQUISISTES
- python 3.9+
- -MongoDB
- -Required python packages('requirements.txt')

  ### Installation

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up environment variables (optional):**
   Create a `.env` file in the root directory:
   ```
   MONGODB_URI=your_mongodb_connection_string
   CORS_ORIGINS=*
   SEND_SMS=false
   ```

3. **Run the server:**
   ```bash
   uvicorn main:app --reload
    ```

## API Endpoints

**GET** `/`
- **RESPONSE:**
```json
{
  "status": "ok",
  "message": "Birsakisan Backend is running"
}
```

## AUTHENTICATION
**POST** `/api/auth/login` - Request OTP
**POST** `/api/auth/verify-otp` - Verify OTP and login
**POST** `/api/auth/resend-otp` - Resend OTP

### Crop Prediction (`/api/crop`)

**POST** `/api/crop/predict` - Get crop recommendation

### Leaf Disease Detection (`/api/leaf`)

**POST** `/api/leaf/upload` - Upload leaf image (multipart/form-data)

### Location & Weather (`/api/location`)

**GET** `/api/location/details?lat={latitude}&lon={longitude}` - Get location and weather info

### Yield Prediction (`/api/yield`)

**POST** `/api/yield/predict` - Predict crop yield

### Fertilizer Recommendation (`/api/fertilizer`)

**POST** `/api/fertilizer/recommend` - Get fertilizer recommendation

### Irrigation Planning (`/api/irrigation`)

**POST** `/api/irrigation/plan` - Get irrigation plan

### Voice Assistant (`/api/voice`)

**POST** `/api/voice/assistant` - Voice assistant (multipart/form-data)

## API Documentation

Interactive API documentation is available at:
-swagger UI:`http://127.0.0.1:8000/docs`


