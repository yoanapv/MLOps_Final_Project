import os
import sys

# Add the parent directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(parent_dir)

from fastapi import FastAPI
from starlette.responses import JSONResponse

from predictor.predict import ModelPredictor
from api.models.models import HotelReservation
from train.train_data import HotelReservationsDataPipeline

app = FastAPI()

@app.get('/', status_code=200)
async def healthcheck():
    return 'HotelReservation classifier is all ready to go!'

@app.post('/predict')
def predict(hotelreservation_features: HotelReservation):
    predictor = ModelPredictor("/Users/norma.perez/Documents/GitHub/MLOps_Final_Project/mlops_final_project/mlops_final_project/models/extra_trees_classifier_model_output.pkl")
    X = [hotelreservation_features.no_of_week_nights,
        hotelreservation_features.lead_time,
        hotelreservation_features.arrival_month,
        hotelreservation_features.arrival_date,
        hotelreservation_features.avg_price_per_room,
        hotelreservation_features.no_of_special_requests]
    print([X])
    prediction = predictor.predict([X])
    return JSONResponse(f"Resultado predicción: {prediction}")

