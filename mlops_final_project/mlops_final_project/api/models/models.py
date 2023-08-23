from pydantic import BaseModel


class HotelReservation(BaseModel):
    """
    Represents a customer who made a hotel reservation with various attributes.
    
    Attributes:
        no_of_week_nights (int): Number of week nights (Monday to Friday) the guest stayed or booked to stay at the hotel.
        lead_time (int): P Number of days between the date of booking and the arrival date.
        arrival_month (int):  Month of arrival date.
        arrival_date (int):  Date of the month.
        avg_price_per_room (float):  Average price per day of the reservation; prices of the rooms are dynamic. (in euros).
        no_of_special_requests (int): Total number of special requests made by the customer (e.g. high floor, view from the room, etc).
    """

    no_of_week_nights: int
    lead_time: int
    arrival_month: int
    arrival_date: int
    avg_price_per_room: float
    no_of_special_requests: int

    
    
