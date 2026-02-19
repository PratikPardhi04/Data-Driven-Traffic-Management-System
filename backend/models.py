from sqlalchemy import Column, Integer, Float, String, Boolean
from database import Base

class TrafficState(Base):
    __tablename__ = "traffic_state"

    id = Column(Integer, primary_key=True, index=True)
    intersection_id = Column(Integer)
    vehicle_count = Column(Integer)
    density_score = Column(Float)
    congestion_level = Column(String)
    emergency_present = Column(Boolean)
    timestamp = Column(Float)
