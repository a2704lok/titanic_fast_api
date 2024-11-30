from enum import Enum
from pydantic import BaseModel
class Passenger(Enum):
    FIRST_CLASS = 1
    SECOND_CLASS = 2
    THIRD_CLASS = 3

class Sex(Enum):
    Male = 0
    Female = 1

class Features(BaseModel):
    Pclass: Passenger
    Sex: Sex	
    Age: float
    SibSp: int
    Fare: float
