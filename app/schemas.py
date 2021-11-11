from fastapi import Query
from pydantic import BaseModel


# Define the Text object
# Each Text object is a string that defaults to None and must have a minimum length of 3 characters
class Text(BaseModel):
    text: str = Query(None, min_length=3)


# Define the RetrievePayload object
class RetrievePayload(BaseModel):
    text: Text

    class Config:
        schema_extra = {
            "example": {
                "text": "SARS-CoV-2 binds ACE2 receptor to gain entry into cells."
            }
        }
