from pydantic import BaseModel, ValidationError

class BusinessRecord(BaseModel):
    business_id: str
    rating: float
    pcpi: float
    poverty_rate: float
    median_household_income: float
    unemployment_rate: float
    avg_weekly_wages: float
    is_open: int

def validate_records(df):
    for i, row in enumerate(df.to_dicts()):
        try:
            BusinessRecord(**row)
        except ValidationError as e:
            print(f"Validation error in row {i}: {e}")
            raise e
