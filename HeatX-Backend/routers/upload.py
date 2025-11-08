from fastapi import FastAPI, UploadFile, File, APIRouter
import pandas as pd

router = APIRouter()

@router.post("/api/")
async def upload(file: UploadFile = File(...)):
    df = pd.read_csv(file.file)
    preview = df.head(6).fillna('-').values.tolist()  # only first 6 rows for UI
    full_data = df.fillna('-').values.tolist()        # full dataset for modeling
    return {"preview": preview, "dataset": full_data}

