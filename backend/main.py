from fastapi import FastAPI
from pydantic import BaseModel
import os

app = FastAPI()

class FileRequest(BaseModel):
    filename: str  # CSVファイル名（相対 or 絶対）

@app.post("/api/load-csv")
def load_csv(req: FileRequest):
    # VSCodeのブレークポイントをここに設定
    print(f"🔍 受け取ったファイル名: {req.filename}")
    return {"message": "ファイル名受け取り完了", "filename": req.filename}
