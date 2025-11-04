from pydantic import BaseModel
import torch
import torch.nn.functional as F
import logging
import traceback
import pandas as pd
import json
import os
import io
from datetime import datetime
from fastapi import HTTPException as FastAPIHTTPException, UploadFile

from transformers import AutoTokenizer, AutoModel
from config.logger import logger



class SentenceRequest(BaseModel):
    sentence: str

# No additional models needed - only handling CSV and JSON

logger = logging.getLogger("hrm_agent")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
bert_model = AutoModel.from_pretrained("bert-base-uncased")


UPLOAD_DIR = "uploads"
INPUT_DIR = os.path.join(UPLOAD_DIR, "input")
OUTPUT_DIR = os.path.join(UPLOAD_DIR, "output")
os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
def save_uploaded_file(file: UploadFile) -> str:
    """
    Save uploaded file to uploads/input/ and return the saved file path.
    """
    file_path = os.path.join(INPUT_DIR, file.filename)
    # Make sure to seek to beginning if file was read before
    try:
        file.file.seek(0)
    except Exception:
        pass
    with open(file_path, "wb") as f:
        f.write(file.file.read())
    return file_path


def _write_output_json(df: pd.DataFrame, original_filename: str) -> str:
    """
    Write processed dataframe to uploads/output/<originalname>_processed_<ts>.json
    Return the output file path.
    """
    base_name = os.path.splitext(original_filename)[0]
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    out_filename = f"{base_name}_processed_{timestamp}.json"
    out_path = os.path.join(OUTPUT_DIR, out_filename)
    # Convert to records and write pretty JSON (ensure utf-8)
    records = df.to_dict(orient="records")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    return out_path

def get_token_ids(question:str):
    try:
        logger.info("Question processed to generate TokenIds")
        tokens = tokenizer.tokenize(question)
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        return token_ids
    except Exception as e:
        return []
def get_tokens(question:str):
    try:
        logger.info("Question processed to generate TokenIds")
        tokens = tokenizer.tokenize(question)
        return tokens
    except Exception as e:
        return []
def get_embedding(data: SentenceRequest):
    logger.info(f"GET-EMBEDDING: Processing sentence: {data.sentence[:50]}...")
    try:
        sentence = data.sentence
        logger.info(f"GET-EMBEDDING: Tokenizing sentence of length {len(sentence)}")

        inputs = tokenizer(sentence, return_tensors="pt")
        logger.info(f"GET-EMBEDDING: Input shape: {inputs['input_ids'].shape}")

        with torch.no_grad():
            outputs = bert_model(**inputs)
            embeddings = outputs.last_hidden_state.squeeze(0).tolist()
            logger.info(f"GET-EMBEDDING: Generated embeddings shape: {len(embeddings)}x{len(embeddings[0]) if embeddings else 0}")

        tokens = tokenizer.tokenize(sentence)
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        
        logger.info(f"GET-EMBEDDING: Successfully processed {len(tokens)} tokens")
        return {
            "sentence": sentence,
            "tokens": tokens,
            "token_ids": token_ids,
            "embeddings": embeddings
        }

    except Exception as e:
        logger.error(f"GET-EMBEDDING ERROR: {str(e)}")
        logger.error(f"GET-EMBEDDING TRACEBACK: {traceback.format_exc()}")
        raise FastAPIHTTPException(status_code=500, detail=f"Embedding/Attention error: {str(e)}")


def qkv_attention(data: SentenceRequest):
    logger.info(f"QKV-ATTENTION: Processing sentence: {data.sentence[:50]}...")
    try:
        sentence = data.sentence

        # Tokenize and get input IDs
        inputs = tokenizer(sentence, return_tensors="pt")
        input_ids = inputs["input_ids"].squeeze()
        tokens = tokenizer.convert_ids_to_tokens(input_ids)
        token_ids = input_ids.tolist()
        logger.info(f"QKV-ATTENTION: Tokenized into {len(tokens)} tokens")

        with torch.no_grad():
            outputs = bert_model(**inputs)
            embeddings = outputs.last_hidden_state  # (1, seq_len, hidden_size)

        hidden_size = embeddings.size(-1)
        logger.info(f"QKV-ATTENTION: Hidden size: {hidden_size}")
        
        W_q = torch.nn.Linear(hidden_size, hidden_size)
        W_k = torch.nn.Linear(hidden_size, hidden_size)
        W_v = torch.nn.Linear(hidden_size, hidden_size)

        Q = W_q(embeddings)
        K = W_k(embeddings)
        V = W_v(embeddings)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(hidden_size, dtype=torch.float32))
        attn_weights = F.softmax(scores, dim=-1)  # (1, seq_len, seq_len)

        output_embeddings = torch.matmul(attn_weights, V).squeeze(0)  # (seq_len, hidden_size)

        # Calculate how much attention each token RECEIVES
        attn_received = attn_weights.squeeze(0).sum(dim=0)  # (seq_len,)
        attn_percent = (attn_received / attn_received.sum()) * 100  # original % (sum to 100)

        # Exclude special tokens like [CLS], [SEP] and re-normalize
        attention_info = []
        filtered_embeddings = []
        total_visible_percent = 0

        for idx, (token, percent, emb) in enumerate(zip(tokens, attn_percent, output_embeddings)):
            if token not in ["[CLS]", "[SEP]"]:
                percent_val = round(float(percent.item()), 2)
                attention_info.append({"token": token, "attention_percent": percent_val})
                filtered_embeddings.append(emb)
                total_visible_percent += percent_val

        # Re-normalize to 100% total for visible tokens
        for item in attention_info:
            item["attention_percent"] = round((item["attention_percent"] / total_visible_percent) * 100, 2)

        # Create weighted sentence embedding (based on re-normalized %)
        percent_tensor = torch.tensor([item["attention_percent"] for item in attention_info])
        percent_tensor = percent_tensor.softmax(dim=0)  # normalize for dot product
        filtered_embeddings_tensor = torch.stack(filtered_embeddings)
        weighted_embedding = torch.matmul(percent_tensor, filtered_embeddings_tensor)

        logger.info(f"QKV-ATTENTION: Successfully processed attention for {len(attention_info)} tokens")
        return {
            "sentence": sentence,
            "tokens": [t["token"] for t in attention_info],
            "token_ids": token_ids,
            "attention_percentages": attention_info,
            "output_embeddings": [emb.tolist() for emb in filtered_embeddings],
            "weighted_embedding": weighted_embedding.tolist()
        }

    except Exception as e:
        logger.error(f"QKV-ATTENTION ERROR: {str(e)}")
        logger.error(f"QKV-ATTENTION TRACEBACK: {traceback.format_exc()}")
        raise FastAPIHTTPException(status_code=500, detail=f"QKV Attention error: {str(e)}")


def process_etl_file(file: UploadFile):
    """
    ETL:
    1. Save uploaded file to uploads/input/
    2. Extract from saved file
    3. Transform (clean nulls, duplicates, column names, types)
    4. Write processed JSON to uploads/output/ and return its path in response
    """
    logger.info(f"ETL-PROCESS: Starting processing for upload: {getattr(file, 'filename', None)}")
    try:
        start_time = datetime.now()

        # SAVE -> to uploads/input/
        input_path = save_uploaded_file(file)
        filename = file.filename.lower() if file.filename else ""
        logger.info(f"ETL-PROCESS: Saved uploaded file to: {input_path}")

        # EXTRACT
        if filename.endswith(".csv"):
            logger.info("ETL-PROCESS: Reading CSV from disk")
            df = pd.read_csv(input_path)

        elif filename.endswith(".json"):
            logger.info("ETL-PROCESS: Reading JSON from disk")
            with open(input_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                df = pd.DataFrame(data)
            elif isinstance(data, dict):
                df = pd.DataFrame([data])
            else:
                raise ValueError("JSON must contain array of objects or single object")

        elif filename.endswith((".xlsx", ".xls")):
            logger.info("ETL-PROCESS: Reading Excel from disk")
            df = pd.read_excel(input_path)

        else:
            logger.error(f"ETL-PROCESS: Unsupported file format: {filename}")
            raise FastAPIHTTPException(
                status_code=400,
                detail="Unsupported file format. Please upload CSV, JSON, or Excel file.",
            )

        # keep original counts for stats
        original_rows = len(df)
        original_columns = len(df.columns)
        logger.info(f"ETL-PROCESS: Loaded dataframe with shape: {df.shape}")

        # TRANSFORM
        # 1. Remove completely empty rows
        df = df.dropna(how="all")

        # 2. Remove duplicate rows
        df = df.drop_duplicates()

        # 3. Clean column names
        df.columns = (
            df.columns.str.strip()
            .str.replace(" ", "_")
            .str.replace("[^a-zA-Z0-9_]", "", regex=True)
        )

        # 4. Convert data types and clean strings
        for col in df.columns:
            if df[col].dtype == "object":
                # Try numeric
                numeric_series = pd.to_numeric(df[col], errors="coerce")
                if not numeric_series.isna().all():
                    df[col] = numeric_series
                else:
                    # Force string and strip whitespace
                    df[col] = df[col].astype(str).str.strip()

        # 5. Fill remaining nulls
        for col in df.columns:
            if df[col].dtype.name in ["float64", "int64", "Int64"]:
                df[col] = df[col].fillna(0)
            else:
                df[col] = df[col].fillna("N/A")

        processed_rows = len(df)
        rows_removed = original_rows - processed_rows
        processing_time = (datetime.now() - start_time).total_seconds()

        # Write processed JSON to output folder
        output_path = _write_output_json(df, file.filename or "output")
        logger.info(f"ETL-PROCESS: Written processed JSON to: {output_path}")

        # SUMMARY and numeric stats
        summary = {
            "total_columns": len(df.columns),
            "column_names": df.columns.tolist(),
            "data_types": df.dtypes.astype(str).to_dict(),
            "original_rows": original_rows,
            "processed_rows": processed_rows,
            "rows_removed": rows_removed,
            "original_columns": original_columns,
        }

        numeric_stats = {}
        numeric_columns = df.select_dtypes(include=["float64", "int64"]).columns
        if len(numeric_columns) > 0:
            numeric_stats = df[numeric_columns].describe().to_dict()

        logger.info(
            f"ETL-PROCESS: Transformation completed. Processed {processed_rows} rows in {processing_time:.3f} seconds"
        )

        return {
            "status": "success",
            "message": "File processed and output JSON written successfully",
            "processing_time_seconds": round(processing_time, 3),
            "file_info": {
                "original_filename": file.filename,
                "input_path": input_path,
                "output_path": output_path,
                "file_type": filename.split(".")[-1] if "." in filename else "unknown",
            },
            "summary": summary,
            "numeric_statistics": numeric_stats,
            "preview": df.head(5).to_dict("records"),
        }

    except pd.errors.EmptyDataError:
        logger.error("ETL-PROCESS ERROR: Empty file")
        raise FastAPIHTTPException(status_code=400, detail="The uploaded file is empty")

    except pd.errors.ParserError as e:
        logger.error(f"ETL-PROCESS ERROR: Parser error - {str(e)}")
        raise FastAPIHTTPException(status_code=400, detail=f"Error parsing file: {str(e)}")

    except UnicodeDecodeError as e:
        logger.error(f"ETL-PROCESS ERROR: Encoding error - {str(e)}")
        raise FastAPIHTTPException(status_code=400, detail="File encoding not supported. Please use UTF-8 encoded files.")

    except Exception as e:
        logger.error(f"ETL-PROCESS ERROR: {str(e)}")
        logger.error(f"ETL-PROCESS TRACEBACK: {traceback.format_exc()}")
        raise FastAPIHTTPException(status_code=500, detail=f"ETL processing failed: {str(e)}")
def get_etl_health_check():
    """Health check for ETL functionality"""
    logger.info("ETL-HEALTH: Health check requested")
    try:
        # Test basic pandas functionality
        test_df = pd.DataFrame({'test': [1, 2, 3]})
        test_result = len(test_df)
        
        return {
            "status": "healthy",
            "message": "ETL functionality is working",
            "timestamp": datetime.now().isoformat(),
            "pandas_version": pd.__version__,
            "test_result": f"Successfully processed {test_result} test rows"
        }
    except Exception as e:
        logger.error(f"ETL-HEALTH ERROR: {str(e)}")
        raise FastAPIHTTPException(status_code=500, detail=f"ETL health check failed: {str(e)}")
