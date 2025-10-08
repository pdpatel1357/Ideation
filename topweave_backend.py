import pandas as pd
import os
import io
from typing import Optional, Tuple
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import json

# Initialize the FastAPI application
app = FastAPI(title="Data Center Cutsheet Processor")

# ==============================================================================
# CORE DATA PROCESSING FUNCTIONS
#    Note: These functions need to be defined *before* the API endpoint uses them.
# ==============================================================================

# Data Import Function ---

def import_cutsheet_from_bytes(file_content: bytes) -> Optional[pd.DataFrame]:
    """Reads CSV data from bytes content into a DataFrame."""
    columns = [
        "id", "A-SIDE LOCODE", "A-LOC:CAB:RU", "A-SIDE-DNS-NAME", "A-MODEL", "A-PORT",
        "A-BREAKOUT LOC:CAB:RU", "A-BREAKOUT SLOT:PORT", "A-OPTIC", "A-PATCH-PANEL LOC:CAB:RU:PORT",
        "Z-SIDE LOCODE", "Z-LOC:CAB:RU", "Z-SIDE-DNS-NAME", "Z-MODEL", "Z-PORT",
        "Z-BREAKOUT LOC:CAB:RU", "Z-BREAKOUT SLOT:PORT", "Z-OPTIC", "Z-PATCH-PANEL LOC:CAB:RU:PORT",
        "CABLE"
    ]
    try:
        # Decode bytes to string and use StringIO to treat it like a file
        s = io.StringIO(file_content.decode('utf-8'))
        # df = pd.read_csv(s, header=None, names=columns)
        df = pd.read_csv(s, header=None, skiprows=1, names=columns, skipinitialspace=True)
        df.columns = df.columns.str.strip()

        df['id'] = df['id'].ffill()

        df_filtered = df[df['A-SIDE LOCODE'].notna()].copy()
        df_filtered = df_filtered.reset_index(drop=True)
        
        return df_filtered
    except Exception as e:
        # General error for parsing issues
        print(f"Error reading or parsing file content: {e}")
        return None

# ----------------------------------------------------------------------
# HELPER FUNCTIONS
# ----------------------------------------------------------------------

def split_dns_cfn(dns_series: pd.Series) -> Tuple[pd.Series, pd.Series]:
    """Helper: Splits full DNS into Base DNS and 3-part CFN using the last 3 hyphens."""
    parts = dns_series.str.rsplit('-', n=3, expand=True)
    base_dns = parts[0]
    cfn = parts[1].fillna('') + '-' + parts[2].fillna('') + '-' + parts[3].fillna('')
    return base_dns, cfn.str.strip('-')

def extract_loc_code(base_dns: pd.Series) -> Tuple[pd.Series, pd.Series]:
    """Helper: Conditionally extracts the 'dh' LOC code from the start of the Base DNS."""
    loc = pd.Series(pd.NA, index=base_dns.index)
    clean_base_dns = base_dns.copy()
    should_split = clean_base_dns.str.startswith('dh', na=False)
    parts = clean_base_dns[should_split].str.split('-', n=1, expand=True)
    loc.loc[should_split] = parts[0]
    clean_base_dns.loc[should_split] = parts[1]
    return loc, clean_base_dns

# ----------------------------------------------------------------------
# MAIN PROCESSING STEP FUNCTIONS
# ----------------------------------------------------------------------

def process_location_columns(org_df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Applies split logic to A/Z-LOC:CAB:RU columns."""
    if org_df is None: # FIX: Changed check to org_df
        print("Cannot process location columns: Input DataFrame is None.")
        return None
        
    loc_cols_to_split = [('A-LOC:CAB:RU', 'A'), ('Z-LOC:CAB:RU', 'Z')]
    df = org_df.copy()

    for org_col, prefix in loc_cols_to_split:
        if org_col in df.columns:
            try:
                new_cols = df[org_col].str.split(':', expand=True)
                new_cols.columns = [f'{prefix}-LOC', f'{prefix}-CAB', f'{prefix}-RU']
                df = pd.concat([df, new_cols], axis=1)
            except Exception as e:
                print(f"Error splitting column {org_col}: {e}")
                return None
        else:
            print(f"Column {org_col} not found in DataFrame.")
            return None
    return df

def process_dns_columns(org_df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Applies modular DNS splitting functions to A/Z DNS columns."""
    if org_df is None:
        print("Cannot process DNS columns: Input DataFrame is None.")
        return None
    
    dns_cols_to_split = [('A-SIDE-DNS-NAME', 'A'), ('Z-SIDE-DNS-NAME', 'Z')]

    df = org_df.copy()

    for org_col, prefix in dns_cols_to_split:
        if org_col in df.columns:
            try:
                base_dns, cfn = split_dns_cfn(df[org_col].astype(str))
                loc, clean_base_dns = extract_loc_code(base_dns)

                new_cols = pd.DataFrame({
                    f'{prefix}-DNS-LOC': loc,
                    f'{prefix}-BASE-DNS': clean_base_dns,
                    f'{prefix}-CFN': cfn
                }, index=df.index)

                df = pd.concat([df, new_cols], axis=1)
            except Exception as e:
                print(f"Error splitting column {org_col}: {e}")
                return None
        else:
            print(f"Column {org_col} not found in DataFrame.")
            return None
    return df

# ----------------------------------------------------------------------
# MAIN DRIVER FUNCTION
# ----------------------------------------------------------------------

def execute_cutsheet_processing(file_content: bytes) -> Optional[pd.DataFrame]:
    """Main driver function to execute the full cutsheet processing workflow."""
    
    # 1. Import Data (using the modified in-memory function)
    df = import_cutsheet_from_bytes(file_content)
    if df is None:
        raise HTTPException(status_code=400, detail="Failed to read or parse the CSV file content.")

    # 2. Process Location Columns
    df = process_location_columns(df)
    if df is None:
        raise HTTPException(status_code=500, detail="Failed during location column processing.")

    # 3. Process DNS Columns
    df = process_dns_columns(df)
    if df is None:
        raise HTTPException(status_code=500, detail="Failed during DNS column processing.")

    return df

def get_model_count(df: pd.DataFrame) -> Tuple[dict, dict]:
    if 'A-MODEL' not in df.columns or 'Z-MODEL' not in df.columns:
        print("Model columns not found in DataFrame.")
        return {}, {}
    
    def calculate_counts(series: pd.Series) -> dict:
        cleaned_series = series.astype(str).str.strip()
        cleaned_series = cleaned_series[cleaned_series != 'nan']

        counts = cleaned_series.value_counts()
        return counts.to_dict()
    
    a_model_counts = calculate_counts(df['A-MODEL'])
    z_model_counts = calculate_counts(df['Z-MODEL'])

    return a_model_counts, z_model_counts

# ==============================================================================
# FASTAPI ENDPOINT
# ==============================================================================

@app.post("/process-cutsheet/")
async def process_cutsheet_endpoint(file: UploadFile = File(...)):
    """
    Receives a CSV file, processes it through the data cleaning pipeline,
    and returns the resulting DataFrame as a list of JSON objects.
    """
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a CSV file.")
    
    try:
        # Read the file content into bytes
        file_content = await file.read()
        
        # Process the data using the core Python functions
        df_processed = execute_cutsheet_processing(file_content)
        
        if df_processed is None:
            # This should be caught by HTTPException inside the driver, but acts as a final safeguard
            raise HTTPException(status_code=500, detail="Processing failed for an unknown reason.")

        # Convert the resulting DataFrame to a list of dictionaries (JSON format)
        # return JSONResponse(content=df_processed.to_dict('records'))
        
        # json_str = df_processed.to_json(orient='records')
        # json_data = json.loads(json_str)

        PREVIEW_ROWS = 100
        preview_df = df_processed.head(PREVIEW_ROWS)

        json_str = preview_df.to_json(orient="records")
        json_data = json.loads(json_str)

        return JSONResponse(content=json_data)

    except HTTPException as e:
        # Re-raise explicit HTTP exceptions from the driver function
        raise e
    except Exception as e:
        # Catch unexpected errors during I/O or other steps
        print(f"An unexpected error occurred: {e}")
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {e}")
    
@app.post("/get-model-inventory/")
async def get_model_inventory_endpoint(file: UploadFile = File(...)):
    """
    Receives a CSV file, processes it, and returns the counts of unique models
    in the A-MODEL and Z-MODEL columns.
    """
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a CSV file.")
    
    try:
        # Read the file content into bytes
        file_content = await file.read()
        
        # Process the data using the core Python functions
        df_processed = execute_cutsheet_processing(file_content)
        
        if df_processed is None:
            raise HTTPException(status_code=500, detail="Processing failed before inventory calculation.")

        # Get model counts
        a_inventory, z_inventory = get_model_count(df_processed)
        total_unique_a = len(a_inventory)
        total_unique_z = len(z_inventory)
        total_a_devices = sum(a_inventory.values())
        total_z_devices = sum(z_inventory.values())

        return JSONResponse(content={
            "Total Unique A-MODEL Devices": total_unique_a,
            "Total A-MODEL Devices": total_a_devices,
            "A-MODEL Device Inventory": a_inventory,
            "Total Unique Z-MODEL Devices": total_unique_z,
            "Total Z-MODEL Devices": total_z_devices,
            "Z-MODEL Device Inventory": z_inventory
        })

    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {e}")