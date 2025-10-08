import pandas as pd
import os
import io
from typing import Optional, Tuple
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

# Initialize the FastAPI application
app = FastAPI(title="Data Center Cutsheet Processor")

# --- 1. Data Import Function ---

def import_cutsheet(file_name: str) -> Optional[pd.DataFrame]:
    """Imports CSV data into a DataFrame with specified column names."""
    columns = [
        "A-SIDE LOCODE", "A-LOC:CAB:RU", "A-SIDE-DNS-NAME", "A-MODEL", "A-PORT",
        "A-BREAKOUT LOC:CAB:RU", "A-BREAKOUT SLOT:PORT", "A-OPTIC", "A-PATCH-PANEL LOC:CAB:RU:PORT",
        "Z-SIDE LOCODE", "Z-LOC:CAB:RU", "Z-SIDE-DNS-NAME", "Z-MODEL", "Z-PORT",
        "Z-BREAKOUT LOC:CAB:RU", "Z-BREAKOUT SLOT:PORT", "Z-OPTIC", "Z-PATCH-PANEL LOC:CAB:RU:PORT",
        "CABLE"
    ]

    try:
        df = pd.read_csv(file_name, header=None, names=columns)
        return df
    except FileNotFoundError:
        print(f"File '{file_name}' not found.")
        return None
    except pd.errors.EmptyDataError:
        print(f"File '{file_name}' is empty.")
        return None
    except pd.errors.ParserError as e:
        print(f"Error parsing file '{file_name}': {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

# ----------------------------------------------------------------------
# HELPER FUNCTIONS
# ----------------------------------------------------------------------

def split_dns_cfn(dns_series: pd.Series) -> Tuple[pd.Series, pd.Series]:
    """
    Helper: Splits full DNS into Base DNS (everything before CFN) and 
    the 3-part CFN using the last 3 hyphens.
    """
    parts = dns_series.str.rsplit('-', n=3, expand=True)
    base_dns = parts[0]
    
    # Recombine the 3 CFN parts
    cfn = parts[1].fillna('') + '-' + parts[2].fillna('') + '-' + parts[3].fillna('')
    return base_dns, cfn.str.strip('-')

def extract_loc_code(base_dns: pd.Series) -> Tuple[pd.Series, pd.Series]:
    """
    Helper: Conditionally extracts the 'dh' LOC code from the start of the 
    Base DNS and returns the separated LOC and the cleaned Base DNS.
    """
    loc = pd.Series(pd.NA, index=base_dns.index)
    clean_base_dns = base_dns.copy()

    # Conditional check: only split if the string starts with 'dh'
    should_split = clean_base_dns.str.startswith('dh', na=False)
    
    # Perform split only on matching rows (n=1 separates LOC from the rest)
    parts = clean_base_dns[should_split].str.split('-', n=1, expand=True)

    # Assign the split parts
    loc.loc[should_split] = parts[0]
    clean_base_dns.loc[should_split] = parts[1]

    return loc, clean_base_dns

# ----------------------------------------------------------------------
# MAIN PROCESSING STEP FUNCTIONS
# ----------------------------------------------------------------------

def process_location_columns(org_df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Applies split logic to A/Z-LOC:CAB:RU columns."""
    if df is None:
        print("Cannot process location columns: Input DataFrame is None.")
        return None
        
    loc_cols_to_split = [('A-LOC:CAB:RU', 'A'), ('Z-LOC:CAB:RU', 'Z')]
    df = org_df.copy()

    for org_col, prefix in loc_cols_to_split:
        if org_col in df.columns:
            try:
                new_cols = df[org_col].str.split(':', expand=True)
                new_cols.columns = [f'{prefix}-LOC-SITE', f'{prefix}-CAB', f'{prefix}-RU']
                df = pd.concat([df, new_cols], axis=1)
            except Exception as e:
                print(f"Error splitting column {org_col}: {e}")
                return None
        else:
            print(f"Column {org_col} not found in DataFrame.")
            return None
    return df

def process_dns_columns(org_df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Applies modular DNS splitting functions (split_dns_cfn and extract_loc_code) 
    to A/Z DNS columns.
    """
    if org_df is None:
        print("Cannot process DNS columns: Input DataFrame is None.")
        return None
    
    dns_cols_to_split = [('A-SIDE-DNS-NAME', 'A'), ('Z-SIDE-DNS-NAME', 'Z')]

    df = org_df.copy()

    for org_col, prefix in dns_cols_to_split:
        if org_col in df.columns:
            try:
                # 1. Split CFN from the full DNS name
                base_dns, cfn = split_dns_cfn(df[org_col].astype(str))
                
                # 2. Conditionally split LOC from the Base DNS
                loc, clean_base_dns = extract_loc_code(base_dns)

                # 3. Create and concatenate new columns
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

def execute_cutsheet_processing(file_name: str) -> Optional[pd.DataFrame]:
    """Main driver function to execute the full cutsheet processing workflow."""
    df = import_cutsheet(file_name)
    if df is None:
        return None

    df = process_location_columns(df)
    if df is None:
        return None

    df = process_dns_columns(df)
    if df is None:
        return None

    return df

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
        return JSONResponse(content=df_processed.to_dict('records'))

    except HTTPException as e:
        # Re-raise explicit HTTP exceptions from the driver function
        raise e
    except Exception as e:
        # Catch unexpected errors during I/O or other steps
        print(f"An unexpected error occurred: {e}")
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {e}")