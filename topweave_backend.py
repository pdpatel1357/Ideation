import pandas as pd
import os
import io
from typing import Optional, Tuple, Dict, Any, List
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

        for col in df_filtered.select_dtypes(include=['object']).columns:
            df_filtered[col] = df_filtered[col].str.lower()
        
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

def create_graph_data(df: pd.DataFrame) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Transforms the processed DataFrame into a list of Nodes and a list of Edges
    suitable for graph visualization.
    """
    if df is None or df.empty:
        print("Input DataFrame is None or empty.")
        return [], []
    
    nodes: Dict[str, Dict[str, Any]] = {}
    edges: List[Dict[str, Any]] = []

    # Helper function to ensure we get a clean string or the specific 'NaN' string
    def get_clean_value(value):
        """Checks for NaN and returns 'NaN' string, otherwise converts to stripped string."""
        if pd.isna(value):
            return 'NaN'
        # Convert to string and strip any lingering whitespace.
        # Note: Lowercasing is handled in the execute_cutsheet_processing pipeline.
        return str(value).strip()

    device_id_cols = {
        'A': ['A-LOC:CAB:RU', 'A-BASE-DNS', 'A-MODEL', 'A-PORT', 'A-OPTIC'],
        'Z': ['Z-LOC:CAB:RU', 'Z-BASE-DNS', 'Z-MODEL', 'Z-PORT', 'Z-OPTIC']
    }

    for index, row in df.iterrows():
        # a_loc = row['A-LOC:CAB:RU'].strip()
        # a_dns = row['A-BASE-DNS'].strip()
        # a_node_id = f"{a_dns}-{a_loc}"
        a_loc = get_clean_value(row['A-LOC:CAB:RU'])
        a_dns = get_clean_value(row['A-BASE-DNS'])
        a_node_id = f"{a_dns}-{a_loc}"


        if a_node_id not in nodes:
            nodes[a_node_id] = {
                'id': a_node_id,
                'location': a_loc,
                'base_dns': a_dns,
                # 'model': row['A-MODEL'].strip(),
                'model': get_clean_value(row['A-MODEL']),
                'side': 'A'
            }
        
        # z_loc = row['Z-LOC:CAB:RU'].strip()
        # z_dns = row['Z-BASE-DNS'].strip()
        # z_node_id = f"{z_dns}-{z_loc}"
        z_loc = get_clean_value(row['Z-LOC:CAB:RU'])
        z_dns = get_clean_value(row['Z-BASE-DNS'])
        z_node_id = f"{z_dns}-{z_loc}"

        if z_node_id not in nodes:
            nodes[z_node_id] = {
                'id': z_node_id,
                'location': z_loc,
                'base_dns': z_dns,
                # 'model': row['Z-MODEL'].strip(),
                'model': get_clean_value(row['Z-MODEL']),
                'side': 'Z'
            }
        elif nodes[z_node_id]['side'] == 'A':
            nodes[z_node_id]['side'] = 'Both'

        edge_id = f"{a_node_id} <-> {z_node_id}"
        edges.append({
            'id': edge_id,
            'source': a_node_id,
            'target': z_node_id,
            # 'a_port': row['A-PORT'].strip(),
            # 'a_optic': row['A-OPTIC'].strip(),
            # 'z_port': row['Z-PORT'].strip(),
            # 'z_optic': row['Z-OPTIC'].strip(),
            # 'cable': row['CABLE'].strip()
            'a_port': get_clean_value(row['A-PORT']),
            'a_optic': get_clean_value(row['A-OPTIC']),
            'z_port': get_clean_value(row['Z-PORT']),
            'z_optic': get_clean_value(row['Z-OPTIC']),
            'cable': get_clean_value(row['CABLE'])
        })

        if nodes[a_node_id]['side'] == 'Z':
            nodes[a_node_id]['side'] = 'Both'
        
    return list(nodes.values()), edges

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

        a_models = set(a_inventory.keys())
        z_models = set(z_inventory.keys())
        combined_unique_models = a_models | z_models
        total_combined_unique = len(combined_unique_models)

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
            "Z-MODEL Device Inventory": z_inventory,

            "Total Combined Unique Models (A or Z)": total_combined_unique,
            "Combined Unique Model List": list(combined_unique_models)
        })

    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {e}")

@app.post("/get-graph-data/")
async def get_graph_data_endpoint(file: UploadFile = File(...)):
    """
    Receives a CSV file, processes it, and returns the data structured as 
    Nodes and Edges for graph visualization.
    """
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a CSV file.")
    
    try:
        # Read the file content into bytes
        file_content = await file.read()
        
        # Process the data using the core Python functions
        # This returns the fully processed DataFrame
        df_processed = execute_cutsheet_processing(file_content)
        
        if df_processed is None:
            raise HTTPException(status_code=500, detail="Processing failed before graph data creation.")

        # # Transform DataFrame into Nodes and Edges
        # nodes, edges = create_graph_data(df_processed)

        # return JSONResponse(content={
        #     "nodes": nodes,
        #     "edges": edges
        # })
        
        all_nodes, all_edges = create_graph_data(df_processed)

        MAX_EDGES = 100
        
        # 1. Limit the edges list (take the first N edges)
        limited_edges = all_edges[:MAX_EDGES]
        
        # 2. Identify all unique node IDs present in the limited edges
        # This ensures we only return nodes that are actually connected in the preview graph
        involved_node_ids = set()
        for edge in limited_edges:
            involved_node_ids.add(edge['source'])
            involved_node_ids.add(edge['target'])
            
        # 3. Filter the full list of nodes based on the involved IDs
        limited_nodes = [
            node for node in all_nodes
            if node['id'] in involved_node_ids
        ]
        
        print(f"Graph Data Limited: Returning {len(limited_nodes)} nodes and {len(limited_edges)} edges.")
        # ----------------------------------------------------------------------

        return JSONResponse(content={
            "nodes": limited_nodes,
            "edges": limited_edges
        })

    except HTTPException as e:
        # Re-raise explicit HTTP exceptions
        raise e
    except Exception as e:
        # Catch unexpected errors
        print(f"An unexpected error occurred: {e}")
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {e}")