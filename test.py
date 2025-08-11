import boto3
from botocore.handlers import disable_signing
from botocore.config import Config
from botocore import UNSIGNED
import re
import pandas as pd 
import io 
import time 




def list_parquet_files(bucket_name, prefix):
    """
    Lists all 'metadata.parquet' files within a given S3 prefix.

    Args:
        bucket_name (str): The name of the S3 bucket.
        prefix (str): The S3 prefix to search within.

    Returns:
        list: A list of S3 object keys (paths) for the found parquet files.
    """
    s3_client = boto3.client('s3',config=Config(signature_version=UNSIGNED))
    paginator = s3_client.get_paginator('list_objects_v2')
    matching_keys = []
    #regex_pattern = r"metadata\/parquet\/year=.*\/court=.*\/bench=.*"

    regex_pattern = r"*.parquet"

    #regex_pattern = re.escape(regex_pattern)


    pages = paginator.paginate(Bucket=bucket_name,Prefix=prefix)

    for page in pages:
        if 'Contents' in page:
            for obj in page['Contents']:
                if ".parquet" in obj['Key']:
                    matching_keys.append(obj['Key'])
    
    dfs = []
    start_time = time.time()
    print("started adding parquet to dataframes")
    i = 0 
    tot = len(matching_keys)
    print(f"total objects to add {tot}")
    for obj in matching_keys:
        data = s3_client.get_object(Bucket=bucket_name,Key=obj)
        buffer = io.BytesIO(data['Body'].read())
        df = pd.read_parquet(buffer)
        dfs.append(df)
        i+=1
        print(f"added {i} object")
    
    end_time = time.time()
    first_execution_time = end_time - start_time



    print(f"Time for first execution: {first_execution_time:.4f} seconds")
    
        

    


    return parquet_files

def test_boto3() :
       bucket = "indian-high-court-judgments"
       base_prefix = "metadata/parquet"

       found_files = list_parquet_files(bucket, base_prefix)

       if found_files:
         print("Found 'metadata.parquet' files:")
         for file_path in found_files:
            print(f"File: {file_path}")
       else:
         print("No 'metadata.parquet' files found in the specified path.")


test_boto3()