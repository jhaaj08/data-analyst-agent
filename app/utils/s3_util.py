import boto3
from botocore.handlers import disable_signing
from botocore.config import Config
from botocore import UNSIGNED
from typing import List

class S3_Util:

    def __init__(self):
            self.s3_client = boto3.client('s3',config=Config(signature_version=UNSIGNED))

    
    def get_s3_file_list(self,bucket_name,prefix) -> List[str] :
            files = []
            paginator = self.s3_client.get_paginator('list_objects_v2')
            pages = paginator.paginate(Bucket=bucket_name,Prefix=prefix)
            for page in pages:
                if 'Contents' in page:
                    for obj in page['Contents']:
                        if ".parquet" in obj['Key']:
                            files.append(obj['Key'])
            return files



    

    
