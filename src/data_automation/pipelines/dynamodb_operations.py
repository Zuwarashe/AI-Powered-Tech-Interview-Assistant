from decimal import Decimal
import boto3
from datetime import datetime
from typing import Dict, List, Optional
from boto3.dynamodb.conditions import Key

class DynamoDBHandler:
    def __init__(self, table_name: str, region_name: str = "eu-central-1"):
        self.dynamodb = boto3.resource('dynamodb', region_name=region_name)
        self.table = self.dynamodb.Table(table_name)
        
    def _convert_floats_to_decimals(self, data: Dict) -> Dict:
        """Recursively convert all float values in a dictionary to Decimals"""
        if isinstance(data, dict):
            return {k: self._convert_floats_to_decimals(v) for k, v in data.items()}
        elif isinstance(data, list):
            # Special handling for embedding lists
            if data and all(isinstance(x, (float, int)) for x in data):
                return [Decimal(str(x)) for x in data]
            return [self._convert_floats_to_decimals(v) for v in data]
        elif isinstance(data, float):
            return Decimal(str(data))
        return data
        
    def save_resume(self, resume_data: Dict) -> bool:
        try:
            # Convert any floats to Decimals
            resume_data = self._convert_floats_to_decimals(resume_data)
            resume_data['last_updated'] = datetime.utcnow().isoformat()
            self.table.put_item(Item=resume_data)
            return True
        except Exception as e:
            print(f"Error saving resume to DynamoDB: {e}")
            return False
            
    def save_job_description(self, jd_data: Dict) -> bool:
        try:
            # Convert any floats to Decimals
            jd_data = self._convert_floats_to_decimals(jd_data)
            jd_data['last_updated'] = datetime.utcnow().isoformat()
            self.table.put_item(Item=jd_data)
            return True
        except Exception as e:
            print(f"Error saving job description to DynamoDB: {e}")
            return False
            
    def get_resume(self, resume_id: str) -> Optional[Dict]:
        try:
            response = self.table.get_item(Key={'id': resume_id})
            return response.get('Item')
        except Exception as e:
            print(f"Error getting resume from DynamoDB: {e}")
            return None
            
    def get_all_resumes(self) -> List[Dict]:
        try:
            response = self.table.scan(
                FilterExpression=Key('type').eq('resume')  # Only get resumes
            )
            items = response.get('Items', [])
        
            # Debugging - remove after confirmation
            print(f"Found {len(items)} resumes in DynamoDB")
            if items:
                print(f"First resume has embedding: {'embedding' in items[0]}")
                if 'embedding' in items[0]:
                    print(f"Embedding type: {type(items[0]['embedding'])}")
                
            return items
        except Exception as e:
            print(f"Error getting all resumes from DynamoDB: {e}")
            return []
        
            
    def get_all_job_descriptions(self) -> List[Dict]:
        try:
            response = self.table.scan()
            return [item for item in response.get('Items', []) if item.get('type') == 'job_description']
        except Exception as e:
            print(f"Error getting all job descriptions from DynamoDB: {e}")
            return []
            
    def delete_item(self, item_id: str) -> bool:
        try:
            self.table.delete_item(Key={'id': item_id})
            return True
        except Exception as e:
            print(f"Error deleting item from DynamoDB: {e}")
            return False