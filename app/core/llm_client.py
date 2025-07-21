"""
Enhanced LLM Client with OpenAI + Gemini fallback
"""
import json
import numpy as np
from typing import Dict, Any, Optional, List
from openai import OpenAI

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

from .config import settings


def make_json_safe(obj):
    """Convert numpy types to JSON-serializable types"""
    if isinstance(obj, dict):
        return {key: make_json_safe(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [make_json_safe(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(make_json_safe(item) for item in obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif hasattr(obj, 'item'):  # numpy scalar
        return obj.item()
    else:
        return obj

def safe_json_dumps(obj, indent=2):
    """JSON dumps that handles numpy types"""
    safe_obj = make_json_safe(obj)
    return json.dumps(safe_obj, indent=indent)


class LLMClient:
    """Client for interacting with multiple LLM providers"""
    
    def __init__(self):
        # Initialize OpenAI client (primary)
        self.openai_client = None
        if settings.openai_api_key:
            try:
                self.openai_client = OpenAI(api_key=settings.openai_api_key)
                print("✅ OpenAI client initialized")
            except Exception as e:
                print(f"❌ OpenAI initialization failed: {e}")
        
        # Initialize Gemini client (backup)
        self.gemini_client = None
        if settings.gemini_api_key and GEMINI_AVAILABLE:
            try:
                genai.configure(api_key=settings.gemini_api_key)
                self.gemini_client = genai.GenerativeModel('gemini-pro')
                print("✅ Gemini client initialized")
            except Exception as e:
                print(f"❌ Gemini initialization failed: {e}")
    
    async def parse_analysis_request(self, question: str, data_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse a natural language analysis request with multi-provider fallback
        """
        prompt = f"""
        Analyze the following data analysis request and create a structured plan:
        
        Question: {question}
        Data Info: {safe_json_dumps(data_info)}
        
        Return a JSON object with:
        1. analysis_type: (descriptive, diagnostic, predictive, prescriptive)
        2. data_operations: List of required data processing steps
        3. analysis_steps: List of analysis operations to perform
        4. visualization_requirements: List of charts/visualizations needed
        5. expected_outputs: Description of expected results
        
        Respond only with valid JSON.
        """
        
        # Try OpenAI first (primary)
        if self.openai_client:
            try:
                response = self.openai_client.chat.completions.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1
                )
                print("🤖 Using OpenAI GPT-4")
                return json.loads(response.choices[0].message.content)
            except Exception as e:
                print(f"❌ OpenAI failed: {e}")
                print("🔄 Falling back to Gemini...")
        
        # Fallback to Gemini
        if self.gemini_client:
            try:
                response = self.gemini_client.generate_content(prompt)
                print("🤖 Using Google Gemini")
                return json.loads(response.text)
            except Exception as e:
                print(f"❌ Gemini failed: {e}")
                print("🔄 Using basic fallback...")
        
        # Final fallback to hardcoded logic
        print("🤖 Using basic fallback")
        return self._basic_parse(question, data_info)
    
    async def generate_analysis_code(self, analysis_plan: Dict[str, Any], data_schema: Dict[str, Any]) -> str:
        """
        Generate Python code with multi-provider fallback
        """
        prompt = f"""
        Generate Python code for data analysis based on this plan:
        
        Analysis Plan: {safe_json_dumps(analysis_plan)}
        Data Schema: {safe_json_dumps(data_schema)}
        
        Generate complete Python code using pandas, numpy, matplotlib, seaborn, and plotly.
        The code should:
        1. Process the data according to data_operations
        2. Perform the analysis steps
        3. Create the required visualizations
        4. Return results as a dictionary
        
        Assume the data is loaded in a pandas DataFrame called 'df'.
        """
        
        # Try OpenAI first
        if self.openai_client:
            try:
                response = self.openai_client.chat.completions.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1
                )
                return response.choices[0].message.content
            except Exception as e:
                print(f"OpenAI code generation failed: {e}")
        
        # Fallback to Gemini
        if self.gemini_client:
            try:
                response = self.gemini_client.generate_content(prompt)
                return response.text
            except Exception as e:
                print(f"Gemini code generation failed: {e}")
        
        # Final fallback
        return self._basic_analysis_code()
    
    def _basic_parse(self, question: str, data_info: Dict[str, Any]) -> Dict[str, Any]:
        """Basic fallback parsing when all LLMs fail"""
        return {
            "analysis_type": "descriptive",
            "data_operations": ["clean_data", "basic_stats"],
            "analysis_steps": ["summary_statistics", "correlation_analysis"],
            "visualization_requirements": ["histogram", "correlation_heatmap"],
            "expected_outputs": "Basic statistical analysis and visualizations"
        }
    
    def _basic_analysis_code(self) -> str:
        """Basic fallback analysis code"""
        return """
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Basic analysis
results = {}
results['shape'] = df.shape
results['columns'] = df.columns.tolist()
results['dtypes'] = df.dtypes.to_dict()
results['missing_values'] = df.isnull().sum().to_dict()
results['numeric_summary'] = df.describe().to_dict()

# Create basic visualization
plt.figure(figsize=(10, 6))
df.hist(bins=20, figsize=(15, 10))
plt.tight_layout()
plt.savefig('analysis_histogram.png')
plt.close()

results['visualizations'] = ['analysis_histogram.png']
""" 