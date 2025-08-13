"""
Simple FastAPI route for parsing questions into data sources and questions lists
"""
from fastapi import APIRouter, File, UploadFile
from typing import Dict, Any, List
from ..utils.s3_util import S3_Util
import json
import re
import sys
import os
import tiktoken
import asyncio


# Add root directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))


try:
    from app.api.question_answer import answer_questions_with_llm
    QUESTION_ANSWERER_AVAILABLE = True
    print("✅ Question Answerer imported successfully")
except ImportError as e:
    print(f"⚠️ Question Answerer not available: {e}")
    QUESTION_ANSWERER_AVAILABLE = False

# Add the parent directory to path to import scrapping module
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from app.utils import s3_util
from scrapping import scrape_data_sources

from app.core.llm_client import LLMClient

router = APIRouter()

# Initialize LLM client
llm_client = LLMClient()


@router.post("/parse")
async def parse_question(
    file: UploadFile = File(..., description="Question file (question.txt)")
):
    """
    Simple parser: Question text → Data sources list + Questions list
    
    Returns:
    - data_sources: List of URLs/sources found
    - questions: List of individual questions
    """
    try:
        # Read the question file
        question_content = await file.read()
        question_text = question_content.decode('utf-8')
        
        print(f"📝 Parsing question ({len(question_text)} chars)")
        
        # Parse using LLM
        parsed_data = await _parse_with_llm(question_text)
        
        # Extract clean lists
        data_sources = _extract_data_sources(parsed_data)
        questions = _extract_questions(parsed_data)
        
        return {
            "success": True,
            "data_sources": data_sources,
            "questions": questions,
            "counts": {
                "data_sources": len(data_sources),
                "questions": len(questions)
            },
            "original_text": question_text
        }
        
    except Exception as e:
        print(f"❌ Parsing failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "data_sources": [],
            "questions": []
        }


@router.post("/scrape")
async def scrape_from_file(
    file: UploadFile = File(..., description="Question file (question.txt)")
):
    """
    Complete pipeline: Parse question file + Scrape data sources
    
    Returns:
    - parsed data (data sources + questions)
    - scraped results (DataFrames info)
    """
    try:
        # Step 1: Parse the question file
        question_content = await file.read()
        question_text = question_content.decode('utf-8')
        
        print(f"📝 Parsing question ({len(question_text)} chars)")
        parsed_data = await _parse_with_llm(question_text)
        
        # Extract data sources and questions
        data_sources = _extract_data_sources(parsed_data)
        questions = _extract_questions(parsed_data)
        applicable_sources = await _extract_applicable_sources(questions,data_sources)
        
        print(f"🔍 Found {len(applicable_sources)} data sources, {len(questions)} questions")
        
        # Step 2: Scrape the data sources
        if data_sources:
            print(f"📡 Scraping {len(data_sources)} data sources...")
            scrape_results = scrape_data_sources(applicable_sources)
            
            # Remove the actual DataFrame from results (too large for JSON)
            clean_results = []
            for result in scrape_results:
                clean_result = {k: v for k, v in result.items() if k != 'dataframe'}
                clean_results.append(clean_result)
        else:
            clean_results = []
        
        return {
            "success": True,
            "message": f"Parsed and scraped {len(data_sources)} data sources",
            "parsed": {
                "data_sources": data_sources,
                "questions": questions,
                "counts": {
                    "data_sources": len(data_sources),
                    "questions": len(questions)
                }
            },
            "scraped": clean_results
        }
        
    except Exception as e:
        print(f"❌ Scraping pipeline failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "message": "Failed to parse and scrape data"
        }


@router.post("/")
async def analyze_complete_pipeline(
    file: UploadFile = File(..., description="Question file (question.txt)")
):
    """
    COMPLETE PIPELINE: Parse + Scrape + Answer Questions
    """
    try:
        print("🚀 STARTING COMPLETE ANALYSIS PIPELINE")
        
        # Step 1: Parse
        question_content = await file.read()
        question_text = question_content.decode('utf-8')
        
        print(f"📝 STEP 1: Parsing...")
        parsed_data = await _parse_with_llm(question_text)
        data_sources = _extract_data_sources(parsed_data)
        questions = _extract_questions(parsed_data)
        applicable_sources = await _extract_applicable_sources(questions,data_sources)
        print(f"✅ Found {len(applicable_sources)} sources, {len(questions)} questions")
        
        # Step 2: Scrape
        scrape_results = []
        if data_sources:
            print(f"📡 STEP 2: Scraping...")
            scrape_results = scrape_data_sources(applicable_sources)
            print(f"✅ Scraped {len(scrape_results)} sources")
        
        # Step 3: Answer Questions
        answers = []
        if questions and scrape_results and QUESTION_ANSWERER_AVAILABLE:
            print(f"🤔 STEP 3: Answering questions...")
            answers = answer_questions_with_llm(scrape_results, questions)
            print(f"✅ Generated {len(answers)} answers")
        else:
            print("⚠️ Skipping question answering")
            answers = [{"error": "Question answerer not available", "success": False}]
        
        # Create clean results (remove DataFrames for JSON)
        clean_results = []
        for result in scrape_results:
            clean_result = {k: v for k, v in result.items() if k != 'dataframe'}
            clean_results.append(clean_result)
        
        # 🎯 EXTRACT JUST THE ANSWER VALUES AS A SIMPLE LIST
        simple_answers = []
        for answer_dict in answers:
            if answer_dict.get('success', False):
                simple_answers.append(answer_dict.get('answer', 'No answer'))
            else:
                simple_answers.append(f"Error: {answer_dict.get('error', 'Failed')}")
        
        return simple_answers  # ✅ Return simple list instead of complex dict
        
    except Exception as e:
        print(f"❌ Complete pipeline failed: {e}")
        return [f"Pipeline Error: {str(e)}"]  # ✅ Return error as simple list too


@router.get("/health")
async def health_check():
    """Health check"""
    return {"status": "healthy", "service": "question-parser"}


# LLM Parsing
async def _parse_with_llm(question_text: str) -> Dict[str, Any]:
    """
    Use LLM to parse question text into structured data
    """
    
    prompt = f"""
    Parse this data analysis request and extract:
    
    REQUEST:
    {question_text}
    
    Return JSON with:
    1. data_sources: Array of URLs  or data sources found.if data_sources has s3 bucket path , return bucket name and prefix separated by colon. prefix should be till the point without regex.
    2. s3_paths : if data_sources has s3 bucket path , return bucket name and prefix separated by colon. prefix should be till the point without regex.
    3. questions: Array of individual questions/tasks to answer
    4. format_requirements: Any output format requirements (JSON array, base64, etc.)
    
    Example:
    {{
        "data_sources": ["https://example.com/data.csv"],
        "s3_paths": ["bucket_name:prefix"],
        "questions": ["How many records?", "What is the average?"],
        "format_requirements": "JSON array response"
    }}
    
    Respond only with valid JSON.
    """
    
    # Try OpenAI
    if llm_client.openai_client:
        try:
            print("🤖 Using OpenAI...")
            response = llm_client.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            result = json.loads(response.choices[0].message.content)
            result["_parser"] = "openai"
            return result
        except Exception as e:
            print(f"❌ OpenAI failed: {e}")
    
    # Try Gemini fallback
    if llm_client.gemini_client:
        try:
            print("🤖 Using Gemini...")
            response = llm_client.gemini_client.generate_content(prompt)
            result = json.loads(response.text)
            result["_parser"] = "gemini"
            return result
        except Exception as e:
            print(f"❌ Gemini failed: {e}")
    
    # Regex fallback
    print("🤖 Using regex fallback...")
    return _parse_with_regex(question_text)



async def _parse_valid_sources(question_text: str,data_source : str) -> Dict[str, Any]:
    """
    Use LLM to parse question text into structured data
    """
    
    prompt = f"""
    filter valid sources from data_source which can help answer question_text:
    
    REQUEST:
    {question_text,data_source}
    
    Return JSON with:
    1. valid_source : return data source 

    
    Example:
    {{
        "valid_source": ["data source"],
    }}
    
    Respond only with valid JSON.
    """
    
    # Try OpenAI
    if llm_client.openai_client:
        try:
            print("🤖 Using OpenAI...")
            response = llm_client.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            result = json.loads(response.choices[0].message.content)
            result["_parser"] = "openai"
            return result
        except Exception as e:
            return {"valid_source" : []}
    

    return {"valid_source" : []}
    
    # Try Gemini fallback
    # if llm_client.gemini_client:
    #     try:
    #         print("🤖 Using Gemini...")
    #         response = llm_client.gemini_client.generate_content(prompt)
    #         result = json.loads(response.text)
    #         result["_parser"] = "gemini"
    #         return result
    #     except Exception as e:
    #         print(f"❌ Gemini failed: {e}")


def _parse_with_regex(question_text: str) -> Dict[str, Any]:
    """
    Fallback regex parsing
    """
    # Extract URLs
    url_pattern = r'https?://[^\s\n]+'
    urls = re.findall(url_pattern, question_text)
    
    # Extract numbered questions
    question_pattern = r'\d+\.\s*([^?\n]+\??)'
    questions = re.findall(question_pattern, question_text)
    
    # Format requirements
    format_req = "standard"
    if "json array" in question_text.lower():
        format_req = "JSON array"
    if "base64" in question_text.lower():
        format_req += " with base64 images"
    
    return {
        "data_sources": urls,
        "questions": questions if questions else [question_text.strip()],
        "format_requirements": format_req,
        "_parser": "regex"
    }


# Helper functions
def _extract_data_sources(parsed_data: Dict[str, Any]) -> List[str]:
    """
    Extract clean list of data source URLs
    """
    sources = parsed_data.get("data_sources", [])
    s3_paths = parsed_data.get("s3_paths",[])

    
    if not s3_paths:
        if isinstance(sources, str):
            sources = [sources]
    else:
        sources = []
        s3Util = S3_Util()
        for s3_path in s3_paths:
            path_var = s3_path.split(":",1)
            sources.extend(s3Util.get_s3_file_list(path_var[0],path_var[1]))

    
    # Remove empty strings and duplicates
    clean_sources = []
    for source in sources:
        if source and source.strip():
            clean_sources.append(source.strip())
    
    return list(set(clean_sources))  # Remove duplicates


def _extract_questions(parsed_data: Dict[str, Any]) -> List[str]:
    """
    Extract clean list of questions
    """
    questions = parsed_data.get("questions", [])
    if isinstance(questions, str):
        questions = [questions]
    
    # Remove empty strings
    clean_questions = []
    for question in questions:
        if question and question.strip():
            clean_questions.append(question.strip())
    
    return clean_questions


async def _extract_applicable_sources(questions : List[str],data_sources: List[str]) -> List[str]:
    num_of_sources = len(data_sources)
    if num_of_sources  < 10 :
       return data_sources

    filtered_source = []
    filtered_sources = []
    applicable_sources = []
    for index,source in enumerate(data_sources):
       num_of_tokens =  num_tokens_from_string(",".join(filtered_source))
       if num_of_tokens < 3000:
            filtered_source.append(source)
       elif num_of_tokens >= 3000 or  index == num_of_sources-1 :
            filtered_sources.append(filtered_source)
            filtered_source =[] 

    tasks = [_parse_valid_sources(questions,fs) for fs in filtered_sources]
    results = await asyncio.gather(*tasks)

    for i, result in enumerate(results):
         applicable_sources.extend(result.get("valid_source", []))
    

    print(f"parsing complete with {len(applicable_sources)}")
    return applicable_sources

    






def num_tokens_from_string(string: str, encoding_name: str = "cl100k_base") -> int:
        """Returns the number of tokens in a text string."""
        encoding = tiktoken.get_encoding(encoding_name)
        num_tokens = len(encoding.encode(string))
        return num_tokens