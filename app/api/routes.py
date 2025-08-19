"""
Simple FastAPI route for parsing questions into data sources and questions lists
"""
from fastapi import APIRouter, File, Request, UploadFile
from typing import Dict, Any, List,Optional
from ..utils.s3_util import S3_Util
import json
import re
import sys
import os
import asyncio
import uuid
import datetime
import pandas as pd
import numpy as np
import logging

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
from app.core.config import settings

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


# Add the generic JSON detection function before the main endpoint

def _detect_and_build_json_response(questions_text: str, raw_answers: list) -> dict:
    """Generic JSON object detection and building from questions text"""
    import re
    
    questions_lower = questions_text.lower()
    
    # Check for JSON object request patterns
    json_indicators = [
        "return a json object",
        "json object with keys",
        "return json",
        "output json",
        "respond with json"
    ]
    
    is_json_request = any(indicator in questions_lower for indicator in json_indicators)
    
    if not is_json_request:
        return None  # Return None to indicate array format should be used
    
    # Extract expected keys generically using the `key_name` pattern
    key_patterns = re.findall(r'-\s*`([^`]+)`\s*:', questions_text)
    
    if not key_patterns:
        return None  # No keys found, fallback to array
    
    # Build JSON object by mapping answers to keys in order
    json_response = {}
    
    for i, key in enumerate(key_patterns):
        if i < len(raw_answers):
            json_response[key] = raw_answers[i]
        else:
            json_response[key] = None  # Missing answer
    
    return json_response


def log_evaluation_request(analysis_id: str, request_data: dict, questions_content: str = None, files_info: list = None):
    """Log detailed evaluation request information to file"""
    try:
        log_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "analysis_id": analysis_id,
            "request_info": request_data,
            "questions_content": questions_content,
            "files_info": files_info
        }
        
        # Ensure logs directory exists
        os.makedirs("logs", exist_ok=True)
        
        # Write to log file
        log_filename = f"logs/evaluation_requests_{datetime.datetime.now().strftime('%Y-%m-%d')}.jsonl"
        with open(log_filename, "a", encoding='utf-8') as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
            
        print(f"📝 Logged request details to: {log_filename}")
        
    except Exception as e:
        print(f"⚠️ Failed to write log file: {e}")


@router.post("/")
async def analyze_complete_pipeline(request: Request):
    """
    Phase 1 + 2: Accept multipart form data with file size validation
    
    Expected usage:
    curl "https://app.example.com/api/" \
         -F "questions.txt=@question.txt" \
         -F "image.png=@image.png" \
         -F "data.csv=@data.csv"
    
    Returns:
    - analysis_id: Unique identifier for this request
    - files: Dictionary grouping files by type
    - questions_text_length: Length of the questions content
    """
    # ====== COMPREHENSIVE REQUEST LOGGING ======
    request_timestamp = datetime.datetime.now().isoformat()
    analysis_id = str(uuid.uuid4())
    
    print("\n" + "="*80)
    print(f"🔍 EVALUATION REQUEST LOG - {request_timestamp}")
    print(f"📋 Analysis ID: {analysis_id}")
    print(f"🌐 Client IP: {request.client.host if request.client else 'Unknown'}")
    print(f"🎯 Endpoint Hit: {request.url}")
    print(f"📡 Method: {request.method}")
    print(f"📋 Headers: {dict(request.headers)}")
    print("="*80)
    
    try:
        print("🚀 PHASE 2: Processing multipart form data with validation")
        
        # Get form data
        form = await request.form()
        
        if not form:
            print("❌ ERROR: No form data provided")
            return {
                "success": False,
                "error": "No form data provided"
            }
        
        # ====== LOG ALL FORM FIELDS ======
        print(f"\n📦 FORM DATA RECEIVED:")
        print(f"   Total fields: {len(form.multi_items())}")
        
        for field_name, field_data in form.multi_items():
            if hasattr(field_data, 'filename') and field_data.filename:
                print(f"   📁 File: {field_name} = {field_data.filename} ({field_data.content_type})")
            else:
                print(f"   📝 Field: {field_name} = [non-file data]")
        
        # Look for required questions.txt
        questions_file = None
        other_files = []
        
        print(f"\n🔍 PROCESSING FILES:")
        
        for field_name, file_data in form.multi_items():
            if field_name == "questions.txt":
                questions_file = file_data
                print(f"   ✅ Found questions.txt")
            elif hasattr(file_data, 'filename') and file_data.filename:
                # Read file content to check size
                file_content = await file_data.read()
                file_size = len(file_content)
                
                print(f"   📁 Processing file: {file_data.filename}")
                print(f"      - Size: {file_size} bytes")
                print(f"      - Content-Type: {file_data.content_type}")
                
                # Validate file size
                if file_size > settings.max_file_size:
                    return {
                        "success": False,
                        "error": f"File '{file_data.filename}' ({file_size:,} bytes) exceeds maximum size limit of {settings.max_file_size:,} bytes"
                    }
                
                # Reset file position for later use
                await file_data.seek(0)
                
                # Validate file type
                file_ext = os.path.splitext(file_data.filename)[1].lower() if file_data.filename else ""
                allowed_extensions = {'.csv', '.json', '.xlsx', '.xls', '.parquet', '.tsv', 
                                    '.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', 
                                    '.txt', '.md', '.log'}
                
                if file_ext not in allowed_extensions:
                    allowed_list = ', '.join(sorted(allowed_extensions))
                    return {
                        "success": False,
                        "error": f"File '{file_data.filename}' has unsupported extension '{file_ext}'. Allowed types: {allowed_list}"
                    }
                
                other_files.append({
                    "field_name": field_name,
                    "filename": file_data.filename,
                    "content_type": getattr(file_data, 'content_type', 'unknown'),
                    "file_object": file_data,
                    "file_size": file_size
                })
        
        # Validate required questions.txt
        if not questions_file:
            return {
                "success": False,
                "error": "Missing required 'questions.txt' field in form data"
            }
        
        # Generate unique analysis ID
        # analysis_id = str(uuid.uuid4()) # This line is now redundant as analysis_id is generated above
        
        # Read and validate questions content
        try:
            questions_content = await questions_file.read()
            
            # Validate questions.txt size
            if len(questions_content) > settings.max_file_size:
                return {
                    "success": False,
                    "error": f"questions.txt ({len(questions_content):,} bytes) exceeds maximum size limit of {settings.max_file_size:,} bytes"
                }
            
            questions_text = questions_content.decode('utf-8')
        except UnicodeDecodeError:
            return {
                "success": False,
                "error": "questions.txt must be valid UTF-8 encoded text"
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to read questions.txt: {str(e)}"
            }
        
        # Create analysis directory and save files
        analysis_dir = os.path.join(settings.upload_dir, analysis_id)
        os.makedirs(analysis_dir, exist_ok=True)
        print(f"📁 Created analysis directory: {analysis_dir}")
        
        # Save questions.txt first with enhanced metadata
        questions_file_path = os.path.join(analysis_dir, "questions.txt")
        questions_save_time = datetime.datetime.now()
        
        with open(questions_file_path, "wb") as f:
            f.write(questions_content)
        
        # Get questions file stats
        questions_stats = os.stat(questions_file_path)
        questions_metadata = {
            "saved_path": questions_file_path,
            "size_on_disk": questions_stats.st_size,
            "saved_at": questions_save_time.isoformat(),
            "verified": questions_stats.st_size == len(questions_content)
        }
        print(f"💾 Saved questions.txt: {questions_file_path} ({questions_stats.st_size} bytes)")
        
        # Save other files with enhanced metadata and organized directories
        successful_files = []
        failed_files = []
        
        for file_info in other_files:
            try:
                file_obj = file_info["file_object"]
                filename = file_info["filename"]
                file_ext = os.path.splitext(filename)[1].lower() if filename else ""
                
                # Determine file category and subdirectory
                data_extensions = {'.csv', '.json', '.xlsx', '.xls', '.parquet', '.tsv'}
                image_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff'}
                text_extensions = {'.txt', '.md', '.log'}
                
                if file_ext in data_extensions:
                    subdir = "data"
                elif file_ext in image_extensions:
                    subdir = "images"
                elif file_ext in text_extensions:
                    subdir = "text"
                else:
                    subdir = "other"
                
                # Create subdirectory if it doesn't exist
                type_dir = os.path.join(analysis_dir, subdir)
                os.makedirs(type_dir, exist_ok=True)
                
                # Read file content
                file_content = await file_obj.read()
                
                # Save to organized subdirectory
                file_path = os.path.join(type_dir, filename)
                save_time = datetime.datetime.now()
                
                with open(file_path, "wb") as f:
                    f.write(file_content)
                
                # Get file stats and create enhanced metadata
                file_stats = os.stat(file_path)
                
                # Update file_info with enhanced metadata
                file_info.update({
                    "saved_path": file_path,
                    "relative_path": os.path.join(subdir, filename),
                    "category": subdir,
                    "size_on_disk": file_stats.st_size,
                    "saved_at": save_time.isoformat(),
                    "verified": file_stats.st_size == len(file_content),
                    "upload_size": len(file_content),
                    "status": "success"
                })
                
                successful_files.append(file_info)
                print(f"💾 Saved {filename} → {subdir}/{filename}: {file_path} ({file_stats.st_size} bytes)")
                
            except Exception as e:
                # Handle individual file failure
                error_info = {
                    "filename": file_info.get("filename", "unknown"),
                    "field_name": file_info.get("field_name", "unknown"),
                    "error": str(e),
                    "status": "failed"
                }
                failed_files.append(error_info)
                print(f"❌ Failed to save {file_info.get('filename', 'unknown')}: {e}")
        
        # Check if we have any successful files to continue with
        if not successful_files and failed_files:
            # All files failed - cleanup and return error
            print(f"❌ All {len(failed_files)} files failed to save. Cleaning up...")
            _cleanup_analysis_directory(analysis_dir)
            return {
                "success": False,
                "error": f"Failed to save all {len(failed_files)} files",
                "failed_files": failed_files,
                "analysis_id": analysis_id
            }

        # Categorize other files by type with enhanced metadata
        files_by_type = {
            "data": [],
            "images": [],
            "text": [],
            "other": []
        }
        
        # Define file type mappings (reuse for consistency)
        data_extensions = {'.csv', '.json', '.xlsx', '.xls', '.parquet', '.tsv'}
        image_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff'}
        text_extensions = {'.txt', '.md', '.log'}
        
        # Load data files into DataFrames
        loaded_data = []
        data_files = [f for f in successful_files if f["category"] == "data"]
        
        print(f"📊 Loading {len(data_files)} data files into DataFrames...")
        
        for file_info in data_files:
            try:
                file_path = file_info["saved_path"]
                filename = file_info["filename"]
                file_ext = os.path.splitext(filename)[1].lower()
                
                # Load DataFrame based on file type
                df = _load_dataframe_from_file(file_path, file_ext)
                
                if df is not None and not df.empty:
                    # Create data summary
                    data_summary = {
                        "filename": filename,
                        "file_path": file_path,
                        "shape": [int(df.shape[0]), int(df.shape[1])],
                        "columns": df.columns.tolist(),
                        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
                        "sample_data": df.head(3).to_dict('records'),
                        "memory_usage_mb": round(float(df.memory_usage(deep=True).sum() / (1024 * 1024)), 3),
                        "dataframe": df,  # Include for analysis
                        "loaded_successfully": True
                    }
                    
                    # Add to file_info for reference
                    file_info["data_loaded"] = True
                    file_info["data_shape"] = data_summary["shape"]
                    file_info["column_count"] = len(df.columns)
                    
                    loaded_data.append(data_summary)
                    print(f"✅ Loaded {filename}: {df.shape[0]} rows × {df.shape[1]} columns")
                    
                else:
                    file_info["data_loaded"] = False
                    file_info["load_error"] = "Empty or invalid DataFrame"
                    print(f"⚠️ Failed to load {filename}: Empty or invalid data")
                    
            except Exception as e:
                file_info["data_loaded"] = False
                file_info["load_error"] = str(e)
                print(f"❌ Failed to load {filename}: {e}")
        
        # Update file summaries with loading info
        for file_info in successful_files:
            filename = file_info["filename"]
            file_ext = os.path.splitext(filename)[1].lower()
            
            file_summary = {
                "field_name": file_info["field_name"],
                "filename": filename,
                "extension": file_ext,
                "content_type": file_info["content_type"],
                "category": file_info["category"],
                "relative_path": file_info["relative_path"],
                "upload_size": file_info["file_size"],
                "size_on_disk": file_info["size_on_disk"],
                "saved_path": file_info["saved_path"],
                "saved_at": file_info["saved_at"],
                "verified": file_info["verified"]
            }
            
            # Add data loading info if it's a data file
            if file_info["category"] == "data":
                file_summary["data_loaded"] = file_info.get("data_loaded", False)
                if file_info.get("data_loaded"):
                    file_summary["data_shape"] = file_info.get("data_shape")
                    file_summary["column_count"] = file_info.get("column_count")
                else:
                    file_summary["load_error"] = file_info.get("load_error", "Unknown error")
            
            # Categorize by extension (now matches the directory structure)
            if file_ext in data_extensions:
                files_by_type["data"].append(file_summary)
            elif file_ext in image_extensions:
                files_by_type["images"].append(file_summary)
            elif file_ext in text_extensions:
                files_by_type["text"].append(file_summary)
            else:
                files_by_type["other"].append(file_summary)
        
        # Create enhanced summary with data loading info
        total_size_on_disk = sum(f["size_on_disk"] for f in successful_files) + questions_stats.st_size
        all_verified = all(f["verified"] for f in successful_files) and questions_metadata["verified"]
        created_subdirs = set(f["category"] for f in successful_files)
        
        summary = {
            "data": len(files_by_type["data"]),
            "images": len(files_by_type["images"]),
            "text": len(files_by_type["text"]),
            "other": len(files_by_type["other"]),
            "total_attachments": len(other_files),
            "successful_files": len(successful_files),
            "failed_files": len(failed_files),
            "data_files_loaded": len(loaded_data),
            "data_files_failed": len(data_files) - len(loaded_data),
            "total_upload_size_bytes": sum(f["file_size"] for f in other_files) + len(questions_content),
            "total_size_on_disk_bytes": total_size_on_disk,
            "analysis_directory": analysis_dir,
            "subdirectories_created": sorted(list(created_subdirs)),
            "all_files_verified": all_verified,
            "save_completed_at": datetime.datetime.now().isoformat(),
            "partial_success": len(failed_files) > 0 and len(successful_files) > 0
        }
        
        # Include loaded data summary in response
        result = {
            "success": True,
            "analysis_id": analysis_id,
            "files": files_by_type,
            "summary": summary,
            "loaded_data": {
                "count": len(loaded_data),
                "files": [
                    {
                        "filename": data["filename"],
                        "shape": data["shape"],
                        "columns": data["columns"],
                        "memory_usage_mb": data["memory_usage_mb"]
                    } for data in loaded_data
                ],
                "total_memory_mb": round(sum(data["memory_usage_mb"] for data in loaded_data), 3)
            },
            "questions": {
                "length": len(questions_text),
                "preview": questions_text[:200] + "..." if len(questions_text) > 200 else questions_text,
                **questions_metadata
            }
        }
        
        if failed_files:
            result["failed_files"] = failed_files
            result["warning"] = f"{len(failed_files)} files failed to save but analysis can continue"
        
        print(f"✅ Analysis ID: {analysis_id}")
        print(f"📝 Questions length: {len(questions_text)} chars")
        print(f"📁 Enhanced file summary: {summary}")
        
        # Parse questions with file context for better understanding
        print(f"🧠 Parsing questions with file context...")
        file_context = {
            "uploaded_files": [
                {
                    "filename": data["filename"],
                    "columns": data["columns"],
                    "shape": data["shape"],
                    "type": "data"
                } for data in loaded_data
            ],
            "total_data_files": len(loaded_data),
            "has_data": len(loaded_data) > 0
        }
        
        parsed_data = await _parse_with_llm_context(questions_text, file_context)
        data_sources = _extract_data_sources(parsed_data)
        parsed_questions = _extract_questions(parsed_data)
        
        print(f"✅ Context-aware parsing complete:")
        print(f"   External sources: {len(data_sources)}")
        print(f"   Questions: {len(parsed_questions)}")
        print(f"   Use uploaded data: {parsed_data.get('use_uploaded_data', False)}")
        
        # Step 3: Execute integrated analysis pipeline
        analysis_results = []
        all_data_sources = []
        
        if parsed_questions:
            print(f"🔬 STEP 3: Executing integrated analysis pipeline...")
            
            # Add uploaded DataFrames to analysis sources
            if parsed_data.get("use_uploaded_data", False) and loaded_data:
                print(f"📊 Using {len(loaded_data)} uploaded DataFrames for analysis")
                for data in loaded_data:
                    all_data_sources.append({
                        "success": True,
                        "source_file": data["filename"],
                        "dataframe": data["dataframe"],
                        "shape": data["shape"],
                        "columns": data["columns"],
                        "source_type": "uploaded"
                    })
            
            # Add external scraped data if needed
            if parsed_data.get("external_data_needed", False) and data_sources:
                print(f"📡 Scraping {len(data_sources)} external sources...")
                try:
                    scraped_data = scrape_data_sources(data_sources)
                    external_success = [d for d in scraped_data if d.get("success")]
                    all_data_sources.extend(external_success)
                    print(f"✅ Added {len(external_success)} external data sources")
                except Exception as e:
                    print(f"⚠️ External scraping failed: {e}")
            
            # Execute question answering if we have data and questions
            if all_data_sources and QUESTION_ANSWERER_AVAILABLE:
                print(f"🤔 Answering {len(parsed_questions)} questions with {len(all_data_sources)} data sources...")
                try:
                    analysis_results = answer_questions_with_llm(all_data_sources, parsed_questions)
                    print(f"✅ Generated {len(analysis_results)} analysis results")
                except Exception as e:
                    print(f"❌ Question answering failed: {e}")
                    analysis_results = [{"error": f"Analysis failed: {str(e)}", "success": False}]
            else:
                if not all_data_sources:
                    analysis_results = [{"error": "No data sources available for analysis", "success": False}]
                elif not QUESTION_ANSWERER_AVAILABLE:
                    analysis_results = [{"error": "Question answerer not available", "success": False}]
                else:
                    analysis_results = [{"error": "No questions to answer", "success": False}]
        
        # Extract raw answer values for simple array response format
        raw_answers = []
        for result in analysis_results:
            if result.get("success", False):
                # Extract the actual answer value (could be number, string, base64 image, etc.)
                answer = result.get("answer", None)
                
                # Convert numpy types to native Python types for JSON serialization
                answer = _convert_numpy_types(answer)
                
                raw_answers.append(answer)
            else:
                # For errors, include the error message
                raw_answers.append(f"Error: {result.get('error', 'Unknown error')}")
        
        print(f"✅ Analysis ID: {analysis_id}")
        print(f"📝 Questions length: {len(questions_text)} chars")
        print(f"📁 Enhanced file summary: {summary}")
        print(f"🔬 Analysis results: {len(raw_answers)} raw answers generated")
        print(f"📤 Returning simple array format: {raw_answers}")
        
        # Try to detect and build JSON object response generically
        json_response = _detect_and_build_json_response(questions_text, raw_answers)
        
        # Before returning the final response
        print(f"\n📤 FINAL RESPONSE:")
        print(f"   Format: {'JSON Object' if json_response is not None else 'Array'}")
        if json_response is not None:
            print(f"   Keys: {list(json_response.keys())}")
            print(f"   Response: {json_response}")
        else:
            print(f"   Response: {raw_answers}")
        
        print("="*80 + "\n")
        
        # Return the response...
        if json_response is not None:
            print(f"📤 Returning JSON object format with keys: {list(json_response.keys())}")
            return json_response
        else:
            print(f"📤 Returning simple array format: {raw_answers}")
            return raw_answers
        
    except Exception as e:
        print(f"❌ Phase 2 failed: {e}")
        # Cleanup on total failure
        if 'analysis_dir' in locals():
            _cleanup_analysis_directory(analysis_dir)
        return {
            "success": False,
            "error": f"Unexpected error: {str(e)}"
        }


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
    if isinstance(sources, str):
        sources = [sources]
    
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


def _cleanup_analysis_directory(analysis_dir: str) -> bool:
    """Clean up analysis directory and all its contents"""
    try:
        import shutil
        if os.path.exists(analysis_dir):
            shutil.rmtree(analysis_dir)
            print(f"🧹 Cleaned up analysis directory: {analysis_dir}")
            return True
    except Exception as e:
        print(f"⚠️ Failed to cleanup directory {analysis_dir}: {e}")
        return False
    return False

def _load_dataframe_from_file(file_path: str, file_ext: str) -> pd.DataFrame:
    """Load DataFrame from file based on extension"""
    try:
        if file_ext == '.csv':
            return pd.read_csv(file_path)
        elif file_ext == '.tsv':
            return pd.read_csv(file_path, sep='\t')
        elif file_ext == '.json':
            return pd.read_json(file_path)
        elif file_ext in ['.xlsx', '.xls']:
            return pd.read_excel(file_path)
        elif file_ext == '.parquet':
            return pd.read_parquet(file_path)
        else:
            print(f"⚠️ Unsupported data file type: {file_ext}")
            return None
    except Exception as e:
        print(f"❌ Error loading {file_path}: {e}")
        return None

async def _parse_with_llm_context(question_text: str, file_context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Enhanced LLM parsing that considers uploaded file context
    """
    
    # Build context description
    if file_context["has_data"]:
        context_desc = f"""
AVAILABLE DATA FILES:
{chr(10).join([f"- {f['filename']}: {f['shape'][0]} rows × {f['shape'][1]} columns, columns: {f['columns']}" for f in file_context['uploaded_files']])}
"""
    else:
        context_desc = "No data files uploaded."
    
    prompt = f"""
Parse this data analysis request considering the available uploaded files:

QUESTION:
{question_text}

{context_desc}

Return JSON with:
1. data_sources: Array of external URLs/sources mentioned (NOT the uploaded files)
2. questions: Array of individual questions/tasks to answer
3. use_uploaded_data: Boolean - should use the uploaded files for analysis
4. external_data_needed: Boolean - are external data sources also needed
5. format_requirements: Any output format requirements

The uploaded files are already available for analysis, so don't include them in data_sources.
Only include external URLs or web sources that need to be scraped in data_sources.

Example:
{{
    "data_sources": ["https://example.com/external-data.csv"],
    "questions": ["How many records are in the uploaded data?", "What is the average value?"],
    "use_uploaded_data": true,
    "external_data_needed": false,
    "format_requirements": "JSON array response"
}}

Respond only with valid JSON.
"""
    
    # Try OpenAI
    if llm_client.openai_client:
        try:
            print("🤖 Using OpenAI with file context...")
            response = llm_client.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            result = json.loads(response.choices[0].message.content)
            result["_parser"] = "openai_context"
            return result
        except Exception as e:
            print(f"❌ OpenAI context parsing failed: {e}")
    
    # Try Gemini fallback
    if llm_client.gemini_client:
        try:
            print("🤖 Using Gemini with file context...")
            response = llm_client.gemini_client.generate_content(prompt)
            result = json.loads(response.text)
            result["_parser"] = "gemini_context"
            return result
        except Exception as e:
            print(f"❌ Gemini context parsing failed: {e}")
    
    # Smart fallback
    print("🤖 Using smart fallback with file context...")
    return _parse_with_context_fallback(question_text, file_context)


def _parse_with_context_fallback(question_text: str, file_context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Smart fallback parsing that considers file context
    """
    # Extract URLs (external sources only)
    url_pattern = r'https?://[^\s\n]+'
    urls = re.findall(url_pattern, question_text)
    
    # Extract numbered questions
    question_pattern = r'\d+\.\s*([^?\n]+\??)'
    questions = re.findall(question_pattern, question_text)
    
    # If no numbered questions, use the whole text
    if not questions:
        questions = [question_text.strip()]
    
    # Smart detection based on content
    has_uploaded_data = file_context["has_data"]
    
    # Check if question refers to uploaded data
    upload_keywords = ["uploaded", "csv", "file", "data", "dataset", "analyze"]
    uses_uploaded = any(keyword in question_text.lower() for keyword in upload_keywords)
    
    return {
        "data_sources": urls,
        "questions": questions,
        "use_uploaded_data": has_uploaded_data and (uses_uploaded or not urls),
        "external_data_needed": len(urls) > 0,
        "format_requirements": "standard",
        "_parser": "context_fallback"
    }

def _convert_numpy_types(obj):
    """Convert numpy types to native Python types for JSON serialization"""
    if isinstance(obj, np.integer):
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
