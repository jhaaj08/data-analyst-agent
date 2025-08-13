"""
Unit tests for parsing functions - corrected and comprehensive
"""
import pytest
import json
from unittest.mock import Mock, patch

from app.api.routes import _parse_question_file, _parse_question_file_regex


class TestQuestionParsing:
    """Test the question file parsing logic"""
    
    @pytest.mark.asyncio
    async def test_llm_parsing_with_openai_success(self):
        """Test successful OpenAI parsing"""
        
        # Patch the actual instance methods
        with patch('app.api.routes.llm_client') as mock_llm:
            mock_llm.openai_client.chat.completions.create.return_value = Mock(
                choices=[Mock(message=Mock(content=json.dumps({
                    "data_sources": ["https://example.com/data.csv"],
                    "questions": ["What are the trends?"],
                    "output_format": {"type": "json_array"},
                    "requires_web_scraping": False
                })))]
            )
            
            question_text = "Analyze data from: https://example.com/data.csv"
            result = await _parse_question_file(question_text)
            
            assert result["data_sources"] == ["https://example.com/data.csv"]
            assert result["questions"] == question_text
            assert result["output_format"]["type"] == "json_array"
    
    @pytest.mark.asyncio
    async def test_llm_parsing_with_gemini_fallback(self):
        """Test Gemini fallback when OpenAI fails"""
        
        with patch('app.api.routes.llm_client') as mock_llm:
            # Mock OpenAI failure
            mock_llm.openai_client.chat.completions.create.side_effect = Exception("OpenAI failed")
            
            # Mock Gemini success
            mock_llm.gemini_client.generate_content.return_value = Mock(
                text=json.dumps({
                    "data_sources": ["https://example.com/backup.csv"],
                    "output_format": {"type": "standard"},
                    "requires_web_scraping": False
                })
            )
            
            question_text = "Simple analysis question"
            result = await _parse_question_file(question_text)
            
            assert result["data_sources"] == ["https://example.com/backup.csv"]
            assert result["output_format"]["type"] == "standard"
            assert result["questions"] == question_text
    
    @pytest.mark.asyncio
    async def test_llm_parsing_all_fail_regex_fallback(self):
        """Test regex fallback when both LLMs fail"""
        
        with patch('app.api.routes.llm_client') as mock_llm:
            # Mock both LLMs failing
            mock_llm.openai_client.chat.completions.create.side_effect = Exception("OpenAI failed")
            mock_llm.gemini_client.generate_content.side_effect = Exception("Gemini failed")
            
            question_text = "Analyze data from: https://en.wikipedia.org/wiki/Test"
            result = await _parse_question_file(question_text)
            
            # Should fall back to regex parsing (note singular "data_source")
            assert result["data_source"] == "https://en.wikipedia.org/wiki/Test"
            assert result["questions"] == question_text
            assert "requires_web_scraping" in result
    
    @pytest.mark.asyncio
    async def test_llm_parsing_no_clients_available(self):
        """Test when no LLM clients are available"""
        
        with patch('app.api.routes.llm_client') as mock_llm:
            # No LLM clients available
            mock_llm.openai_client = None
            mock_llm.gemini_client = None
            
            question_text = "Simple analysis question without URLs"
            result = await _parse_question_file(question_text)
            
            # Should use regex parsing immediately
            assert result["data_source"] is None
            assert result["requires_web_scraping"] is False
            assert result["output_format"]["type"] == "standard"
    
    @pytest.mark.asyncio
    async def test_real_llm_parsing(self):
        """Test with real LLM (only runs if configured)"""
        
        # This test uses the actual LLM configuration
        question_text = """
        Analyze sales data from the following source:
        https://example.com/sales-data.csv
        
        1. What are the top 5 products by revenue?
        2. Show monthly sales trends
        
        Respond with a JSON array.
        """
        
        result = await _parse_question_file(question_text)
        
        # Should have structured data (either from LLM or regex fallback)
        assert "questions" in result
        assert result["questions"] == question_text
        
        # Check if LLM parsing worked (has data_sources) or regex (has data_source)
        has_llm_format = "data_sources" in result
        has_regex_format = "data_source" in result
        
        assert has_llm_format or has_regex_format
        
        print(f"📊 Parsing used: {'LLM' if has_llm_format else 'Regex'}")
        print(f"🔗 Data source found: {result.get('data_sources') or result.get('data_source')}")


    def test_regex_parsing_fallback(self):
        """Test regex-based parsing fallback"""
        
        question_text = """
        Scrape Wikipedia: https://en.wikipedia.org/wiki/Test
        1. How many items are there?
        2. What's the average?
        Respond with JSON array.
        """
        
        result = _parse_question_file_regex(question_text)
        
        assert result["data_source"] == "https://en.wikipedia.org/wiki/Test"
        assert result["requires_web_scraping"] is True
        assert result["output_format"]["type"] == "json_array"
        assert len(result["numbered_questions"]) == 2


    def test_regex_parsing_no_urls(self):
        """Test regex parsing with no URLs"""
        
        question_text = "Simple analysis question without URLs"
        
        result = _parse_question_file_regex(question_text)
        
        assert result["data_source"] is None
        assert result["requires_web_scraping"] is False
        assert result["output_format"]["type"] == "standard" 
        