"""
Test script for multi-LLM setup
"""
import asyncio
from app.core.config import settings
from app.core.llm_client import LLMClient

async def test_multi_llm():
    """Test multi-LLM configuration"""
    
    print("🔧 Multi-LLM Configuration Test")
    print("=" * 40)
    
    # Check configuration
    print(f"OpenAI Key: {'✅ SET' if settings.openai_api_key else '❌ MISSING'}")
    print(f"Gemini Key: {'✅ SET' if settings.gemini_api_key else '❌ MISSING'}")
    
    # Initialize client
    llm_client = LLMClient()
    
    # Test simple parsing
    test_question = """
    Analyze sales data and answer:
    1. What are the top performing products?
    2. Any seasonal trends?
    """
    
    print("\n🧪 Testing LLM parsing...")
    try:
        result = await llm_client.parse_analysis_request(test_question, {})
        print("✅ Parsing successful!")
        print(f"📊 Analysis type: {result.get('analysis_type', 'unknown')}")
        print(f"📋 Steps: {len(result.get('analysis_steps', []))}")
        
        return True
    except Exception as e:
        print(f"❌ Parsing failed: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_multi_llm())
    
    if success:
        print("\n🎉 Multi-LLM setup is working!")
    else:
        print("\n💥 Setup needs fixing!") 