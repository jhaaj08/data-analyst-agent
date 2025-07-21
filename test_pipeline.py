"""
Test Pipeline: Parse Question + Scrape Data Sources
"""
import json
from scrapping import scrape_data_sources


def test_full_pipeline():
    """
    Test the complete pipeline: parsing → scraping
    """
    
    # Simulated output from /parse endpoint
    parse_result = {
        "success": True,
        "data_sources": [
            "https://en.wikipedia.org/wiki/List_of_highest-grossing_films"
        ],
        "questions": [
            "How many $2 bn movies were released before 2020?",
            "Which is the earliest film that grossed over $1.5 bn?",
            "What's the correlation between the Rank and Peak?",
            "Draw a scatterplot of Rank and Peak along with a dotted red regression line through it."
        ],
        "counts": {
            "data_sources": 1,
            "questions": 4
        }
    }
    
    print("🔄 Testing Full Pipeline")
    print("=" * 50)
    
    # Step 1: Show parsed data
    print("\n📝 STEP 1: Parsed Data")
    print(f"Data Sources: {parse_result['data_sources']}")
    print(f"Questions: {len(parse_result['questions'])} questions")
    for i, q in enumerate(parse_result['questions'], 1):
        print(f"  {i}. {q}")
    
    # Step 2: Scrape the data sources
    print("\n📡 STEP 2: Scraping Data Sources")
    scrape_results = scrape_data_sources(parse_result['data_sources'])
    
    # Step 3: Show results
    print("\n📊 STEP 3: Scraping Results")
    for result in scrape_results:
        if result['success']:
            print(f"✅ {result['source_url']}")
            print(f"   Shape: {result['shape'][0]} rows × {result['shape'][1]} columns")
            print(f"   Columns: {result['columns']}")
            print(f"   Sample data: {result['sample_data'][0] if result['sample_data'] else 'No sample'}")
            print(f"   Time: {result['scrape_time_seconds']}s")
            
            # Show DataFrame info
            if 'dataframe' in result:
                df = result['dataframe']
                print(f"   DataFrame ready for analysis: {type(df)}")
        else:
            print(f"❌ {result['source_url']}: {result.get('error', 'Unknown error')}")
    
    print("\n🎯 PIPELINE COMPLETE!")
    print("Ready for analysis step...")
    
    return scrape_results


if __name__ == "__main__":
    results = test_full_pipeline() 