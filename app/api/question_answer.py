"""
GENERIC Question Answerer - Works with ANY DataFrame
"""
import pandas as pd
import numpy as np
import sys
import os
import re

# Add app to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

def answer_questions_with_llm(dataframes, questions):
    """
    GENERIC function: Answer questions using ANY DataFrame type
    """
    print("🤖 GENERIC QUESTION ANSWERER STARTING")
    print("=" * 50)
    
    # Import LLM client
    try:
        from app.core.llm_client import LLMClient
        llm_client = LLMClient()
        print("✅ LLM Client loaded")
    except Exception as e:
        print(f"❌ LLM Client failed: {e}")
        return generic_fallback(dataframes, questions)
    
    # Get DataFrame
    df = get_main_dataframe(dataframes)
    if df is None:
        return [{"error": "No DataFrame", "success": False}]
    
    # Clean DataFrame GENERICALLY
    df = clean_dataframe_generic(df)
    
    # Answer each question
    results = []
    for i, question in enumerate(questions, 1):
        print(f"\n🤔 QUESTION {i}: {question}")
        print("-" * 40)
        
        result = answer_single_question_generic(llm_client, df, question, i)
        results.append(result)
        
        if result.get('success'):
            print(f"✅ ANSWER: {result['answer']}")
        else:
            print(f"❌ ERROR: {result.get('error')}")
    
    print("\n🎯 GENERIC QUESTION ANSWERING COMPLETE")
    return results


def get_main_dataframe(dataframes):
    """Get the main DataFrame (same as before)"""
    print("🔍 Looking for DataFrame...")
    
    for result in dataframes:
        if result.get('success') and 'dataframe' in result:
            df = result['dataframe']
            print(f"✅ Found DataFrame: {df.shape[0]} rows × {df.shape[1]} columns")
            print(f"📊 Columns: {df.columns.tolist()}")
            return df
    
    print("❌ No DataFrame found")
    return None


def clean_dataframe_generic(df):
    """
    GENERIC DataFrame cleaning - works with ANY data type
    """
    print("🧹 Generic DataFrame cleaning...")
    
    clean_df = df.copy()
    numeric_conversions = 0
    
    # Auto-detect and clean numeric columns
    for col in clean_df.columns:
        if clean_df[col].dtype == 'object':  # String columns
            print(f"   🔍 Analyzing column: {col}")
            
            # Try to detect monetary values
            sample_values = clean_df[col].dropna().astype(str).head(10)
            
            # Check if looks like monetary data (contains $, £, €, numbers with commas)
            if any(re.search(r'[\$£€]|^\d{1,3}(,\d{3})*', str(val)) for val in sample_values):
                try:
                    # Clean monetary symbols and convert to numeric
                    clean_col_name = f"{col}_numeric"
                    clean_df[clean_col_name] = pd.to_numeric(
                        clean_df[col].astype(str).str.replace(r'[\$£€,]', '', regex=True),
                        errors='coerce'
                    )
                    valid_count = clean_df[clean_col_name].notna().sum()
                    print(f"   ✅ Created {clean_col_name}: {valid_count}/{len(clean_df)} values converted")
                    numeric_conversions += 1
                except Exception as e:
                    print(f"   ⚠️  Failed to convert {col}: {e}")
            
            # Check if looks like numeric but stored as string
            elif sample_values.str.match(r'^\d+\.?\d*$').any():
                try:
                    clean_col_name = f"{col}_numeric"
                    clean_df[clean_col_name] = pd.to_numeric(clean_df[col], errors='coerce')
                    valid_count = clean_df[clean_col_name].notna().sum()
                    print(f"   ✅ Created {clean_col_name}: {valid_count}/{len(clean_df)} values converted")
                    numeric_conversions += 1
                except Exception:
                    pass
    
    print(f"✅ Generic cleaning complete: {numeric_conversions} numeric columns created")
    print(f"📊 Final columns: {clean_df.columns.tolist()}")
    
    return clean_df


def answer_single_question_generic(llm_client, df, question, question_num):
    """
    GENERIC question answering with visualization support
    """
    
    # Generate code
    print("🤖 Generating GENERIC code with LLM...")
    code = generate_generic_code(llm_client, df, question)
    
    if not code:
        return {"question_number": question_num, "question": question, "success": False, "error": "Code generation failed"}
    
    print(f"✅ Code generated")
    print(f"📝 Code:\n{code}")
    
    # Execute code
    print("⚡ Executing code...")
    try:
        # ENHANCED safe execution environment with plotting
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        import matplotlib.pyplot as plt
        import io
        import base64
        
        safe_globals = {
            'df': df,
            'pd': pd,
            'np': np,
            'plt': plt,           # ✅ Add matplotlib
            'io': io,             # ✅ Add io for base64
            'base64': base64,     # ✅ Add base64
            'len': len,
            'min': min,
            'max': max,
            'str': str,
            'float': float,
            'int': int,
            'sum': sum,
            'round': round,
            'collections': __import__('collections')
        }
        
        # Try to add seaborn if available
        try:
            import seaborn as sns
            safe_globals['sns'] = sns
            print("   ✅ Seaborn available")
        except ImportError:
            print("   ⚠️  Seaborn not available")
        
        safe_locals = {}
        
        print(f"   🔧 Executing generated code:")
        print(f"   {code}")
        
        exec(code, safe_globals, safe_locals)
        answer = safe_locals.get('answer', 'No answer variable')
        
        print(f"   🔧 Executing generated code:")
        print(f"   {code}")
        
        exec(code, safe_globals, safe_locals)
        answer = safe_locals.get('answer', 'No answer variable')
        
        exec(code, safe_globals, safe_locals)
        answer = safe_locals.get('answer', 'No answer variable')
        
        return {
            "question_number": question_num,
            "question": question,
            "answer": answer,
            "code": code,
            "success": True
        }
        
    except Exception as e:
        # Special handling for visualization errors
        if "scatterplot" in question.lower() or "plot" in question.lower():
            print("   ⚠️  Visualization failed, trying simple matplotlib...")
            try:
                # Simple fallback plot with size optimization
                simple_plot_code = f"""
import io
import base64
plt.figure(figsize=(8, 6))
# Use available numeric columns
numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
if len(numeric_cols) >= 2:
    plt.scatter(df[numeric_cols[0]], df[numeric_cols[1]])
    plt.xlabel(numeric_cols[0])
    plt.ylabel(numeric_cols[1])
    plt.title('Scatterplot')
    
    # Save to base64 with size optimization
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=80, bbox_inches='tight', facecolor='white')
    buffer.seek(0)
    img_data = buffer.getvalue()
    
    # Check size and optimize if needed (target under 100KB = 102400 bytes)
    if len(img_data) > 102400:
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=60, bbox_inches='tight', facecolor='white')
        buffer.seek(0)
        img_data = buffer.getvalue()
    
    img_base64 = base64.b64encode(img_data).decode()
    plt.close()
    answer = img_base64  # Return just the base64 string, not the data URI
else:
    answer = "Not enough numeric columns for scatterplot"
"""
                
                exec(simple_plot_code, safe_globals, safe_locals)
                answer = safe_locals.get('answer', 'Fallback plot failed')
                
                return {
                    "question_number": question_num,
                    "question": question,
                    "answer": answer,
                    "code": simple_plot_code,
                    "success": True
                }
            except:
                pass
        
        return {
            "question_number": question_num,
            "question": question,
            "error": f"Code execution failed: {str(e)}",
            "generated_code": code,
            "success": False
        }


def generate_generic_code(llm_client, df, question):
    """
    Generate GENERIC Python code with visualization support
    """
    
    # Create generic data summary
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    text_cols = df.select_dtypes(include=['object']).columns.tolist()
    numeric_converted_cols = [col for col in df.columns if col.endswith('_numeric')]
    
    prompt = f"""
Generate Python pandas code to answer this question about a dataset:

QUESTION: {question}

DATASET INFO:
- Variable name: df
- Shape: {df.shape[0]} rows × {df.shape[1]} columns
- Numeric columns: {numeric_cols}
- Text columns: {text_cols}
- Auto-converted numeric columns: {numeric_converted_cols}
- Sample row: {df.head(1).to_dict('records')[0]}

RULES:
1. Use 'df' for the DataFrame variable
2. Store final answer in 'answer' variable
3. For counting/numbers: answer = int(count_value)  # Return as integer
4. For finding names/text: answer = "found_name"  # Return as string  
5. For statistics/calculations: answer = round(float(value), 4)  # Return as float
6. For correlations: answer = round(float(correlation_value), 4)  # Return as float
7. For visualizations: Create plot and save as base64 string in answer
8. For DATE correlations: Choose appropriate date component based on context
   - Consider day-of-month for periodic patterns, month for seasonal trends
   - Use business logic to determine which date aspect is most meaningful
9. Available libraries: pandas (pd), numpy (np), matplotlib.pyplot (plt), seaborn (sns), io, base64, collections
10. IMPORTANT: Use .iloc[0] or .item() to extract single values from pandas Series
11. IMPORTANT: Always convert to correct Python type (int, float, str) before assigning to answer

VISUALIZATION EXAMPLE:
```python
# For plot questions:
import io
import base64
plt.figure(figsize=(8, 6))  # Smaller figure size for smaller file
plt.scatter(df['x_col'], df['y_col'])
# Add regression line if needed
plt.title('Plot Title')
plt.xlabel('X Label')
plt.ylabel('Y Label')

# Save to base64 with size optimization
buffer = io.BytesIO()
plt.savefig(buffer, format='png', dpi=80, bbox_inches='tight', facecolor='white')
buffer.seek(0)
img_data = buffer.getvalue()

# Check size and optimize if needed (target under 100KB = 102400 bytes)
if len(img_data) > 102400:
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=60, bbox_inches='tight', facecolor='white')
    buffer.seek(0)
    img_data = buffer.getvalue()

img_base64 = base64.b64encode(img_data).decode()
plt.close()
answer = img_base64  # Return just the base64 string, not the data URI
```

Generate ONLY Python code (no explanations):
"""
    
    # Try OpenAI
    if llm_client.openai_client:
        try:
            print("   🤖 Trying OpenAI...")
            response = llm_client.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=400
            )
            code = response.choices[0].message.content.strip()
            
            # Clean markdown
            if code.startswith('```'):
                lines = code.split('\n')
                code = '\n'.join(lines[1:-1])
            
            print("   ✅ OpenAI success")
            return code
            
        except Exception as e:
            print(f"   ❌ OpenAI failed: {e}")
    
    # Try Gemini
    if llm_client.gemini_client:
        try:
            print("   🤖 Trying Gemini...")
            response = llm_client.gemini_client.generate_content(prompt)
            code = response.text.strip()
            
            # Clean markdown
            if code.startswith('```'):
                lines = code.split('\n')
                code = '\n'.join(lines[1:-1])
            
            print("   ✅ Gemini success")
            return code
            
        except Exception as e:
            print(f"   ❌ Gemini failed: {e}")
    
    print("   ❌ All LLM attempts failed")
    return None


def generic_fallback(dataframes, questions):
    """
    GENERIC fallback - provides basic analysis for ANY dataset
    """
    print("🔧 Using GENERIC fallback logic...")
    
    df = get_main_dataframe(dataframes)
    if df is None:
        return []
    
    df = clean_dataframe_generic(df)
    
    results = []
    for i, question in enumerate(questions, 1):
        # Very basic generic responses
        if "how many" in question.lower():
            answer = len(df)  # Return number, not descriptive string
        elif "what" in question.lower() and "average" in question.lower():
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                col = numeric_cols[0]
                avg = df[col].mean()
                answer = round(avg, 2)  # Return number, not descriptive string
            else:
                answer = "No numeric columns found for average"
        else:
            answer = f"Generic analysis not implemented for: {question}"
        
        results.append({
            "question_number": i,
            "question": question,
            "answer": answer,
            "success": True
        })
    
    return results


# Test function with GENERIC data
if __name__ == "__main__":
    print("🧪 Testing GENERIC Question Answerer...")
    
    # Test with different data types
    test_df = pd.DataFrame({
        'ID': [1, 2, 3],
        'Name': ['Product A', 'Product B', 'Product C'],
        'Price': ['$100', '$200', '$150'],
        'Sales': [1000, 2000, 1500]
    })
    
    test_dataframes = [{"success": True, "dataframe": test_df}]
    test_questions = ["How many products cost over $150?"]
    
    results = answer_questions_with_llm(test_dataframes, test_questions)
    
    print("\n📊 GENERIC TEST RESULTS:")
    for result in results:
        print(f"Q{result['question_number']}: {result.get('answer', result.get('error'))}")