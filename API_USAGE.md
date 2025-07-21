# 🚀 **Data Analyst Agent API - Usage Guide**

## 📋 **API Specification**

The API exposes a single endpoint that accepts POST requests with file uploads.

**Endpoint:** `/api/`  
**Method:** `POST`  
**Content-Type:** `multipart/form-data`

## 🎯 **Exact Usage**

### **Command Format:**
```bash
curl "https://app.example.com/api/" -F "@question.txt"
```

### **Local Development:**
```bash
curl "http://localhost:8000/api/" -F "@test_question.txt"
```

## 📄 **Question File Format**

The uploaded file should contain:

1. **Analysis instructions** in natural language
2. **Data source URLs** (if needed)
3. **Output format requirements**
4. **Specific questions** to answer

### **Example question.txt:**
```
Scrape the list of highest grossing films from Wikipedia. It is at the URL:
https://en.wikipedia.org/wiki/List_of_highest-grossing_films

Answer the following questions and respond with a JSON array of strings containing the answer.

1. How many $2 bn movies were released before 2020?
2. Which is the earliest film that grossed over $1.5 bn?
3. What's the correlation between the Rank and Peak?
4. Draw a scatterplot of Rank and Peak along with a dotted red regression line through it.
   Return as a base-64 encoded data URI, `"data:image/png;base64,iVBORw0KG..."` under 100,000 bytes.
```

## 📊 **Response Format**

The API automatically detects the requested output format from the question file:

### **Standard Response:**
```json
{
  "success": true,
  "analysis_id": "abc123",
  "results": {...},
  "visualizations": [...],
  "execution_time": 45.2
}
```

### **JSON Array Response** (when requested):
```json
[
  "7",
  "Avatar (2009)",
  "0.85",
  "data:image/png;base64,iVBORw0KG..."
]
```

## 🧪 **Testing**

### **1. Start the Server:**
```bash
python main.py
```

### **2. Create a Question File:**
```bash
echo "Analyze the data and provide summary statistics." > simple_question.txt
```

### **3. Test the API:**
```bash
curl "http://localhost:8000/api/" -F "@simple_question.txt"
```

### **4. Test with Python:**
```python
import requests

with open('question.txt', 'rb') as f:
    response = requests.post('http://localhost:8000/api/', files={'file': f})
    print(response.json())
```

## 🌐 **Supported Data Sources**

- **File uploads:** CSV, JSON, Excel, Parquet
- **URLs:** Direct file links, APIs
- **Web scraping:** Wikipedia, HTML tables
- **No data:** General analysis questions

## 🎨 **Output Formats**

- **JSON:** Standard structured response
- **JSON Array:** Simple array of string answers
- **Base64 Images:** Embedded visualizations
- **HTML:** Interactive dashboards

## ⚡ **Performance**

- **Timeout:** 3 minutes maximum
- **File Size:** 100MB limit
- **Concurrent:** Multiple requests supported
- **Response Time:** Typically 10-60 seconds

## 🔧 **Examples**

### **Data Analysis:**
```bash
echo "Load data from https://example.com/data.csv and find correlations" > analysis.txt
curl "http://localhost:8000/api/" -F "@analysis.txt"
```

### **Web Scraping:**
```bash
echo "Scrape https://example.com/table and analyze trends" > scrape.txt
curl "http://localhost:8000/api/" -F "@scrape.txt"
```

### **Visualization:**
```bash
echo "Create a histogram of the data and return as base64 image" > viz.txt
curl "http://localhost:8000/api/" -F "@viz.txt"
```

---

**🎯 That's it! The API is designed to be simple:**
1. **Write your question** in a text file
2. **Upload it** with curl
3. **Get intelligent analysis** back automatically 