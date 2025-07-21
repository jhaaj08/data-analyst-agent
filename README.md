# Data Analyst Agent API

A comprehensive Python-based data analyst agent API that uses Large Language Models (LLMs) to automatically source, prepare, analyze, and visualize data.

## 🚀 Features

- **LLM-Powered Analysis**: Uses OpenAI GPT-4 or Anthropic Claude for intelligent data analysis planning
- **Multiple Data Sources**: Supports file uploads (CSV, JSON, Excel, Parquet) and URL-based data fetching
- **Automated Processing**: Intelligent data cleaning, missing value handling, and preprocessing
- **Comprehensive Analysis**: Statistical analysis, correlation analysis, regression, clustering, and time series
- **Rich Visualizations**: Generates histograms, heatmaps, scatter plots, box plots, and interactive dashboards
- **RESTful API**: FastAPI-based with automatic documentation and async support
- **3-Minute Timeout**: Designed to return results within the required timeframe
- **Containerized**: Docker support for easy deployment

## 📋 API Endpoints

### Main Analysis Endpoint
```bash
POST /api/
```

**Parameters:**
- `file` (optional): Data file upload (CSV, JSON, Excel, etc.)
- `question`: Natural language analysis request
- `format` (optional): Response format (default: "json")
- `data_source` (optional): URL to fetch data from

**Example Request:**
```bash
curl -X POST "http://localhost:8000/api/" \
  -F "file=@data.csv" \
  -F "question=Analyze the correlation between sales and marketing spend"
```

### Other Endpoints
- `GET /` - Root endpoint with API information
- `GET /health` - Health check
- `GET /docs` - Interactive API documentation
- `GET /api/visualization/{analysis_id}/{filename}` - Serve visualization files

## 🛠 Installation

### Local Development

1. **Clone and Setup:**
```bash
git clone <repository-url>
cd data-analyst-agent
```

2. **Install Dependencies:**
```bash
pip install -r requirements.txt
```

3. **Environment Configuration:**
```bash
cp .env.example .env
# Edit .env with your API keys and configuration
```

4. **Run the Application:**
```bash
python main.py
```

The API will be available at `http://localhost:8000`

### Docker Deployment

1. **Using Docker Compose (Recommended):**
```bash
docker-compose up -d
```

2. **Using Docker directly:**
```bash
docker build -t data-analyst-agent .
docker run -p 8000:8000 -v $(pwd)/data:/app/data data-analyst-agent
```

## 📊 Usage Examples

### Basic Analysis
```bash
curl -X POST "http://localhost:8000/api/" \
  -F "file=@sales_data.csv" \
  -F "question=Provide summary statistics and identify trends"
```

### Correlation Analysis
```bash
curl -X POST "http://localhost:8000/api/" \
  -F "file=@customer_data.csv" \
  -F "question=Find correlations between customer demographics and purchase behavior"
```

### Time Series Analysis
```bash
curl -X POST "http://localhost:8000/api/" \
  -F "file=@stock_prices.csv" \
  -F "question=Analyze the time series trend and seasonality patterns"
```

### External Data Source
```bash
curl -X POST "http://localhost:8000/api/" \
  -F "data_source=https://api.example.com/data.json" \
  -F "question=Analyze the distribution of values and detect outliers"
```

## 🏗 Project Structure

```
data-analyst-agent/
├── main.py                 # FastAPI application entry point
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── LICENSE                # MIT License
├── Dockerfile             # Container configuration
├── docker-compose.yml     # Multi-container setup
├── .env.example          # Environment configuration template
├── app/
│   ├── __init__.py
│   ├── api/
│   │   ├── __init__.py
│   │   ├── routes.py      # API endpoints
│   │   └── models.py      # Pydantic models
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py      # Application configuration
│   │   └── llm_client.py  # LLM integration
│   ├── services/
│   │   ├── __init__.py
│   │   ├── data_sourcer.py    # Data loading service
│   │   ├── data_processor.py  # Data cleaning service
│   │   ├── analyzer.py        # Analysis service
│   │   └── visualizer.py      # Visualization service
│   └── utils/
│       ├── __init__.py
│       └── file_handler.py    # File operations
├── data/
│   ├── uploads/           # Uploaded files
│   └── outputs/           # Generated visualizations
└── tests/                 # Test files (to be implemented)
```

## 🔧 Configuration

### Environment Variables

Create a `.env` file based on `.env.example`:

```bash
# LLM API Keys (at least one required for advanced features)
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Server Configuration
HOST=0.0.0.0
PORT=8000
DEBUG=False

# Processing Limits
MAX_FILE_SIZE=104857600  # 100MB
ANALYSIS_TIMEOUT=180     # 3 minutes
```

## 📈 Analysis Types

The system supports multiple analysis types:

1. **Descriptive Analysis**: Summary statistics, data quality assessment
2. **Diagnostic Analysis**: Correlation analysis, distribution analysis
3. **Predictive Analysis**: Regression models, feature importance
4. **Prescriptive Analysis**: Clustering, segmentation

## 🎨 Visualization Types

Automatically generates appropriate visualizations:

- **Histograms**: Distribution of numeric variables
- **Correlation Heatmaps**: Relationships between variables
- **Scatter Plots**: Bivariate relationships with trend lines
- **Box Plots**: Outlier detection and quartile analysis
- **Bar Charts**: Categorical data distribution
- **Line Charts**: Time series and trends
- **Interactive Dashboards**: HTML summaries with insights

## 🔒 Security Considerations

- Input validation for file types and sizes
- Rate limiting (implement as needed)
- API key management through environment variables
- CORS configuration for production deployment

## 🚀 Deployment Options

### Cloud Platforms

1. **AWS ECS/Fargate**
2. **Google Cloud Run**
3. **Azure Container Instances**
4. **Heroku**
5. **DigitalOcean App Platform**

### Example Cloud Deployment (AWS ECS)

```bash
# Build and push to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account>.dkr.ecr.us-east-1.amazonaws.com
docker build -t data-analyst-agent .
docker tag data-analyst-agent:latest <account>.dkr.ecr.us-east-1.amazonaws.com/data-analyst-agent:latest
docker push <account>.dkr.ecr.us-east-1.amazonaws.com/data-analyst-agent:latest
```

## 🧪 Testing

```bash
# Test the health endpoint
curl http://localhost:8000/health

# Test with sample data
curl -X POST "http://localhost:8000/api/test"
```

## 📊 Performance

- **Timeout**: 3-minute maximum processing time
- **File Size**: 100MB maximum upload
- **Concurrent**: Async processing for multiple requests
- **Memory**: Optimized pandas operations

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- **Documentation**: Visit `/docs` for interactive API documentation
- **Issues**: Report bugs and feature requests via GitHub issues
- **API Status**: Check `/health` endpoint for service status

## 🚧 Roadmap

- [ ] WebSocket support for real-time analysis updates
- [ ] Additional LLM providers (Gemini, Claude-3)
- [ ] Database integration (PostgreSQL, MongoDB)
- [ ] Advanced ML models (XGBoost, Neural Networks)
- [ ] Caching layer for repeated analyses
- [ ] User authentication and rate limiting
- [ ] Batch processing for large datasets 