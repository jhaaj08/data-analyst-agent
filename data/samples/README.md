# Sample Data for Testing

This folder contains sample datasets for testing the Data Analyst Agent.

## Usage

Upload CSV, JSON, or provide Wikipedia URLs through the web interface at:
- **Web Interface**: http://localhost:8000/api/
- **API Endpoint**: http://localhost:8000/api/frontend

## Supported Data Sources

### CSV Files
- Must have headers in the first row
- Numeric columns will be auto-detected and cleaned
- Monetary values (with $, £, €) are automatically converted

### JSON Files
- Arrays of objects
- Single objects
- Nested structures (will be flattened)

### Wikipedia URLs
- Automatically finds the best data table
- Handles complex Wikipedia page structures
- Example: https://en.wikipedia.org/wiki/List_of_highest-grossing_films

## Example Questions

You can ask questions like:
- "How many records have values over X?"
- "What is the correlation between column A and column B?"
- "Which record has the highest/lowest value?"
- "Create a scatterplot showing the relationship between X and Y"

## File Size Limits

- Maximum file size: 100MB
- For large datasets, consider using remote URLs instead
