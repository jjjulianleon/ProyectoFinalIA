# CareerData MCP Server Guide

## Overview

The CareerData MCP (Model Context Protocol) server provides a structured interface for accessing career prediction data and ML model functionality. It demonstrates the MCP pattern for AI-powered applications.

## What is MCP?

The Model Context Protocol (MCP) is a standardized way to expose AI/ML model capabilities and data operations through a well-defined interface. It allows:
- Consistent API for model interactions
- Easy integration with different applications
- Separation of concerns between data, models, and business logic

## Available Methods

### 1. `list_careers`

Lists all available career categories in the dataset.

**Parameters:** None

**Example:**
```python
from src.mcp.career_data_server import CareerDataMCPServer

server = CareerDataMCPServer()
result = server.handle_request("list_careers", {})
print(result)
```

**Response:**
```json
{
  "success": true,
  "result": {
    "total_careers": 10,
    "careers": ["Software Engineer", "Teacher", "..."],
    "distribution": {
      "Teacher": 211,
      "Mechanical Engineer": 164,
      ...
    }
  }
}
```

---

### 2. `get_career_info`

Get detailed statistics about a specific career.

**Parameters:**
- `career_name` (string): Name of the career

**Example:**
```python
result = server.handle_request("get_career_info", {
    "career_name": "Software Engineer"
})
```

**Response:**
```json
{
  "success": true,
  "result": {
    "career": "Software Engineer",
    "found": true,
    "total_samples": 62,
    "statistics": {
      "Numerical_Aptitude": {
        "mean": 73.78,
        "median": 74.50,
        "min": 45.20,
        "max": 95.30
      },
      ...
    }
  }
}
```

---

### 3. `get_dataset_stats`

Get overall dataset statistics.

**Parameters:** None

**Example:**
```python
result = server.handle_request("get_dataset_stats", {})
```

**Response:**
```json
{
  "success": true,
  "result": {
    "total_records": 1000,
    "total_features": 10,
    "careers": 10,
    "feature_statistics": {
      "Openness": {
        "mean": 6.02,
        "std": 1.90,
        "min": 1.00,
        "max": 10.00
      },
      ...
    }
  }
}
```

---

### 4. `predict_career`

Predict career path based on input features.

**Parameters:**
- `features` (dict): Dictionary of feature values

**Example:**
```python
result = server.handle_request("predict_career", {
    "features": {
        'Openness': 8.5,
        'Conscientiousness': 7.0,
        'Extraversion': 6.0,
        'Agreeableness': 7.5,
        'Neuroticism': 4.0,
        'Numerical_Aptitude': 85.0,
        'Spatial_Aptitude': 75.0,
        'Perceptual_Aptitude': 80.0,
        'Abstract_Reasoning': 82.0,
        'Verbal_Reasoning': 70.0
    }
})
```

**Response:**
```json
{
  "success": true,
  "result": {
    "predicted_career": "Software Engineer",
    "confidence": 0.75,
    "top_predictions": [
      {"career": "Software Engineer", "probability": 0.75},
      {"career": "Data Scientist", "probability": 0.65},
      {"career": "Business Analyst", "probability": 0.45}
    ],
    "model": "RandomForestClassifier"
  }
}
```

---

### 5. `search_similar_profiles`

Find sample profiles for a specific career.

**Parameters:**
- `target_career` (string): Career to search for
- `top_n` (int): Number of samples to return (default: 5)

**Example:**
```python
result = server.handle_request("search_similar_profiles", {
    "target_career": "Teacher",
    "top_n": 3
})
```

**Response:**
```json
{
  "success": true,
  "result": {
    "career": "Teacher",
    "samples_count": 3,
    "profiles": [
      {
        "Openness": 6.5,
        "Conscientiousness": 8.0,
        ...
      },
      ...
    ]
  }
}
```

---

## Integration Examples

### Python Application

```python
from src.mcp.career_data_server import CareerDataMCPServer

# Initialize server
server = CareerDataMCPServer()

# Get career information
career_info = server.handle_request("get_career_info", {
    "career_name": "Data Scientist"
})

if career_info['success']:
    print(f"Found {career_info['result']['total_samples']} samples")
    print(f"Average aptitude: {career_info['result']['statistics']}")
```

### REST API Wrapper (Future Enhancement)

```python
from flask import Flask, request, jsonify
from src.mcp.career_data_server import CareerDataMCPServer

app = Flask(__name__)
server = CareerDataMCPServer()

@app.route('/mcp/<method>', methods=['POST'])
def mcp_endpoint(method):
    params = request.json
    result = server.handle_request(method, params)
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
```

## Error Handling

All methods return a standardized response format:

**Success:**
```json
{
  "success": true,
  "result": { ... }
}
```

**Error:**
```json
{
  "success": false,
  "error": "Error message here"
}
```

## Architecture Benefits

1. **Modularity**: MCP server can be used independently or integrated into larger systems
2. **Testability**: Clear interface makes unit testing straightforward
3. **Scalability**: Can be exposed as REST API, gRPC service, or other protocols
4. **Consistency**: Standardized request/response format
5. **Documentation**: Self-documenting through method signatures

## Future Enhancements

- Add caching for frequently requested data
- Implement batch prediction endpoints
- Add authentication and rate limiting
- Create REST API wrapper
- Add WebSocket support for real-time predictions
- Implement model versioning and A/B testing

## Testing

Run the test suite:
```bash
python src/mcp/career_data_server.py
```

This will execute all test cases and verify MCP functionality.
