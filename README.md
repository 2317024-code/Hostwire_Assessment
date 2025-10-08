# Delivery Route Optimization Analysis

DeliveryAI is an advanced AI-driven logistics analytics and route optimization system that automates operational insights for delivery fleets.
It integrates Machine Learning, Generative AI (Gemini API), and Data Science to analyze routes, detect inefficiencies, and provide actionable cost-reduction recommendations.

## Key Features

 - Data Quality & Preprocessing – Cleans and enriches delivery route data automatically.

 - Route Efficiency Analysis – Evaluates total vs. average distance, idle time, and success rates per driver.

 - Time Window Violation Detection – Identifies early, late, or on-time deliveries with financial impact estimation.

 - Failed Delivery Analysis – Classifies and visualizes delivery failures with repeat occurrence detection.

 - Driver Performance Scorecard – Computes efficiency scores based on time, distance, and success metrics.

 - Route Optimization (DBSCAN Clustering) – Groups deliveries geographically to find consolidation opportunities.

 - AI-Powered Recommendations – Uses Google Generative AI (Gemini 2.5 Flash) to produce data-driven improvement plans.

 - Predictive Modeling – Trains a Random Forest model to identify performance-driving features.

 - Visual Analytics – Automatically generates and saves plots for performance, utilization, and failure reasons.

## Workflow Overview
    
 - Preprocess Delivery Data → Cleans timestamps, adds KPIs (duration, utilization, violations).
    
 - Analyze Performance → Evaluates route efficiency, time windows, and delivery success.
    
 - Cluster Routes → Groups close deliveries using DBSCAN to detect redundant trips.
    
 - Train Predictive Model → Builds a feature-importance model with Random Forest.
    
 - Generate Reports → Prints detailed tables, driver insights, and cost summaries.
    
 - Create Visualizations → Exports comparative charts and performance graphs.
    
 - Get AI Recommendations → Calls Gemini API for actionable optimization suggestions.

## Example requirements.txt:

  pandas
  
  numpy
  
  matplotlib
  
  seaborn
  
  scikit-learn
  
  google-generativeai

## Deliverables

 - Route-by-route efficiency table
 - Time window violation summary with financial impact
 - Failed delivery classification
 - Driver performance scorecard
 - Optimized delivery route clusters
 - Cost-reduction recommendations (AI-generated)
 - Visual charts saved as performance_charts.png

## Visualizations
- Time-Distance Relationship
- Driver Performance Comparison (Bar Chart)
- Failure Reason Breakdown (Pie Chart)
- Vehicle Capacity Utilization (Histogram)

All figures are saved automatically as: performance_charts.png

## AI Recommendations Workflow

 - When connected to Google Gemini API, the model generates 6 actionable recommendations, such as:

 - Implement dynamic routing to reduce backtracking (15–20% savings)

 - Introduce real-time customer notifications to lower failed deliveries

 - Improve time window scheduling accuracy to reduce penalties

 - Reassign vehicles based on capacity utilization metrics

 - Conduct driver training for compliance improvement

 - Apply predictive analytics to identify failure-prone addresses

## Predictive Model Insights
The Random Forest model highlights the top influencing features driving successful deliveries:

    Feature	Importance
    Distance (km)	High
    Idle Time (mins)	Moderate
    Package Weight (kg)	Moderate
    Route Sequence	Low
    Vehicle Capacity	Low

## API Key Setup

    Got an API key from Google AI Studio
    Pass it when creating the class instance:
    DeliveryAI("path_to_csv", api_key="YOUR_API_KEY")

## Technologies Used

 - Python 3.9+
 - Pandas / NumPy / Seaborn / Matplotlib
 - Scikit-Learn (RandomForestRegressor)
 - DBSCAN (Route Clustering)
 - Google Generative AI (Gemini 2.5 Flash)
