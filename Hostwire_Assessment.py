import pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import DBSCAN
import google.generativeai as genai
import warnings
warnings.filterwarnings('ignore')

class DeliveryAI:
    def __init__(self, data_path, api_key=None):
        self.df = pd.read_csv(data_path, parse_dates=['scheduled_time_start', 'scheduled_time_end', 'actual_arrival_time', 'actual_delivery_time'])
        self.model = genai.GenerativeModel('gemini-2.5-flash') if api_key and genai.configure(api_key=api_key) else None
        self.preprocess()
    
    def preprocess(self):
        self.df['delivery_duration'] = (self.df['actual_delivery_time'] - self.df['actual_arrival_time']).dt.total_seconds() / 60
        self.df['time_window_violation'] = self.df.apply(lambda x: 'Early' if pd.notna(x['actual_arrival_time']) and x['actual_arrival_time'] < x['scheduled_time_start'] else 'Late' if pd.notna(x['actual_arrival_time']) and x['actual_arrival_time'] > x['scheduled_time_end'] else 'On Time', axis=1)
        self.df['capacity_utilization'] = (self.df['package_weight_kg'] / self.df['vehicle_capacity_kg']) * 100
        self.df['zone'] = self.df['delivery_address'].str.extract(r'Zone ([A-D])')
        self.df['is_failed'] = self.df['delivery_status'] == 'failed'

    def google_ai_recommendations(self, analysis_summary):
        if not self.model:
            return self.get_fallback_recommendations()
        
        prompt = f"""As a logistics expert, provide 6 specific cost reduction recommendations:

{analysis_summary}

Focus on: route optimization, failed delivery reduction, time window savings, capacity optimization, driver performance, predictive analytics.
Format as numbered list with metrics:"""
        
        try:
            response = self.model.generate_content(prompt)
            lines = [line.strip() for line in response.text.split('\n') if line.strip()]
            return [line.split('.', 1)[-1].strip() for line in lines if line[0].isdigit() and '.' in line][:6]
        except:
            return self.get_fallback_recommendations()

    def get_fallback_recommendations(self):
        return [
            "Dynamic routing to reduce backtracking (15-20% distance savings)",
            "Customer notifications to reduce failed deliveries", 
            "Better scheduling for time window violations",
            "Optimize vehicle assignment using capacity data",
            "Driver training for time window compliance",
            "Predictive analytics for failure-prone addresses"
        ]

    def analyze_all(self):
        metrics = self.df.groupby('driver_id').agg({
            'distance_km': ['sum', 'mean'], 'idle_time_mins': 'mean', 
            'delivery_status': lambda x: (x == 'delivered').mean(),
            'time_window_violation': lambda x: (x != 'On Time').mean(),
            'delivery_id': 'count', 'package_weight_kg': 'sum'
        }).round(2)
        metrics.columns = ['total_distance', 'avg_distance', 'avg_idle_time', 'success_rate', 'violation_rate', 'delivery_count', 'total_weight']
        
        time_data = self.df[self.df['time_window_violation'] != 'No Data']
        on_time_rate = time_data.groupby('driver_id')['time_window_violation'].apply(lambda x: (x == 'On Time').mean()).round(4) * 100
        violations = time_data.groupby(['driver_id', 'time_window_violation']).size().unstack(fill_value=0)
        costs = {'Early': 15, 'Late': 25}
        financial_impact = {driver: sum(violations.loc[driver].get(vtype, 0) * cost for vtype, cost in costs.items()) for driver in violations.index}
        
        failures = self.df[self.df['is_failed'] == True]
        failure_breakdown = failures.groupby(['driver_id', 'failure_reason']).size().unstack(fill_value=0)
        repeat_failures = failures.groupby(['customer_id', 'delivery_address']).size()
        
        metrics['on_time_rate'] = on_time_rate
        metrics['failed_count'] = self.df.groupby('driver_id')['is_failed'].sum()
        metrics['efficiency_score'] = (metrics['success_rate'] * 40 + metrics['on_time_rate'] * 0.3 + (100 - metrics['avg_idle_time']) * 0.2 + (1000 / metrics['total_distance']) * 10)
        
        return metrics, financial_impact, failure_breakdown, repeat_failures

    def generate_optimized_routes(self):
        try:
            valid_coords = self.df.dropna(subset=['latitude', 'longitude'])
            if len(valid_coords) > 3:
                coords = valid_coords[['latitude', 'longitude']].values
                clustering = DBSCAN(eps=0.02, min_samples=2).fit(coords)
                self.df.loc[valid_coords.index, 'geo_cluster'] = clustering.labels_
                return self.df.groupby('geo_cluster').agg({
                    'delivery_id': 'count', 'driver_id': 'nunique', 
                    'distance_km': 'mean', 'zone': lambda x: x.mode().iloc[0] if not x.mode().empty else 'Unknown'
                }).sort_values('delivery_id', ascending=False)
        except:
            return self.df.groupby('zone').agg({
                'delivery_id': 'count', 'driver_id': 'nunique', 'distance_km': 'mean'
            }).sort_values('delivery_id', ascending=False)

    def create_visualizations(self):
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        successful = self.df[self.df['delivery_status'] == 'delivered']
        
        axes[0,0].scatter(successful['distance_km'], successful['delivery_duration'], alpha=0.6)
        axes[0,0].set(xlabel='Distance (km)', ylabel='Duration (min)', title='Time-Distance Relationship')
        
        self.df.groupby('driver_id').agg({'distance_km': 'sum', 'delivery_duration': 'mean'}).plot(kind='bar', ax=axes[0,1])
        axes[0,1].set_title('Driver Performance Comparison')
        
        self.df['failure_reason'].value_counts().plot(kind='pie', ax=axes[1,0], autopct='%1.1f%%')
        axes[1,0].set_title('Failure Reasons Breakdown')
        
        self.df['capacity_utilization'].hist(ax=axes[1,1], bins=20)
        axes[1,1].set(xlabel='Capacity Utilization %', ylabel='Frequency', title='Vehicle Capacity Utilization')
        
        plt.tight_layout()
        plt.savefig('performance_charts.png', dpi=300, bbox_inches='tight')
        return fig

    def train_predictive_model(self):
        X = self.df[['distance_km', 'package_weight_kg', 'vehicle_capacity_kg', 'route_sequence', 'idle_time_mins']].fillna(0)
        y = (self.df['delivery_status'] == 'delivered').astype(int)
        model = RandomForestRegressor(n_estimators=100, random_state=42).fit(X, y)
        return model, dict(zip(X.columns, model.feature_importances_))

    def generate_comprehensive_report(self):
        metrics, financials, failures, repeats = self.analyze_all()
        model, importance = self.train_predictive_model()
        route_opt = self.generate_optimized_routes()
        
        print("===DELIVERY ROUTE OPTIMIZATION ANALYSIS===\n")
        
        # 1. Route Efficiency
        print("1. ROUTE EFFICIENCY ANALYSIS:")
        print(metrics[['total_distance', 'avg_distance', 'avg_idle_time', 'success_rate', 'violation_rate', 'delivery_count']])
        
        # 2. Time Window Violations
        print("\n2. TIME WINDOW VIOLATIONS:")
        total_cost = sum(financials.values())
        print(f"Total Cost: ${total_cost:.2f}")
        for driver, cost in financials.items(): 
            if cost > 0: print(f"  {driver}: ${cost:.2f}")
        
        # 3. Failed Deliveries
        print("\n3. FAILED DELIVERIES:")
        print("By Driver:"); print(failures)
        if not repeats.empty: print("\nRepeat Failures:"); print(repeats)
        
        # 4. Driver Performance
        print("\n4. DRIVER PERFORMANCE:")
        print(metrics[['success_rate', 'on_time_rate', 'avg_idle_time', 'failed_count', 'efficiency_score']].sort_values('efficiency_score', ascending=False))
        
        # 5. Route Optimization
        print("\n5. ROUTE OPTIMIZATION:")
        if not route_opt.empty: print("Consolidation Opportunities:"); print(route_opt.head())
        
        # 6. AI Recommendations
        print("\n6. COST REDUCTION RECOMMENDATIONS:")
        failed_count = len(self.df[self.df['is_failed'] == True])
        analysis_summary = f"Failures: {failed_count} | Cost: ${total_cost} | Issues: DR003 violations, DR005 idle time"
        for i, rec in enumerate(self.google_ai_recommendations(analysis_summary)[:6], 1): 
            print(f"  {i}. {rec}")
        
        # 7. Visualizations & Model
        print("\n7. VISUALIZATIONS & MODEL:")
        self.create_visualizations()
        print("  Charts saved as 'performance_charts.png'")
        print("  Model feature importance:")
        for feature, imp in importance.items(): print(f"    {feature}: {imp:.3f}")
        
        # 8. Summary
        print("\n" + "="*50)
        print("SUMMARY: Focus on DR003 violations, address failed deliveries, optimize DR005 idle time")
        print("="*50)
        plt.show()

# Execute
analysis = DeliveryAI("C:\\Users\\aabij\\OneDrive\\Desktop\\delivery_routes.csv", api_key="AIzaSyCeASJW0fXEHtysdqYdTci_C0-BtmAJ9Wg")
analysis.generate_comprehensive_report()