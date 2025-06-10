import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
import warnings
warnings.filterwarnings('ignore')

class FruitPricePredictor:
    def __init__(self):
        self.models = {}
        self.encoders = {}
        self.scaler = StandardScaler()
        self.feature_columns = []
        
    def create_sample_data(self, n_samples=5000):
        """Create realistic Indian fruit price dataset"""
        np.random.seed(42)
        
        # Indian fruits and their typical price ranges (per kg)
        fruits = {
            'Apple': {'base_price': 120, 'variance': 40},
            'Banana': {'base_price': 40, 'variance': 15},
            'Orange': {'base_price': 60, 'variance': 20},
            'Mango': {'base_price': 80, 'variance': 30},
            'Grapes': {'base_price': 100, 'variance': 35},
            'Papaya': {'base_price': 35, 'variance': 12},
            'Pineapple': {'base_price': 45, 'variance': 15},
            'Pomegranate': {'base_price': 150, 'variance': 50},
            'Guava': {'base_price': 50, 'variance': 18},
            'Watermelon': {'base_price': 25, 'variance': 8}
        }
        
        # Indian cities
        cities = ['Mumbai', 'Delhi', 'Bangalore', 'Chennai', 'Kolkata', 'Hyderabad', 
                 'Pune', 'Ahmedabad', 'Jaipur', 'Lucknow']
        
        # Market types
        markets = ['Wholesale', 'Retail', 'Mandi']
        
        # Quality grades
        qualities = ['Premium', 'Good', 'Average']
        
        # Seasons
        seasons = ['Summer', 'Monsoon', 'Winter', 'Spring']
        
        data = []
        
        for _ in range(n_samples):
            fruit = np.random.choice(list(fruits.keys()))
            city = np.random.choice(cities)
            market = np.random.choice(markets)
            quality = np.random.choice(qualities)
            season = np.random.choice(seasons)
            
            # Base price
            base_price = fruits[fruit]['base_price']
            variance = fruits[fruit]['variance']
            
            # Market type adjustment
            market_multiplier = {'Wholesale': 0.8, 'Retail': 1.2, 'Mandi': 1.0}[market]
            
            # Quality adjustment
            quality_multiplier = {'Premium': 1.3, 'Good': 1.0, 'Average': 0.8}[quality]
            
            # City adjustment (metro cities are more expensive)
            city_multiplier = 1.2 if city in ['Mumbai', 'Delhi', 'Bangalore'] else 1.0
            
            # Season adjustment (varies by fruit)
            seasonal_factor = np.random.uniform(0.9, 1.1)
            if fruit == 'Mango' and season == 'Summer':
                seasonal_factor = 0.8  # Mango is cheaper in summer
            elif fruit == 'Apple' and season == 'Winter':
                seasonal_factor = 0.9  # Apple is cheaper in winter
            
            # Supply and demand factors
            supply = np.random.uniform(0.5, 2.0)  # Supply level
            demand = np.random.uniform(0.7, 1.5)  # Demand level
            supply_demand_factor = demand / supply
            
            # Transportation cost (distance from production centers)
            transport_cost = np.random.uniform(5, 25)
            
            # Calculate final price
            price = (base_price * market_multiplier * quality_multiplier * 
                    city_multiplier * seasonal_factor * supply_demand_factor +
                    transport_cost + np.random.normal(0, variance * 0.2))
            
            # Ensure minimum price
            price = max(price, base_price * 0.5)
            
            data.append({
                'Fruit': fruit,
                'City': city,
                'Market_Type': market,
                'Quality': quality,
                'Season': season,
                'Supply_Level': supply,
                'Demand_Level': demand,
                'Transport_Cost': transport_cost,
                'Price_per_kg': round(price, 2)
            })
        
        return pd.DataFrame(data)
    
    def prepare_features(self, df):
        """Prepare features for training"""
        df_processed = df.copy()
        
        # Encode categorical variables
        categorical_columns = ['Fruit', 'City', 'Market_Type', 'Quality', 'Season']
        
        for col in categorical_columns:
            if col not in self.encoders:
                self.encoders[col] = LabelEncoder()
                df_processed[col] = self.encoders[col].fit_transform(df_processed[col])
            else:
                df_processed[col] = self.encoders[col].transform(df_processed[col])
        
        # Create additional features
        df_processed['Supply_Demand_Ratio'] = df_processed['Demand_Level'] / df_processed['Supply_Level']
        df_processed['Price_Category'] = pd.cut(df_processed['Price_per_kg'], 
                                              bins=[0, 50, 100, 200, np.inf], 
                                              labels=['Low', 'Medium', 'High', 'Premium'])
        
        # Encode the new categorical feature
        if 'Price_Category' not in self.encoders:
            self.encoders['Price_Category'] = LabelEncoder()
            df_processed['Price_Category'] = self.encoders['Price_Category'].fit_transform(df_processed['Price_Category'])
        
        return df_processed
    
    def train_models(self, df):
        """Train multiple models and select the best one"""
        # Prepare data
        df_processed = self.prepare_features(df)
        
        # Features and target
        X = df_processed.drop(['Price_per_kg'], axis=1)
        y = df_processed['Price_per_kg']
        
        self.feature_columns = X.columns.tolist()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train multiple models
        models_to_train = {
            'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
            'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'LinearRegression': LinearRegression()
        }
        
        best_model = None
        best_score = float('inf')
        
        for name, model in models_to_train.items():
            if name == 'LinearRegression':
                model.fit(X_train_scaled, y_train)
                predictions = model.predict(X_test_scaled)
            else:
                model.fit(X_train, y_train)
                predictions = model.predict(X_test)
            
            mae = mean_absolute_error(y_test, predictions)
            mse = mean_squared_error(y_test, predictions)
            r2 = r2_score(y_test, predictions)
            
            print(f"{name} - MAE: {mae:.2f}, MSE: {mse:.2f}, R2: {r2:.3f}")
            
            self.models[name] = {
                'model': model,
                'mae': mae,
                'mse': mse,
                'r2': r2,
                'scaled': name == 'LinearRegression'
            }
            
            if mae < best_score:
                best_score = mae
                best_model = name
        
        print(f"\nBest model: {best_model} with MAE: {best_score:.2f}")
        return best_model
    
    def predict_price(self, fruit, city, market_type, quality, season, 
                     supply_level, demand_level, transport_cost, model_name='RandomForest'):
        """Predict price for given parameters"""
        # Create input dataframe
        input_data = pd.DataFrame({
            'Fruit': [fruit],
            'City': [city],
            'Market_Type': [market_type],
            'Quality': [quality],
            'Season': [season],
            'Supply_Level': [supply_level],
            'Demand_Level': [demand_level],
            'Transport_Cost': [transport_cost],
            'Price_per_kg': [0]  # Placeholder
        })
        
        # Process the input
        processed_input = self.prepare_features(input_data)
        processed_input = processed_input.drop(['Price_per_kg'], axis=1)
        
        # Ensure all columns are present
        for col in self.feature_columns:
            if col not in processed_input.columns:
                processed_input[col] = 0
        
        processed_input = processed_input[self.feature_columns]
        
        # Make prediction
        model_info = self.models[model_name]
        model = model_info['model']
        
        if model_info['scaled']:
            processed_input = self.scaler.transform(processed_input)
        
        prediction = model.predict(processed_input)[0]
        return round(prediction, 2)
    
    def save_model(self, filepath='fruit_price_model.pkl'):
        """Save the trained model"""
        model_data = {
            'models': self.models,
            'encoders': self.encoders,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath='fruit_price_model.pkl'):
        """Load a trained model"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.models = model_data['models']
        self.encoders = model_data['encoders']
        self.scaler = model_data['scaler']
        self.feature_columns = model_data['feature_columns']
        print(f"Model loaded from {filepath}")

def main():
    # Create and train the model
    predictor = FruitPricePredictor()
    
    print("Creating sample dataset...")
    df = predictor.create_sample_data(5000)
    print(f"Dataset created with {len(df)} samples")
    print("\nSample data:")
    print(df.head())
    
    print("\nTraining models...")
    best_model = predictor.train_models(df)
    
    # Save the model
    predictor.save_model()
    
    # Save the dataset for reference
    df.to_csv('indian_fruit_prices.csv', index=False)
    print("Dataset saved to indian_fruit_prices.csv")
    
    # Test prediction
    print("\nTesting prediction...")
    price = predictor.predict_price(
        fruit='Apple',
        city='Mumbai',
        market_type='Retail',
        quality='Good',
        season='Winter',
        supply_level=1.0,
        demand_level=1.2,
        transport_cost=15.0
    )
    print(f"Predicted price for Apple in Mumbai: ₹{price}/kg")

if __name__ == "__main__":
    main()