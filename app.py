import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from fruit_price_model import FruitPricePredictor
import os

# Page configuration
st.set_page_config(
    page_title="🍎 Fruit Price Predictor",
    page_icon="🍎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #2E8B57;
        margin-bottom: 30px;
    }
    .prediction-box {
        background-color: #f0f8f0;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #2E8B57;
        margin: 20px 0;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load the fruit price dataset"""
    if os.path.exists('indian_fruit_prices.csv'):
        return pd.read_csv('indian_fruit_prices.csv')
    else:
        # Create sample data if file doesn't exist
        predictor = FruitPricePredictor()
        df = predictor.create_sample_data(2000)
        df.to_csv('indian_fruit_prices.csv', index=False)
        return df

@st.cache_resource
def load_model():
    """Load the trained model"""
    predictor = FruitPricePredictor()
    
    if os.path.exists('fruit_price_model.pkl'):
        predictor.load_model()
    else:
        # Train model if it doesn't exist
        st.info("Training model for the first time. This may take a moment...")
        df = load_data()
        predictor.train_models(df)
        predictor.save_model()
    
    return predictor

def main():
    # Header
    st.markdown("<h1 class='main-header'>🍎 Fruit Price Predictor for Indian Markets</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #666;'>Predict optimal fruit prices for better business decisions</p>", unsafe_allow_html=True)
    
    # Load data and model
    df = load_data()
    predictor = load_model()
    
    # Sidebar for inputs
    st.sidebar.header("🔧 Price Prediction Settings")
    
    # Input parameters
    fruit = st.sidebar.selectbox(
        "Select Fruit",
        options=['Apple', 'Banana', 'Orange', 'Mango', 'Grapes', 'Papaya', 
                'Pineapple', 'Pomegranate', 'Guava', 'Watermelon'],
        index=0
    )
    
    city = st.sidebar.selectbox(
        "Select City",
        options=['Mumbai', 'Delhi', 'Bangalore', 'Chennai', 'Kolkata', 
                'Hyderabad', 'Pune', 'Ahmedabad', 'Jaipur', 'Lucknow'],
        index=0
    )
    
    market_type = st.sidebar.selectbox(
        "Market Type",
        options=['Wholesale', 'Retail', 'Mandi'],
        index=0
    )
    
    quality = st.sidebar.selectbox(
        "Quality Grade",
        options=['Premium', 'Good', 'Average'],
        index=1
    )
    
    season = st.sidebar.selectbox(
        "Season",
        options=['Summer', 'Monsoon', 'Winter', 'Spring'],
        index=0
    )
    
    supply_level = st.sidebar.slider(
        "Supply Level",
        min_value=0.5,
        max_value=2.0,
        value=1.0,
        step=0.1,
        help="Lower values indicate low supply"
    )
    
    demand_level = st.sidebar.slider(
        "Demand Level",
        min_value=0.7,
        max_value=1.5,
        value=1.0,
        step=0.1,
        help="Higher values indicate high demand"
    )
    
    transport_cost = st.sidebar.slider(
        "Transport Cost (₹)",
        min_value=5.0,
        max_value=25.0,
        value=15.0,
        step=1.0,
        help="Transportation cost per kg"
    )
    
    model_choice = st.sidebar.selectbox(
        "Select Model",
        options=['RandomForest'],
        index=0
    )
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Prediction section
        st.subheader("💰 Price Prediction")
        
        if st.button("Predict Price", type="primary", use_container_width=True):
            try:
                predicted_price = predictor.predict_price(
                    fruit=fruit,
                    city=city,
                    market_type=market_type,
                    quality=quality,
                    season=season,
                    supply_level=supply_level,
                    demand_level=demand_level,
                    transport_cost=transport_cost,
                    model_name=model_choice
                )
                
                st.markdown(f"""
                <div class='prediction-box'>
                    <h2 style='color: #2E8B57; margin: 0;'>Predicted Price: ₹{predicted_price}/kg</h2>
                    <p style='margin: 10px 0 0 0; color: #666;'>
                        For {quality} quality {fruit} in {city} ({market_type} market)
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                # Price recommendations
                st.subheader("💡 Pricing Recommendations")
                
                rec_col1, rec_col2, rec_col3 = st.columns(3)
                
                with rec_col1:
                    competitive_price = predicted_price * 0.95
                    st.metric(
                        "Competitive Price",
                        f"₹{competitive_price:.2f}",
                        delta=f"-₹{predicted_price - competitive_price:.2f}",
                        delta_color="inverse"
                    )
                
                with rec_col2:
                    premium_price = predicted_price * 1.1
                    st.metric(
                        "Premium Price",
                        f"₹{premium_price:.2f}",
                        delta=f"+₹{premium_price - predicted_price:.2f}"
                    )
                
                with rec_col3:
                    bulk_price = predicted_price * 0.85
                    st.metric(
                        "Bulk Discount Price",
                        f"₹{bulk_price:.2f}",
                        delta=f"-₹{predicted_price - bulk_price:.2f}",
                        delta_color="inverse"
                    )
                
                # Market insights
                st.subheader("📊 Market Insights")
                
                # Filter data for selected fruit and city
                filtered_data = df[(df['Fruit'] == fruit) & (df['City'] == city)]
                
                if not filtered_data.empty:
                    avg_price = filtered_data['Price_per_kg'].mean()
                    min_price = filtered_data['Price_per_kg'].min()
                    max_price = filtered_data['Price_per_kg'].max()
                    
                    insight_col1, insight_col2, insight_col3 = st.columns(3)
                    
                    with insight_col1:
                        st.metric("Average Market Price", f"₹{avg_price:.2f}")
                    with insight_col2:
                        st.metric("Minimum Price", f"₹{min_price:.2f}")
                    with insight_col3:
                        st.metric("Maximum Price", f"₹{max_price:.2f}")
                    
                    # Price comparison
                    comparison = "above" if predicted_price > avg_price else "below"
                    diff = abs(predicted_price - avg_price)
                    
                    if diff > avg_price * 0.1:  # More than 10% difference
                        st.warning(f"⚠️ Your predicted price is significantly {comparison} the market average by ₹{diff:.2f}")
                    else:
                        st.success(f"✅ Your predicted price is close to the market average")
                
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")
    
    with col2:
        # Model performance
        st.subheader("🎯 Model Performance")
        
        model_info = predictor.models.get(model_choice, {})
        if model_info:
            st.metric("Mean Absolute Error", f"₹{model_info.get('mae', 0):.2f}")
            st.metric("R² Score", f"{model_info.get('r2', 0):.3f}")
            
            # Performance interpretation
            r2_score = model_info.get('r2', 0)
            if r2_score > 0.8:
                st.success("🌟 Excellent model performance")
            elif r2_score > 0.6:
                st.info("👍 Good model performance")
            else:
                st.warning("⚠️ Model needs improvement")
        
        # Supply-Demand indicator
        st.subheader("📈 Supply-Demand Analysis")
        sd_ratio = demand_level / supply_level
        
        if sd_ratio > 1.2:
            st.error("🔴 High demand, low supply - Prices likely to increase")
        elif sd_ratio < 0.8:
            st.success("🟢 High supply, low demand - Prices likely to decrease")
        else:
            st.info("🟡 Balanced supply and demand")
        
        # Quick tips
        st.subheader("💡 Pricing Tips")
        st.info("""
        **Smart Pricing Strategies:**
        - Monitor supply-demand ratio
        - Adjust prices based on season
        - Consider transport costs
        - Quality affects pricing significantly
        - Metro cities usually have higher prices
        """)
    
    # Data visualization section
    st.markdown("---")
    st.subheader("📊 Market Analysis Dashboard")
    
    tab1, tab2, tab3 = st.tabs(["Price Trends", "City Comparison", "Market Distribution"])
    
    with tab1:
        # Price trends by fruit
        col1, col2 = st.columns(2)
        
        with col1:
            fruit_prices = df.groupby('Fruit')['Price_per_kg'].mean().sort_values(ascending=False)
            fig1 = px.bar(
                x=fruit_prices.index,
                y=fruit_prices.values,
                title="Average Price by Fruit",
                labels={'x': 'Fruit', 'y': 'Price (₹/kg)'},
                color=fruit_prices.values,
                color_continuous_scale='Viridis'
            )
            fig1.update_layout(showlegend=False)
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            season_prices = df.groupby('Season')['Price_per_kg'].mean()
            fig2 = px.pie(
                values=season_prices.values,
                names=season_prices.index,
                title="Price Distribution by Season"
            )
            st.plotly_chart(fig2, use_container_width=True)
    
    with tab2:
        # City comparison
        city_prices = df.groupby('City')['Price_per_kg'].mean().sort_values(ascending=False)
        fig3 = px.bar(
            x=city_prices.values,
            y=city_prices.index,
            orientation='h',
            title="Average Prices by City",
            labels={'x': 'Price (₹/kg)', 'y': 'City'},
            color=city_prices.values,
            color_continuous_scale='RdYlBu_r'
        )
        st.plotly_chart(fig3, use_container_width=True)
    
    with tab3:
        col1, col2 = st.columns(2)
        
        with col1:
            # Market type distribution
            market_dist = df['Market_Type'].value_counts().reset_index()
            market_dist.columns = ['Market_Type', 'Count']

            fig4 = px.pie(
            data_frame=market_dist,
            names="Market_Type",
            values="Count",
            hole=0.4,  # Donut hole
            title="Market Type Distribution"
            )
            st.plotly_chart(fig4, use_container_width=True)

        
        with col2:
            # Quality distribution
            quality_prices = df.groupby('Quality')['Price_per_kg'].mean()
            fig5 = px.bar(
                x=quality_prices.index,
                y=quality_prices.values,
                title="Average Price by Quality",
                labels={'x': 'Quality', 'y': 'Price (₹/kg)'},
                color=['#ff9999', '#66b3ff', '#99ff99']
            )
            st.plotly_chart(fig5, use_container_width=True)
    
    # Data table
    with st.expander("📋 View Raw Data"):
        st.dataframe(df.head(100), use_container_width=True)
        st.download_button(
            label="Download Complete Dataset",
            data=df.to_csv(index=False),
            file_name="indian_fruit_prices.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main()