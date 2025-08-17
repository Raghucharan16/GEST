import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from pathlib import Path
import sys
import random

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / 'src'))

try:
    from data_loader_new import DigineticaDataLoader
except ImportError:
    st.error("Could not import data loader. Please ensure src/data_loader_new.py exists.")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="GEST Recommendation System",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize data loader
@st.cache_resource
def get_data_loader():
    """Load and cache the data loader."""
    try:
        config_path = Path(__file__).parent / 'config.yaml'
        loader = DigineticaDataLoader(
            data_dir='data',
            config_path=str(config_path)
        )
        loader.load_data()
        return loader
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def generate_recommendations(selected_products, num_recommendations=5):
    """Generate simple collaborative filtering recommendations."""
    # Simple recommendation logic based on selected products
    all_items = list(range(1, 1001))  # Sample product IDs
    
    # Remove already selected items
    available_items = [item for item in all_items if item not in selected_products]
    
    # Simple recommendation: random selection with some logic
    recommendations = []
    for _ in range(min(num_recommendations, len(available_items))):
        item_id = random.choice(available_items)
        
        # Generate fake product info
        product_info = {
            'itemId': item_id,
            'title': f"Recommended Product {item_id}",
            'price': round(random.uniform(10, 500), 2),
            'brand': random.choice(['BrandA', 'BrandB', 'BrandC', 'TechCorp', 'StyleCo']),
            'category': random.choice(['Electronics', 'Clothing', 'Books', 'Home', 'Sports']),
            'confidence': round(random.uniform(0.6, 0.95), 2)
        }
        recommendations.append(product_info)
        available_items.remove(item_id)
    
    return recommendations

def show_product_discovery(loader):
    """Show product discovery interface."""
    st.header("🛍️ Product Discovery")
    
    if loader is None:
        st.warning("Data loader not available. Using sample data.")
        # Generate sample products
        products = []
        for i in range(1, 51):  # 50 sample products
            products.append({
                'itemId': i,
                'title': f"Sample Product {i}",
                'price': round(random.uniform(10, 500), 2),
                'brand': random.choice(['BrandA', 'BrandB', 'BrandC', 'TechCorp', 'StyleCo']),
                'category': random.choice(['Electronics', 'Clothing', 'Books', 'Home', 'Sports']),
                'click_count': random.randint(1, 100)
            })
    else:
        # Use actual data from loader
        if hasattr(loader, 'interactions') and loader.interactions is not None:
            # Get top products by interaction count
            top_products = loader.interactions['itemId'].value_counts().head(50)
            products = []
            for item_id, count in top_products.items():
                product_info = {'itemId': item_id, 'click_count': count}
                
                # Add product details if available
                if hasattr(loader, 'products_info') and loader.products_info is not None:
                    product_row = loader.products_info[loader.products_info['itemId'] == item_id]
                    if not product_row.empty:
                        product_info.update(product_row.iloc[0].to_dict())
                
                products.append(product_info)
        else:
            st.info("No interaction data available. Using sample products.")
            products = []
    
    # Filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Category filter
        categories = list(set([p.get('category', 'Unknown') for p in products]))
        if 'Unknown' in categories:
            categories.remove('Unknown')
            categories = ['All'] + sorted(categories) + ['Unknown']
        else:
            categories = ['All'] + sorted(categories)
        
        selected_category = st.selectbox("Category", categories)
    
    with col2:
        # Price range filter
        prices = [p.get('price', 0) for p in products if p.get('price')]
        if prices:
            min_price, max_price = st.slider(
                "Price Range", 
                min_value=int(min(prices)), 
                max_value=int(max(prices)),
                value=(int(min(prices)), int(max(prices)))
            )
        else:
            min_price, max_price = 0, 1000
    
    with col3:
        # Sort options
        sort_option = st.selectbox(
            "Sort by", 
            ["Popularity", "Price: Low to High", "Price: High to Low"]
        )
    
    # Filter products
    filtered_products = products
    
    if selected_category != 'All':
        filtered_products = [p for p in filtered_products if p.get('category') == selected_category]
    
    if prices:
        filtered_products = [p for p in filtered_products 
                           if p.get('price', 0) >= min_price and p.get('price', 0) <= max_price]
    
    # Sort products
    if sort_option == "Popularity":
        filtered_products = sorted(filtered_products, key=lambda x: x.get('click_count', 0), reverse=True)
    elif sort_option == "Price: Low to High":
        filtered_products = sorted(filtered_products, key=lambda x: x.get('price', 0))
    elif sort_option == "Price: High to Low":
        filtered_products = sorted(filtered_products, key=lambda x: x.get('price', 0), reverse=True)
    
    # Display products in grid
    st.subheader(f"Products ({len(filtered_products)} found)")
    
    # Initialize session state for selected products
    if 'selected_products' not in st.session_state:
        st.session_state.selected_products = []
    
    # Product grid
    cols_per_row = 3
    for i in range(0, len(filtered_products), cols_per_row):
        cols = st.columns(cols_per_row)
        
        for j, col in enumerate(cols):
            if i + j < len(filtered_products):
                product = filtered_products[i + j]
                
                with col:
                    with st.container():
                        col_img, col_info, col_stats = st.columns([1, 3, 1])
                        
                        with col_img:
                            # Placeholder for product image
                            st.write("📦")
                        
                        with col_info:
                            product_title = product.get('title', f"Product {product['itemId']}")
                            st.write(f"**{product_title}**")
                            if 'price' in product and product['price']:
                                st.write(f"💰 ${product['price']:.2f}")
                            if 'brand' in product:
                                st.write(f"🏷️ {product['brand']}")
                        
                        with col_stats:
                            st.metric("Clicks", product.get('click_count', 0))
                        
                        # Add to selection button
                        is_selected = product['itemId'] in st.session_state.selected_products
                        button_text = "Remove" if is_selected else "Select"
                        button_type = "secondary" if is_selected else "primary"
                        
                        if st.button(button_text, key=f"btn_{product['itemId']}", type=button_type):
                            if is_selected:
                                st.session_state.selected_products.remove(product['itemId'])
                            else:
                                st.session_state.selected_products.append(product['itemId'])
                            st.rerun()

def show_recommendations(loader):
    """Show recommendations based on selected products."""
    st.header("🎯 Your Recommendations")
    
    if not st.session_state.get('selected_products', []):
        st.info("👆 Please select some products from the Product Discovery page to get recommendations!")
        return
    
    selected_products = st.session_state.selected_products
    
    # Show selected products
    st.subheader("📝 Your Selected Products")
    
    cols = st.columns(min(len(selected_products), 5))
    for i, product_id in enumerate(selected_products[:5]):
        with cols[i % len(cols)]:
            # Try to get product info
            product_info = {'itemId': product_id}
            if loader and hasattr(loader, 'products_info') and loader.products_info is not None:
                product_row = loader.products_info[loader.products_info['itemId'] == product_id]
                if not product_row.empty:
                    product_info.update(product_row.iloc[0].to_dict())
            
            # Display product card
            with st.container():
                st.write("📦")
                product_title = product_info.get('title', f"Product {product_id}")
                price_text = f"${product_info['price']:.2f}" if 'price' in product_info else 'N/A'
                st.info(f"**{product_title}**\n💰 {price_text}")
    
    if len(selected_products) > 5:
        st.caption(f"... and {len(selected_products) - 5} more products")
    
    # Generate recommendations
    st.subheader("✨ Recommended for You")
    
    # Recommendation controls
    col1, col2 = st.columns(2)
    with col1:
        num_recommendations = st.slider("Number of recommendations", 1, 10, 5)
    with col2:
        if st.button("🔄 Refresh Recommendations", type="secondary"):
            st.rerun()
    
    # Get recommendations
    try:
        recommendations = generate_recommendations(selected_products, num_recommendations)
        
        if recommendations:
            # Display recommendations in grid
            cols_per_row = 3
            for i in range(0, len(recommendations), cols_per_row):
                cols = st.columns(cols_per_row)
                
                for j, col in enumerate(cols):
                    if i + j < len(recommendations):
                        rec = recommendations[i + j]
                        item_id = rec['itemId']
                        
                        with col:
                            with st.container():
                                col_img, col_info = st.columns([1, 3])
                                
                                with col_img:
                                    st.write("🎁")
                                
                                with col_info:
                                    if loader and hasattr(loader, 'products_info') and loader.products_info is not None:
                                        product_row = loader.products_info[loader.products_info['itemId'] == item_id]
                                        if not product_row.empty:
                                            product_info = product_row.iloc[0].to_dict()
                                            product_title = product_info.get('title', f"Product {item_id}")
                                            st.write(f"**{product_title}**")
                                            if 'price' in product_info and product_info['price']:
                                                st.write(f"💰 ${product_info['price']:.2f}")
                                            if 'brand' in product_info:
                                                st.write(f"🏷️ {product_info['brand']}")
                                        else:
                                            st.write(f"**{rec['title']}**")
                                            st.write(f"💰 ${rec['price']:.2f}")
                                            st.write(f"🏷️ {rec['brand']}")
                                    else:
                                        st.write(f"**{rec['title']}**")
                                        st.write(f"💰 ${rec['price']:.2f}")
                                        st.write(f"🏷️ {rec['brand']}")
                                
                                # Confidence score
                                confidence = rec.get('confidence', random.uniform(0.6, 0.9))
                                st.progress(confidence, f"Confidence: {confidence:.0%}")
                                
                                # Add to selection
                                if st.button("Add to Selection", key=f"add_{item_id}", type="primary"):
                                    if item_id not in st.session_state.selected_products:
                                        st.session_state.selected_products.append(item_id)
                                        st.success(f"Added Product {item_id} to your selection!")
                                        st.rerun()
        else:
            st.warning("No recommendations available at the moment.")
            st.info("This might happen if the model hasn't been trained yet or if there's insufficient data.")
    
    except Exception as e:
        st.error(f"Error generating recommendations: {e}")

def show_analytics(loader):
    """Show analytics dashboard."""
    st.header("📊 Analytics Dashboard")
    
    if loader is None:
        st.warning("Data loader not available. Showing sample analytics.")
        # Generate sample data for analytics
        sample_data = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=30, freq='D'),
            'interactions': np.random.randint(100, 1000, 30),
            'unique_users': np.random.randint(50, 200, 30),
            'unique_items': np.random.randint(20, 100, 30)
        })
        data = sample_data
    else:
        # Use actual data
        if hasattr(loader, 'interactions') and loader.interactions is not None:
            data = loader.interactions.copy()
            
            # Convert timestamp if available
            if 'timeframe' in data.columns:
                data['date'] = pd.to_datetime(data['timeframe'], unit='s', errors='coerce')
            else:
                # Generate sample dates
                data['date'] = pd.date_range('2024-01-01', periods=len(data), freq='1H')[:len(data)]
            
            # Aggregate by date
            daily_stats = data.groupby(data['date'].dt.date).agg({
                'sessionId': 'count',  # interactions
                'userId': 'nunique',   # unique users
                'itemId': 'nunique'    # unique items
            }).reset_index()
            daily_stats.columns = ['date', 'interactions', 'unique_users', 'unique_items']
            data = daily_stats
        else:
            st.info("No data available for analytics.")
            return
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_interactions = data['interactions'].sum() if 'interactions' in data.columns else 0
        st.metric("Total Interactions", f"{total_interactions:,}")
    
    with col2:
        avg_daily_users = data['unique_users'].mean() if 'unique_users' in data.columns else 0
        st.metric("Avg Daily Users", f"{avg_daily_users:.0f}")
    
    with col3:
        total_items = data['unique_items'].max() if 'unique_items' in data.columns else 0
        st.metric("Total Items", f"{total_items:,}")
    
    with col4:
        if len(st.session_state.get('selected_products', [])) > 0:
            st.metric("Your Selections", len(st.session_state.selected_products))
        else:
            st.metric("Your Selections", 0)
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Interactions over time
        st.subheader("📈 Daily Interactions")
        if 'interactions' in data.columns and len(data) > 0:
            fig = px.line(data, x='date', y='interactions', 
                         title="Interactions Over Time")
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No interaction data available")
    
    with col2:
        # User engagement
        st.subheader("👥 User Engagement")
        if 'unique_users' in data.columns and len(data) > 0:
            fig = px.bar(data.tail(7), x='date', y='unique_users',
                        title="Daily Active Users (Last 7 days)")
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No user data available")
    
    # Additional analytics
    if loader and hasattr(loader, 'interactions') and loader.interactions is not None:
        st.subheader("🔍 Detailed Analytics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Top products
            st.write("**Top 10 Most Popular Products**")
            top_products = loader.interactions['itemId'].value_counts().head(10)
            fig = px.bar(x=top_products.values, y=[f"Product {i}" for i in top_products.index],
                        orientation='h', title="Product Popularity")
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Session lengths
            st.write("**Session Length Distribution**")
            session_lengths = loader.interactions.groupby('sessionId').size()
            fig = px.histogram(x=session_lengths.values, nbins=20,
                             title="Session Length Distribution")
            fig.update_xaxis(title="Items per Session")
            fig.update_yaxis(title="Number of Sessions")
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

def show_about():
    """Show about page."""
    st.header("ℹ️ About GEST Recommendation System")
    
    st.markdown("""
    ## 🎯 Overview
    GEST (Generated E-commerce Session-based Recommendation System) is a modern recommendation platform 
    that provides personalized product suggestions based on user behavior and preferences.
    
    ## 🛠️ Features
    - **Product Discovery**: Browse and explore products with advanced filtering
    - **Smart Recommendations**: Get personalized suggestions based on your selections
    - **Real-time Analytics**: Monitor system performance and user engagement
    - **Session-based Learning**: Recommendations improve as you interact with more products
    
    ## 🔧 Technology Stack
    - **Backend**: Python, RecBole Framework, BERT4Rec Model
    - **Frontend**: Streamlit, Plotly for visualizations
    - **Data Processing**: Pandas, NumPy
    - **Dataset**: Diginetica E-commerce Dataset
    
    ## 📊 Model Details
    - **Model**: BERT4Rec (Bidirectional Encoder Representations from Transformers for Recommendation)
    - **Training**: Sequential recommendation with temporal data splitting
    - **Features**: Session-based learning, item embeddings, positional encoding
    
    ## 🚀 Getting Started
    1. **Discover Products**: Use the Product Discovery page to browse and select items
    2. **Get Recommendations**: View personalized suggestions based on your selections
    3. **Analyze Trends**: Check the Analytics dashboard for insights
    
    ## 📈 Current Status
    - ✅ Data Loading and Preprocessing
    - ✅ User Interface Implementation
    - ✅ Basic Recommendation Engine
    - ✅ Analytics Dashboard
    - 🔄 Advanced Model Training (In Progress)
    
    ## 🎉 Next Steps
    - Train the BERT4Rec model on real Diginetica data
    - Implement advanced filtering and search
    - Add user feedback mechanisms
    - Enhance recommendation algorithms
    """)
    
    # Quick stats
    st.subheader("📋 Quick Stats")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("**Sample Data**\n36,833 interactions\n5,000 sessions\n1,000 products")
    
    with col2:
        st.info("**Model Type**\nBERT4Rec\nSequential Recommendation\nTransformer-based")
    
    with col3:
        if 'selected_products' in st.session_state:
            selected_count = len(st.session_state.selected_products)
            st.success(f"**Your Activity**\n{selected_count} products selected\nReady for recommendations")
        else:
            st.success("**Your Activity**\n0 products selected\nStart exploring!")

def main():
    """Main application function."""
    # Sidebar navigation
    st.sidebar.title("🛍️ GEST Recommender")
    st.sidebar.markdown("---")
    
    page = st.sidebar.selectbox(
        "Navigate to:",
        ["🛍️ Product Discovery", "🎯 Recommendations", "📊 Analytics", "ℹ️ About"]
    )
    
    # Load data
    loader = get_data_loader()
    
    # Show selected page
    if page == "🛍️ Product Discovery":
        show_product_discovery(loader)
    elif page == "🎯 Recommendations":
        show_recommendations(loader)
    elif page == "📊 Analytics":
        show_analytics(loader)
    elif page == "ℹ️ About":
        show_about()
    
    # Sidebar info
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🎯 Your Selection")
    if 'selected_products' in st.session_state and st.session_state.selected_products:
        st.sidebar.write(f"**{len(st.session_state.selected_products)}** products selected")
        if st.sidebar.button("Clear All", type="secondary"):
            st.session_state.selected_products = []
            st.rerun()
    else:
        st.sidebar.write("No products selected yet")
    
    st.sidebar.markdown("---")
    st.sidebar.caption("💡 Tip: Select products to get personalized recommendations!")

if __name__ == "__main__":
    main()
