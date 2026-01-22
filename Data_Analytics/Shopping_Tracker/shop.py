import streamlit as st
import pandas as pd
from datetime import datetime
import os

# Set page config
st.set_page_config(
    page_title="Shopping Tracker",
    page_icon="�️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .main-title {
        text-align: center;
        color: #11315c;
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        color: #4ECDC4;
    }
    
    [data-baseweb="tab"] {
        font-size: 1.3rem;
        font-weight: 600;
        padding: 0.75rem 1.5rem;
    }
    
    [data-baseweb="tab"]:hover {
        color: #4ECDC4;
    }
    
    .stAlert {
        background-color: #11315c !important;
        color: white !important;
    }
    
    .stAlert p, .stAlert div, .stAlert span {
        color: white !important;
    }
    
    [data-testid="stSidebar"] {
        background: linear-gradient(#09203f 0%, #537895 100%);
    }
    
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p {
        color: white !important;
    }
    
    [data-testid="stSidebar"] input,
    [data-testid="stSidebar"] select,
    [data-testid="stSidebar"] textarea {
        color: #333 !important;
        background-color: rgba(255, 255, 255, 0.95) !important;
        border: 2px solid rgba(255, 255, 255, 0.3) !important;
    }
    
    [data-testid="stSidebar"] input::placeholder {
        color: #999 !important;
    }
    
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3rem;
        font-weight: 600;
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        transform: scale(1.02);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    </style>
""", unsafe_allow_html=True)

# Data file path
DATA_FILE = "shopping_data.csv"

# Categories
CATEGORIES = {
    "Food": "#FF6B6B",
    "Drinks": "#4ECDC4", 
    "Household": "#45B7D1",
    "Investments": "#96CEB4",
    "Sport": "#FF9FF3",
    "Culture": "#54A0FF",
    "Work": "#48DBFB",
    "Insurance": "#48DBFB",
    "Petcare": "#FFEAA7"
}

def load_data():
    """Load shopping data"""
    if os.path.exists(DATA_FILE):
        df = pd.read_csv(DATA_FILE)
        df['date'] = pd.to_datetime(df['date'])
        return df
    return pd.DataFrame(columns=['date', 'item', 'price', 'category'])

def save_data(df):
    """Save shopping data"""
    df.to_csv(DATA_FILE, index=False)

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = load_data()

# Animated title
st.markdown('<h1 class="main-title">Shopping Tracker</h1>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar - Add new item with nice styling
with st.sidebar:
    st.markdown("### Add New Item")
    st.markdown("")
    
    with st.form("add_item_form"):
        date = st.date_input("Date", value=datetime.now())
        item = st.text_input("Item name", placeholder="e.g., Milk, Bread...")
        price = st.number_input("Price (€)", min_value=0.0, step=0.01, format="%.2f")
        category = st.selectbox("Category", list(CATEGORIES.keys()))
        
        submitted = st.form_submit_button("Add Item", type="primary")
        
        if submitted:
            if item and price > 0:
                new_row = pd.DataFrame([{
                    'date': pd.Timestamp(date),
                    'item': item,
                    'price': price,
                    'category': category
                }])
                st.session_state.data = pd.concat([st.session_state.data, new_row], ignore_index=True)
                save_data(st.session_state.data)
                st.success(f"{item} added!")
            else:
                st.error("Please enter item name and price")

# Main area
df = st.session_state.data

if len(df) == 0:
    st.markdown("""
        <div style='text-align: center; padding: 3rem;'>
            <h2 style='color: #667eea;'>Welcome to Your Shopping Tracker!</h2>
            <p style='font-size: 1.2rem; color: #666;'>
                Start by adding your first item using the sidebar
            </p>
            <p style='font-size: 1rem; color: #999;'>
                Track your spending, analyze patterns, and make smarter shopping decisions!
            </p>
        </div>
    """, unsafe_allow_html=True)
else:
    # Summary metrics with enhanced styling
    st.markdown("### Quick Stats")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    # Calculate spending rate
    date_range = (df['date'].max() - df['date'].min()).days + 1
    spending_per_day = df['price'].sum() / date_range if date_range > 0 else df['price'].sum()
    spending_per_hour = spending_per_day / 24
    spending_per_month = spending_per_day * 30
    
    with col1:
        st.metric("Total Spent", f"€{df['price'].sum():.2f}", delta=None)
    with col2:
        st.metric("Total Items", len(df))
    with col3:
        st.metric("Avg Price", f"€{df['price'].mean():.2f}")
    with col4:
        most_cat = df['category'].mode()[0] if len(df) > 0 else "N/A"
        st.metric("Top Category", most_cat)
    with col5:
        # Toggle between per day/per hour/per month
        rate_type = st.selectbox("Rate", ["Per Day", "Per Hour", "Per Month"], label_visibility="collapsed")
        if rate_type == "Per Day":
            st.metric("Per Day", f"€{spending_per_day:.2f}")
        elif rate_type == "Per Hour":
            st.metric("Per Hour", f"€{spending_per_hour:.2f}")
        else:
            st.metric("Per Month", f"€{spending_per_month:.2f}")
    
    st.markdown("---")
    
    # Tabs with nice content
    tab1, tab2 = st.tabs(["Analytics", "Insights"])
    
    with tab1:
        st.markdown("---")
        
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("#### Spending by Category")
            category_data = df.groupby('category')['price'].sum().sort_values(ascending=False)
            
            # Create dataframe for display with labels
            chart_df = category_data.reset_index()
            chart_df.columns = ['category', 'amount']
            chart_df['label'] = chart_df.apply(lambda x: f"€{x['amount']:.2f}", axis=1)
            
            # Display bar chart
            st.bar_chart(category_data, color="#11315c")
            
            # Show values as text annotations below chart
            value_text = " | ".join([f"{cat}: €{val:.2f}" for cat, val in category_data.items()])
            st.caption(value_text)
        
        with col2:
            st.markdown("#### Recent Purchases")
            recent = df.sort_values('date', ascending=False).head(10)[['date', 'item', 'price', 'category']].copy()
            recent['date'] = recent['date'].dt.strftime('%b %d, %Y')
            recent['price'] = recent['price'].apply(lambda x: f"€{x:.2f}")
            st.dataframe(
                recent, 
                hide_index=True, 
                use_container_width=True,
                column_config={
                    "date": "Date",
                    "item": "Item",
                    "price": "Price",
                    "category": "Category"
                }
            )
    
    with tab2:
        st.markdown("#### Your Shopper Profile")
        
        # Import numpy functions at the top
        from numpy import polyfit, polyval
        
        # Calculate statistics for shopper profile
        avg_transaction = df['price'].mean()
        std_transaction = df['price'].std()
        date_range = (df['date'].max() - df['date'].min()).days + 1
        purchase_frequency = len(df) / date_range if date_range > 0 else len(df)
        
        # Spending trend using linear regression
        df_trend = df.copy()
        df_trend['days'] = (df_trend['date'] - df_trend['date'].min()).dt.days
        
        # Initialize default values
        slope = 0
        intercept = 0
        trend = "stable"
        
        if len(df) > 2:
            try:
                slope, intercept = polyfit(df_trend['days'], df_trend['price'], 1)
                trend = "increasing" if slope > 0.5 else "decreasing" if slope < -0.5 else "stable"
            except:
                trend = "stable"
                slope = 0
                intercept = 0
        
        # Determine shopper type based on patterns
        if purchase_frequency > 2 and avg_transaction < 30:
            shopper_type = "Frequent Buyer"
            description = "You shop often with smaller purchases. You prefer regular trips for essentials."
        elif avg_transaction > 100 and purchase_frequency < 1:
            shopper_type = "Big Spender"
            description = "You make fewer but larger purchases. You plan big shopping trips."
        elif std_transaction > avg_transaction * 0.8:
            shopper_type = "Variety Shopper"
            description = "Your spending varies greatly. You buy both big-ticket items and small purchases."
        elif trend == "increasing":
            shopper_type = "Growing Spender"
            description = "Your spending is trending upward over time. Keep an eye on your budget!"
        elif purchase_frequency > 1:
            shopper_type = "Regular Shopper"
            description = "You maintain consistent shopping habits with steady frequency."
        else:
            shopper_type = "Careful Spender"
            description = "You shop thoughtfully with moderate frequency and spending."
        
        # Display shopper profile
        st.markdown(f"""
            <div style='background-image: linear-gradient(to top, #09203f 0%, #537895 100%); 
                        padding: 2rem; border-radius: 15px; color: white; text-align: center;'>
                <h1 style='margin: 0; font-size: 3rem;'>{shopper_type}</h1>
                <p style='font-size: 1.3rem; margin-top: 1rem; opacity: 0.95;'>{description}</p>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("")
        
        # Statistics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Avg Purchase", f"€{avg_transaction:.2f}")
            st.metric("Purchase Frequency", f"{purchase_frequency:.2f}/day")
        
        with col2:
            st.metric("Spending Trend", trend.capitalize())
            st.metric("Variability", f"€{std_transaction:.2f}")
        
        with col3:
            st.metric("Days Tracked", f"{date_range}")
            top_cat = df['category'].mode()[0] if len(df) > 0 else "N/A"
            st.metric("Favorite Category", top_cat)
        
        # Spending trend chart
        if len(df) > 1:
            st.markdown("---")
            st.markdown("#### Spending Trend Analysis")
            
            # Create trend line
            trend_df = df_trend.sort_values('days')
            trend_df['trend_line'] = polyval([slope, intercept], trend_df['days'])
            
            # Combine actual and trend data for chart
            import altair as alt
            
            # Actual spending points
            actual = alt.Chart(trend_df).mark_circle(size=100, color='#4ECDC4').encode(
                x=alt.X('date:T', title='Date'),
                y=alt.Y('price:Q', title='Price (€)'),
                tooltip=['date:T', 'item:N', 'price:Q']
            )
            
            # Trend line
            trend_line = alt.Chart(trend_df).mark_line(color='#4ECDC4', strokeDash=[5, 5]).encode(
                x='date:T',
                y='trend_line:Q'
            )
            
            chart = (actual + trend_line).properties(height=300)
            st.altair_chart(chart, use_container_width=True)
            
            # Interpretation
            if slope > 0.5:
                st.info("Your spending is increasing over time. Consider setting a monthly budget to stay on track!")
            elif slope < -0.5:
                st.success("Great job! Your spending is decreasing over time.")
            else:
                st.info("Your spending remains relatively stable over time.")
        
        # Delete section
        st.markdown("---")
        st.markdown("#### Data Management")
        if st.button("Clear All Data", type="secondary"):
            if os.path.exists(DATA_FILE):
                os.remove(DATA_FILE)
            st.session_state.data = pd.DataFrame(columns=['date', 'item', 'price', 'category'])
            st.success("All data cleared!")
            st.rerun()
