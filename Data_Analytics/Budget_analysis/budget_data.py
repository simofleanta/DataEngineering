import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Page config
st.set_page_config(
    page_title="Budget Impact Dashboard",
    page_icon="�",
    layout="wide"
)

# Get script directory
script_dir = Path(__file__).parent

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    h1 {
        color: #1f77b4;
        text-align: center;
        padding: 20px 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Title
st.title("Budget Impact Analysis Dashboard")
st.markdown("### How Cost Reduction Affects Your Remaining Budget")
st.divider()

# Load data
@st.cache_data
def load_data():
    df_scenario1 = pd.read_excel(script_dir / 'scenarios.xlsx', sheet_name='scenario1')
    df_new_scenario = pd.read_excel(script_dir / 'scenarios.xlsx', sheet_name='new_scenario')
    trends = pd.read_excel(script_dir / 'scenarios.xlsx', sheet_name='trends')
    stats = pd.read_excel(script_dir / 'scenarios.xlsx', sheet_name='statistics')
    return df_scenario1, df_new_scenario, trends, stats

try:
    df_scenario1, df_new_scenario, trends, stats = load_data()
    
    # Calculate key metrics
    avg_remaining_scenario1 = df_scenario1['remaining'].mean()
    avg_remaining_new = df_new_scenario['remaining'].mean()
    budget_increase = avg_remaining_new - avg_remaining_scenario1
    budget_increase_pct = (budget_increase / avg_remaining_scenario1) * 100 if avg_remaining_scenario1 != 0 else 0
    
    avg_total_scenario1 = df_scenario1['total'].mean()
    avg_total_new = df_new_scenario['total'].mean()
    cost_reduction = avg_total_scenario1 - avg_total_new
    cost_reduction_pct = (cost_reduction / avg_total_scenario1) * 100 if avg_total_scenario1 != 0 else 0
    
    # Key Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Budget Increase",
            value=f"{budget_increase:.2f} Ron",
            delta=f"{budget_increase_pct:.1f}%"
        )
    
    with col2:
        st.metric(
            label="Cost Reduction",
            value=f"{cost_reduction:.2f} Ron",
            delta=f"-{cost_reduction_pct:.1f}%",
            delta_color="inverse"
        )
    
    with col3:
        st.metric(
            label="New Avg Remaining",
            value=f"{avg_remaining_new:.2f} Ron"
        )
    
    with col4:
        st.metric(
            label="New Avg Total Cost",
            value=f"{avg_total_new:.2f} Ron"
        )
    
    st.divider()
    
    # Main Charts
    months = range(1, 13)
    
    # Set style
    sns.set_style("whitegrid")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Total Costs Trend")
        fig1, ax1 = plt.subplots(figsize=(8, 5))
        
        ax1.plot(months, df_scenario1['total'], marker='o', label='Scenario 1 Total', 
                linewidth=2.5, markersize=7, color='steelblue', alpha=0.7)
        trend1_total = trends[(trends['scenario'] == 'scenario1') & (trends['metric'] == 'total')]
        if not trend1_total.empty:
            slope1 = trend1_total['slope'].values[0]
            intercept1 = trend1_total['intercept'].values[0]
            trend_line1 = slope1 * np.array(range(12)) + intercept1
            ax1.plot(months, trend_line1, '--', alpha=0.5, linewidth=2, color='darkred', 
                    label=f'S1 Trend (slope={slope1:.2f})')
        
        ax1.plot(months, df_new_scenario['total'], marker='s', label='New Scenario Total', 
                linewidth=2.5, markersize=7, color='teal', alpha=0.7)
        trend_new_total = trends[(trends['scenario'] == 'new_scenario') & (trends['metric'] == 'total')]
        if not trend_new_total.empty:
            slope_new = trend_new_total['slope'].values[0]
            intercept_new = trend_new_total['intercept'].values[0]
            trend_line_new = slope_new * np.array(range(12)) + intercept_new
            ax1.plot(months, trend_line_new, '--', alpha=0.5, linewidth=2, color='cadetblue', 
                    label=f'New Trend (slope={slope_new:.2f})')
        
        ax1.set_xlabel('Month', fontweight='bold')
        ax1.set_ylabel('Total Costs (Ron)', fontweight='bold')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        ax1.set_xticks(months)
        
        st.pyplot(fig1)
        
        st.info(f"**Insight**: By reducing ph & slc costs, total monthly costs decreased by {cost_reduction:.2f} Ron on average ({cost_reduction_pct:.1f}%)")
    
    with col2:
        st.subheader("Remaining Budget Trend")
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        
        ax2.plot(months, df_scenario1['remaining'], marker='o', label='Scenario 1 Remaining', 
                linewidth=2.5, markersize=7, color='cadetblue', alpha=0.7)
        trend1_remaining = trends[(trends['scenario'] == 'scenario1') & (trends['metric'] == 'remaining')]
        if not trend1_remaining.empty:
            slope1 = trend1_remaining['slope'].values[0]
            intercept1 = trend1_remaining['intercept'].values[0]
            trend_line1 = slope1 * np.array(range(12)) + intercept1
            ax2.plot(months, trend_line1, '--', alpha=0.5, linewidth=2, color='cadetblue', 
                    label=f'S1 Trend (slope={slope1:.2f})')
        
        ax2.plot(months, df_new_scenario['remaining'], marker='s', label='New Scenario Remaining', 
                linewidth=2.5, markersize=7, color='darkred', alpha=0.7)
        trend_new_remaining = trends[(trends['scenario'] == 'new_scenario') & (trends['metric'] == 'remaining')]
        if not trend_new_remaining.empty:
            slope_new = trend_new_remaining['slope'].values[0]
            intercept_new = trend_new_remaining['intercept'].values[0]
            trend_line_new = slope_new * np.array(range(12)) + intercept_new
            ax2.plot(months, trend_line_new, '--', alpha=0.5, linewidth=2, color='darkorange', 
                    label=f'New Trend (slope={slope_new:.2f})')
        
        ax2.set_xlabel('Month', fontweight='bold')
        ax2.set_ylabel('Remaining Budget (Ron)', fontweight='bold')
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)
        ax2.set_xticks(months)
        
        st.pyplot(fig2)
        
        st.success(f"**Key Result**: Your remaining budget increased by {budget_increase:.2f} Ron per month ({budget_increase_pct:.1f}% improvement)!")
    
    # Detailed statistics (expandable)
    with st.expander("View Detailed Statistics"):
        st.subheader("Statistical Analysis")
        
        # Filter stats for total and remaining metrics
        stats_display = stats[stats['metric'].isin(['total', 'remaining'])].copy()
        stats_display = stats_display.round(2)
        
        # Display as styled table
        st.dataframe(
            stats_display,
            use_container_width=True,
            hide_index=True
        )
    
    # Raw data tables (expandable)
    with st.expander("View Raw Data"):
        tab1, tab2 = st.tabs(["Scenario 1", "New Scenario"])
        
        with tab1:
            st.dataframe(df_scenario1, use_container_width=True)
        
        with tab2:
            st.dataframe(df_new_scenario, use_container_width=True)

except FileNotFoundError:
    st.error("Error: scenarios.xlsx file not found. Please make sure the file exists in the same directory as this script.")
except Exception as e:
    st.error(f"An error occurred: {str(e)}")
