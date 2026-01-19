"""
Garmin Sleep Data Dashboard
Streamlit application for visualizing sleep metrics from Garmin Connect
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# Page configuration
st.set_page_config(
    page_title="Garmin Sleep Dashboard",
    page_icon="🌙",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
    }
    </style>
    """, unsafe_allow_html=True)

# Title
st.title("Garmin Sleep Data Dashboard")
st.markdown("Comprehensive analysis of sleep metrics and patterns")

# Load data
@st.cache_data
def load_sleep_data():
    garmin_folder = Path(__file__).parent
    csv_file = garmin_folder / 'sleep_garmin.csv'
    
    if not csv_file.exists():
        st.error(f"File {csv_file} not found! Run sleep.py first to extract data.")
        st.stop()
    
    df = pd.read_csv(csv_file)
    df['Date'] = pd.to_datetime(df['Date'])
    return df

try:
    df = load_sleep_data()
    
    # Sidebar filters
    st.sidebar.header("Filters")
    
    # Date range filter
    min_date = df['Date'].min().date()
    max_date = df['Date'].max().date()
    
    date_range = st.sidebar.date_input(
        "Select date range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    
    if len(date_range) == 2:
        start_date, end_date = date_range
        mask = (df['Date'].dt.date >= start_date) & (df['Date'].dt.date <= end_date)
        df_filtered = df.loc[mask]
    else:
        df_filtered = df
    
    # Sleep quality filter
    quality_options = ['All'] + sorted(df['Sleep Quality'].dropna().unique().tolist())
    selected_quality = st.sidebar.selectbox("Sleep quality", quality_options)
    
    if selected_quality != 'All':
        df_filtered = df_filtered[df_filtered['Sleep Quality'] == selected_quality]
    
    # Main metrics
    st.header("Summary Statistics")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        avg_sleep = df_filtered['Total Sleep Duration (hours)'].mean()
        st.metric("Average Sleep Duration", f"{avg_sleep:.2f}h")
    
    with col2:
        avg_score = df_filtered['Sleep Score'].mean()
        st.metric("Average Sleep Score", f"{avg_score:.1f}/100")
    
    with col3:
        avg_deep = df_filtered['Deep Sleep (hours)'].mean()
        st.metric("Average Deep Sleep", f"{avg_deep:.2f}h")
    
    with col4:
        avg_rem = df_filtered['REM Sleep (hours)'].mean()
        st.metric("Average REM Sleep", f"{avg_rem:.2f}h")
    
    with col5:
        avg_stress = df_filtered['Average Stress Level'].mean()
        st.metric("Average Stress Level", f"{avg_stress:.1f}")
    
    # Tabs for different visualizations
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Sleep Duration", 
        "Sleep Phases", 
        "Sleep Score",
        "Weekly Patterns",
        "Daily Analysis",
        "Statistical Analysis"
    ])
    
    # Tab 1: Sleep Duration
    with tab1:
        st.subheader("Sleep Duration Over Time")
        
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(
            x=df_filtered['Date'],
            y=df_filtered['Total Sleep Duration (hours)'],
            mode='lines+markers',
            name='Sleep Duration',
            line=dict(color="#0D7390", width=2),
            marker=dict(size=8),
            hovertemplate='<b>%{x|%Y-%m-%d}</b><br>Duration: %{y:.2f}h<extra></extra>'
        ))
        
        # Add recommended sleep line
        fig1.add_hline(
            y=7, 
            line_dash="dash", 
            line_color="green",
            annotation_text="Recommended minimum: 7h",
            annotation_position="right"
        )
        
        fig1.add_hline(
            y=9, 
            line_dash="dash", 
            line_color="orange",
            annotation_text="Recommended maximum: 9h",
            annotation_position="right"
        )
        
        fig1.update_layout(
            xaxis_title="Date",
            yaxis_title="Hours",
            hovermode='x unified',
            template='plotly_white',
            height=500
        )
        
        st.plotly_chart(fig1, width='stretch')
        
        # Additional stats
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Maximum Sleep", f"{df_filtered['Total Sleep Duration (hours)'].max():.2f}h")
        with col2:
            st.metric("Minimum Sleep", f"{df_filtered['Total Sleep Duration (hours)'].min():.2f}h")
        with col3:
            nights_under_7h = (df_filtered['Total Sleep Duration (hours)'] < 7).sum()
            st.metric("Nights Under 7h", f"{nights_under_7h}")
    
    # Tab 2: Sleep Phases
    with tab2:
        st.subheader("Sleep Phases Distribution")
        
        fig2 = go.Figure()
        
        fig2.add_trace(go.Bar(
            x=df_filtered['Date'],
            y=df_filtered['Deep Sleep (hours)'],
            name='Deep Sleep',
            marker_color='#2C3E50',
            hovertemplate='Deep: %{y:.2f}h<extra></extra>'
        ))
        
        fig2.add_trace(go.Bar(
            x=df_filtered['Date'],
            y=df_filtered['Light Sleep (hours)'],
            name='Light Sleep',
            marker_color="#139CAB",
            hovertemplate='Light: %{y:.2f}h<extra></extra>'
        ))
        
        fig2.add_trace(go.Bar(
            x=df_filtered['Date'],
            y=df_filtered['REM Sleep (hours)'],
            name='REM Sleep',
            marker_color="#E7EEF9",
            hovertemplate='REM: %{y:.2f}h<extra></extra>'
        ))
        
        fig2.add_trace(go.Bar(
            x=df_filtered['Date'],
            y=df_filtered['Awake (hours)'],
            name='Awake',
            marker_color="#981B0E",
            hovertemplate='Awake: %{y:.2f}h<extra></extra>'
        ))
        
        fig2.update_layout(
            barmode='stack',
            xaxis_title="Date",
            yaxis_title="Hours",
            template='plotly_white',
            height=500,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig2, width='stretch')
        
        # Phase distribution pie chart
        st.subheader("Average Phase Distribution")
        
        phase_data = {
            'Phase': ['Deep Sleep', 'Light Sleep', 'REM Sleep', 'Awake'],
            'Hours': [
                df_filtered['Deep Sleep (hours)'].mean(),
                df_filtered['Light Sleep (hours)'].mean(),
                df_filtered['REM Sleep (hours)'].mean(),
                df_filtered['Awake (hours)'].mean()
            ]
        }
        
        fig_pie = px.pie(
            phase_data, 
            values='Hours', 
            names='Phase',
            color='Phase',
            color_discrete_map={
                'Deep Sleep': '#2C3E50',
                'Light Sleep': "#0D6B70",
                'REM Sleep': "#B6B7BC",
                'Awake': "#92190C"
            }
        )
        
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        fig_pie.update_layout(height=400)
        
        st.plotly_chart(fig_pie, width='stretch')
    
    # Tab 3: Sleep Score
    with tab3:
        st.subheader("Sleep Score Trends")
        
        fig3 = go.Figure()
        
        # Create color scale based on score
        colors = df_filtered['Sleep Score'].apply(
            lambda x: "#15778F" if x >= 80 else "#DEEAEE" if x >= 60 else "#A72011"
        )
        
        fig3.add_trace(go.Scatter(
            x=df_filtered['Date'],
            y=df_filtered['Sleep Score'],
            mode='lines+markers',
            name='Sleep Score',
            line=dict(color='#9B9B9B', width=1),
            marker=dict(
                size=10, 
                color=colors,
                line=dict(width=1, color='white')
            ),
            hovertemplate='<b>%{x|%Y-%m-%d}</b><br>Score: %{y:.0f}/100<extra></extra>'
        ))
        
        fig3.update_layout(
            xaxis_title="Date",
            yaxis_title="Score (0-100)",
            hovermode='x unified',
            template='plotly_white',
            height=500
        )
        
        st.plotly_chart(fig3, width='stretch')
        
        # Score distribution
        st.subheader("Score Distribution")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_hist = px.histogram(
                df_filtered, 
                x='Sleep Score',
                nbins=20,
                title="Sleep Score Frequency",
                labels={'Sleep Score': 'Sleep Score', 'count': 'Frequency'},
                color_discrete_sequence=['#0D6B70']  # Culoarea pentru histogram
            )
            fig_hist.update_layout(height=400)
            st.plotly_chart(fig_hist, width='stretch')
        
        with col2:
            # Sleep quality distribution
            quality_counts = df_filtered['Sleep Quality'].value_counts()
            fig_quality = px.bar(
                x=quality_counts.index,
                y=quality_counts.values,
                title="Sleep Quality Distribution",
                labels={'x': 'Quality', 'y': 'Count'},
                color_discrete_sequence=['#92190C']  # Culoarea pentru bar chart
            )
            fig_quality.update_layout(height=400)
            st.plotly_chart(fig_quality, width='stretch')
    
    # Tab 4: Weekly Patterns
    with tab4:
        st.subheader("Sleep Patterns by Day of Week")
        
        # Add day of week to dataframe
        df_weekly = df_filtered.copy()
        df_weekly['Day of Week'] = df_weekly['Date'].dt.day_name()
        df_weekly['Day Number'] = df_weekly['Date'].dt.dayofweek
        
        # Define day order
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        
        # Calculate statistics by day of week
        weekly_stats = df_weekly.groupby('Day of Week').agg({
            'Total Sleep Duration (hours)': ['mean', 'std', 'count'],
            'Sleep Score': ['mean', 'std'],
            'Deep Sleep (hours)': 'mean',
            'Light Sleep (hours)': 'mean',
            'REM Sleep (hours)': 'mean',
            'Awake (hours)': 'mean'
        }).reset_index()
        
        # Flatten column names
        weekly_stats.columns = ['Day of Week', 'Avg Sleep Duration', 'Std Sleep Duration', 'Count',
                                'Avg Sleep Score', 'Std Sleep Score', 'Avg Deep Sleep',
                                'Avg Light Sleep', 'Avg REM Sleep', 'Avg Awake']
        
        # Sort by day order
        weekly_stats['Day Number'] = weekly_stats['Day of Week'].map(
            {day: i for i, day in enumerate(day_order)}
        )
        weekly_stats = weekly_stats.sort_values('Day Number')
        
        # Main visualization: Sleep Duration by Day of Week
        st.subheader("Average Sleep Duration by Day of Week")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_duration_day = go.Figure()
            
            fig_duration_day.add_trace(go.Bar(
                x=weekly_stats['Day of Week'],
                y=weekly_stats['Avg Sleep Duration'],
                marker_color="#1287AE",
                error_y=dict(
                    type='data',
                    array=weekly_stats['Std Sleep Duration'],
                    visible=True
                ),
                text=weekly_stats['Avg Sleep Duration'].round(2),
                textposition='outside',
                hovertemplate='%{x}<br>Avg: %{y:.2f}h<br>Nights: ' + 
                             weekly_stats['Count'].astype(str) + '<extra></extra>'
            ))
            
            # Add recommended sleep line
            fig_duration_day.add_hline(
                y=7,
                line_dash="dash",
                line_color="green",
                annotation_text="Recommended: 7h"
            )
            
            fig_duration_day.update_layout(
                xaxis_title="Day of Week",
                yaxis_title="Average Sleep Duration (hours)",
                template='plotly_white',
                height=400,
                xaxis={'categoryorder': 'array', 'categoryarray': day_order}
            )
            
            st.plotly_chart(fig_duration_day, width='stretch')
        
        with col2:
            # Sleep Score by Day of Week
            fig_score_day = go.Figure()
            
            fig_score_day.add_trace(go.Bar(
                x=weekly_stats['Day of Week'],
                y=weekly_stats['Avg Sleep Score'],
                marker_color="#B0290B",
                error_y=dict(
                    type='data',
                    array=weekly_stats['Std Sleep Score'],
                    visible=True
                ),
                text=weekly_stats['Avg Sleep Score'].round(1),
                textposition='outside',
                hovertemplate='%{x}<br>Avg Score: %{y:.1f}/100<extra></extra>'
            ))
            
            fig_score_day.update_layout(
                xaxis_title="Day of Week",
                yaxis_title="Average Sleep Score",
                template='plotly_white',
                height=400,
                xaxis={'categoryorder': 'array', 'categoryarray': day_order}
            )
            
            st.plotly_chart(fig_score_day, width='stretch')
        
        # Best and Worst Days
        st.subheader("Sleep Performance Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        
        best_duration_day = weekly_stats.loc[weekly_stats['Avg Sleep Duration'].idxmax(), 'Day of Week']
        worst_duration_day = weekly_stats.loc[weekly_stats['Avg Sleep Duration'].idxmin(), 'Day of Week']
        best_score_day = weekly_stats.loc[weekly_stats['Avg Sleep Score'].idxmax(), 'Day of Week']
        worst_score_day = weekly_stats.loc[weekly_stats['Avg Sleep Score'].idxmin(), 'Day of Week']
        
        with col1:
            st.metric(
                "Best Sleep Duration Day",
                best_duration_day,
                f"{weekly_stats.loc[weekly_stats['Avg Sleep Duration'].idxmax(), 'Avg Sleep Duration']:.2f}h"
            )
        
        with col2:
            st.metric(
                "Worst Sleep Duration Day",
                worst_duration_day,
                f"{weekly_stats.loc[weekly_stats['Avg Sleep Duration'].idxmin(), 'Avg Sleep Duration']:.2f}h"
            )
        
        with col3:
            st.metric(
                "Best Sleep Score Day",
                best_score_day,
                f"{weekly_stats.loc[weekly_stats['Avg Sleep Score'].idxmax(), 'Avg Sleep Score']:.1f}/100"
            )
        
        with col4:
            st.metric(
                "Worst Sleep Score Day",
                worst_score_day,
                f"{weekly_stats.loc[weekly_stats['Avg Sleep Score'].idxmin(), 'Avg Sleep Score']:.1f}/100"
            )
        
        # Sleep Phases by Day of Week
        st.subheader("Sleep Phases Distribution by Day of Week")
        
        fig_phases_day = go.Figure()
        
        fig_phases_day.add_trace(go.Bar(
            name='Deep Sleep',
            x=weekly_stats['Day of Week'],
            y=weekly_stats['Avg Deep Sleep'],
            marker_color='#2C3E50'
        ))
        
        fig_phases_day.add_trace(go.Bar(
            name='Light Sleep',
            x=weekly_stats['Day of Week'],
            y=weekly_stats['Avg Light Sleep'],
            marker_color="#0C7FA9"
        ))
        
        fig_phases_day.add_trace(go.Bar(
            name='REM Sleep',
            x=weekly_stats['Day of Week'],
            y=weekly_stats['Avg REM Sleep'],
            marker_color="#DCE1E9"
        ))
        
        fig_phases_day.add_trace(go.Bar(
            name='Awake',
            x=weekly_stats['Day of Week'],
            y=weekly_stats['Avg Awake'],
            marker_color="#A81302"
        ))
        
        fig_phases_day.update_layout(
            barmode='stack',
            xaxis_title="Day of Week",
            yaxis_title="Average Hours",
            template='plotly_white',
            height=500,
            xaxis={'categoryorder': 'array', 'categoryarray': day_order}
        )
        
        st.plotly_chart(fig_phases_day, width='stretch')
        
        # Detailed statistics table
        st.subheader("Detailed Weekly Statistics")
        
        display_stats = weekly_stats[[
            'Day of Week', 
            'Avg Sleep Duration', 
            'Avg Sleep Score',
            'Avg Deep Sleep',
            'Avg Light Sleep',
            'Avg REM Sleep',
            'Count'
        ]].copy()
        
        display_stats.columns = [
            'Day',
            'Avg Duration (h)',
            'Avg Score',
            'Deep Sleep (h)',
            'Light Sleep (h)',
            'REM Sleep (h)',
            'Nights'
        ]
        
        # Format numbers
        for col in ['Avg Duration (h)', 'Deep Sleep (h)', 'Light Sleep (h)', 'REM Sleep (h)']:
            display_stats[col] = display_stats[col].round(2)
        display_stats['Avg Score'] = display_stats['Avg Score'].round(1)
        
        st.dataframe(display_stats, width='stretch', hide_index=True)
        
        # Weekday vs Weekend comparison
        st.subheader("Weekday vs Weekend Comparison")
        
        df_weekly['Is Weekend'] = df_weekly['Day Number'].isin([5, 6])  # Saturday, Sunday
        
        weekday_vs_weekend = df_weekly.groupby('Is Weekend').agg({
            'Total Sleep Duration (hours)': 'mean',
            'Sleep Score': 'mean',
            'Deep Sleep (hours)': 'mean',
            'REM Sleep (hours)': 'mean'
        }).reset_index()
        
        weekday_vs_weekend['Period'] = weekday_vs_weekend['Is Weekend'].map({
            True: 'Weekend',
            False: 'Weekday'
        })
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Duration comparison
            fig_wd_we = go.Figure()
            fig_wd_we.add_trace(go.Bar(
                x=weekday_vs_weekend['Period'],
                y=weekday_vs_weekend['Total Sleep Duration (hours)'],
                marker_color=["#C5CDD3", "#0D7EA0"],
                text=weekday_vs_weekend['Total Sleep Duration (hours)'].round(2),
                textposition='outside'
            ))
            fig_wd_we.update_layout(
                title="Average Sleep Duration",
                yaxis_title="Hours",
                template='plotly_white',
                height=350,
                showlegend=False
            )
            st.plotly_chart(fig_wd_we, width='stretch')
        
        with col2:
            # Score comparison
            fig_score_we = go.Figure()
            fig_score_we.add_trace(go.Bar(
                x=weekday_vs_weekend['Period'],
                y=weekday_vs_weekend['Sleep Score'],
                marker_color=['#C5CDD3', '#0D7EA0'],
                text=weekday_vs_weekend['Sleep Score'].round(1),
                textposition='outside'
            ))
            fig_score_we.update_layout(
                title="Average Sleep Score",
                yaxis_title="Score",
                template='plotly_white',
                height=350,
                showlegend=False
            )
            st.plotly_chart(fig_score_we, width='stretch')
    
    # Tab 5: Daily Analysis
    with tab5:
        st.subheader("Daily Sleep Analysis")
        
        # Select specific date for detailed analysis
        st.write("Select a date to view detailed sleep information:")
        
        available_dates = sorted(df_filtered['Date'].dt.date.unique(), reverse=True)
        
        if len(available_dates) > 0:
            selected_date = st.selectbox(
                "Choose date:",
                available_dates,
                format_func=lambda x: x.strftime('%A, %B %d, %Y')
            )
            
            # Filter data for selected date
            selected_data = df_filtered[df_filtered['Date'].dt.date == selected_date].iloc[0]
            
            # Display main metrics
            st.subheader(f"Sleep Summary for {selected_date.strftime('%A, %B %d, %Y')}")
            
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric(
                    "Total Sleep",
                    f"{selected_data['Total Sleep Duration (hours)']:.2f}h",
                    delta=None
                )
            
            with col2:
                st.metric(
                    "Sleep Score",
                    f"{selected_data['Sleep Score']:.0f}/100",
                    delta=None
                )
            
            with col3:
                st.metric(
                    "Sleep Quality",
                    selected_data['Sleep Quality']
                )
            
            with col4:
                st.metric(
                    "Stress Level",
                    f"{selected_data['Average Stress Level']:.1f}" if pd.notna(selected_data['Average Stress Level']) else "N/A"
                )
            
            with col5:
                day_of_week = pd.to_datetime(selected_date).day_name()
                st.metric(
                    "Day of Week",
                    day_of_week
                )
            
            # Sleep phases breakdown
            st.subheader("Sleep Phases Breakdown")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Pie chart of sleep phases
                phases_data = {
                    'Phase': ['Deep Sleep', 'Light Sleep', 'REM Sleep', 'Awake'],
                    'Duration': [
                        selected_data['Deep Sleep (hours)'],
                        selected_data['Light Sleep (hours)'],
                        selected_data['REM Sleep (hours)'],
                        selected_data['Awake (hours)']
                    ]
                }
                
                fig_phases_pie = px.pie(
                    phases_data,
                    values='Duration',
                    names='Phase',
                    title='Sleep Phases Distribution',
                    color='Phase',
                    color_discrete_map={
                        'Deep Sleep': '#2C3E50',
                        'Light Sleep': "#09758A",
                        'REM Sleep': "#CFD9EB",
                        'Awake': "#9E1E0F"
                    }
                )
                fig_phases_pie.update_traces(textposition='inside', textinfo='percent+label')
                fig_phases_pie.update_layout(height=400)
                st.plotly_chart(fig_phases_pie, width='stretch')
            
            with col2:
                # Bar chart of sleep phases
                fig_phases_bar = go.Figure()
                
                fig_phases_bar.add_trace(go.Bar(
                    x=['Deep', 'Light', 'REM', 'Awake'],
                    y=[
                        selected_data['Deep Sleep (hours)'],
                        selected_data['Light Sleep (hours)'],
                        selected_data['REM Sleep (hours)'],
                        selected_data['Awake (hours)']
                    ],
                    marker_color=['#2C3E50', "#148799", "#0BBAD9", "#9C1F11"],
                    text=[
                        f"{selected_data['Deep Sleep (hours)']:.2f}h",
                        f"{selected_data['Light Sleep (hours)']:.2f}h",
                        f"{selected_data['REM Sleep (hours)']:.2f}h",
                        f"{selected_data['Awake (hours)']:.2f}h"
                    ],
                    textposition='outside'
                ))
                
                fig_phases_bar.update_layout(
                    title='Sleep Phases Duration',
                    xaxis_title='Phase',
                    yaxis_title='Hours',
                    template='plotly_white',
                    height=400,
                    showlegend=False
                )
                
                st.plotly_chart(fig_phases_bar, width='stretch')
            
            # Detailed metrics
            st.subheader("Detailed Metrics")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Deep Sleep", f"{selected_data['Deep Sleep (hours)']:.2f}h")
                deep_pct = (selected_data['Deep Sleep (hours)'] / selected_data['Total Sleep Duration (hours)'] * 100)
                st.caption(f"{deep_pct:.1f}% of total sleep")
            
            with col2:
                st.metric("Light Sleep", f"{selected_data['Light Sleep (hours)']:.2f}h")
                light_pct = (selected_data['Light Sleep (hours)'] / selected_data['Total Sleep Duration (hours)'] * 100)
                st.caption(f"{light_pct:.1f}% of total sleep")
            
            with col3:
                st.metric("REM Sleep", f"{selected_data['REM Sleep (hours)']:.2f}h")
                rem_pct = (selected_data['REM Sleep (hours)'] / selected_data['Total Sleep Duration (hours)'] * 100)
                st.caption(f"{rem_pct:.1f}% of total sleep")
            
            with col4:
                st.metric("Awake Time", f"{selected_data['Awake (hours)']:.2f}h")
                awake_pct = (selected_data['Awake (hours)'] / selected_data['Total Sleep Duration (hours)'] * 100)
                st.caption(f"{awake_pct:.1f}% of total sleep")
            
            # Comparison with averages
            st.subheader("Comparison with Your Averages")
            
            avg_duration = df_filtered['Total Sleep Duration (hours)'].mean()
            avg_score = df_filtered['Sleep Score'].mean()
            avg_deep = df_filtered['Deep Sleep (hours)'].mean()
            avg_rem = df_filtered['REM Sleep (hours)'].mean()
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                duration_diff = selected_data['Total Sleep Duration (hours)'] - avg_duration
                st.metric(
                    "Total Sleep vs Avg",
                    f"{selected_data['Total Sleep Duration (hours)']:.2f}h",
                    delta=f"{duration_diff:+.2f}h"
                )
            
            with col2:
                score_diff = selected_data['Sleep Score'] - avg_score
                st.metric(
                    "Sleep Score vs Avg",
                    f"{selected_data['Sleep Score']:.0f}",
                    delta=f"{score_diff:+.1f}"
                )
            
            with col3:
                deep_diff = selected_data['Deep Sleep (hours)'] - avg_deep
                st.metric(
                    "Deep Sleep vs Avg",
                    f"{selected_data['Deep Sleep (hours)']:.2f}h",
                    delta=f"{deep_diff:+.2f}h"
                )
            
            with col4:
                rem_diff = selected_data['REM Sleep (hours)'] - avg_rem
                st.metric(
                    "REM Sleep vs Avg",
                    f"{selected_data['REM Sleep (hours)']:.2f}h",
                    delta=f"{rem_diff:+.2f}h"
                )
            
            # Sleep times
            if pd.notna(selected_data['Sleep Start']) and pd.notna(selected_data['Sleep End']):
                st.subheader("Sleep Times")
                
                from datetime import datetime
                
                sleep_start = datetime.fromtimestamp(selected_data['Sleep Start'] / 1000)
                sleep_end = datetime.fromtimestamp(selected_data['Sleep End'] / 1000)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Sleep Start", sleep_start.strftime('%I:%M %p'))
                
                with col2:
                    st.metric("Sleep End", sleep_end.strftime('%I:%M %p'))
            
            # Calendar navigation
            st.subheader("Quick Navigation")
            
            col1, col2, col3, col4 = st.columns(4)
            
            current_idx = available_dates.index(selected_date)
            
            with col1:
                if current_idx < len(available_dates) - 1:
                    if st.button("← Previous Day"):
                        st.rerun()
            
            with col2:
                if current_idx > 0:
                    if st.button("Next Day →"):
                        st.rerun()
            
            with col3:
                # Jump to best sleep day
                best_day = df_filtered.loc[df_filtered['Sleep Score'].idxmax(), 'Date'].date()
                if st.button("Best Sleep Day"):
                    st.rerun()
            
            with col4:
                # Jump to worst sleep day
                worst_day = df_filtered.loc[df_filtered['Sleep Score'].idxmin(), 'Date'].date()
                if st.button("Worst Sleep Day"):
                    st.rerun()
        
        else:
            st.warning("No data available for daily analysis.")
    
    # Tab 6: Statistical Analysis
    with tab6:
        st.header("Statistical Analysis")
        
        # Prepare data
        df_stats = df_filtered.copy()
        df_stats['Day Number'] = df_stats['Date'].dt.dayofweek
        df_stats['Is Weekend'] = df_stats['Day Number'].isin([5, 6])
        df_stats['Deep Sleep %'] = (df_stats['Deep Sleep (hours)'] / df_stats['Total Sleep Duration (hours)'] * 100)
        df_stats['REM Sleep %'] = (df_stats['REM Sleep (hours)'] / df_stats['Total Sleep Duration (hours)'] * 100)
        df_stats['Light Sleep %'] = (df_stats['Light Sleep (hours)'] / df_stats['Total Sleep Duration (hours)'] * 100)
        df_stats['Days Since Start'] = (df_stats['Date'] - df_stats['Date'].min()).dt.days
        
        # Section 1: Correlation Analysis
        st.subheader("1. Correlation Analysis")
        
        # Select numeric columns for correlation
        corr_columns = [
            'Total Sleep Duration (hours)',
            'Sleep Score',
            'Deep Sleep (hours)',
            'Light Sleep (hours)',
            'REM Sleep (hours)',
            'Awake (hours)',
            'Deep Sleep %',
            'REM Sleep %',
            'Average Stress Level'
        ]
        
        # Filter columns that exist and have data
        available_corr_cols = [col for col in corr_columns if col in df_stats.columns and df_stats[col].notna().any()]
        
        correlation_matrix = df_stats[available_corr_cols].corr()
        
        # Heatmap
        fig_corr = px.imshow(
            correlation_matrix,
            labels=dict(color="Correlation"),
            x=correlation_matrix.columns,
            y=correlation_matrix.columns,
            color_continuous_scale='PuBu',
            zmin=-1,
            zmax=1,
            title="Correlation Matrix - Sleep Metrics"
        )
        
        fig_corr.update_layout(height=600)
        st.plotly_chart(fig_corr, width='stretch')
        
        st.markdown("**Key Insights:**")
        st.markdown("""
        - **Strong positive correlation (>0.5)**: Variables move together - when one increases, the other tends to increase
        - **Strong negative correlation (<-0.5)**: Variables move in opposite directions
        - **Weak correlation (-0.3 to 0.3)**: Little to no linear relationship between variables
        - Red colors indicate negative correlations, blue indicates positive correlations
        """)
        
        # Key correlations
        st.subheader("Key Correlations with Sleep Score")
        
        score_correlations = correlation_matrix['Sleep Score'].drop('Sleep Score').sort_values(ascending=False)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Positive Correlations:**")
            positive_corr = score_correlations[score_correlations > 0].head(3)
            for metric, value in positive_corr.items():
                st.metric(metric, f"{value:.3f}")
        
        with col2:
            st.write("**Negative Correlations:**")
            negative_corr = score_correlations[score_correlations < 0].tail(3)
            for metric, value in negative_corr.items():
                st.metric(metric, f"{value:.3f}")
        
        st.markdown("**Interpretation:**")
        if len(positive_corr) > 0:
            top_positive = positive_corr.index[0]
            top_pos_value = positive_corr.iloc[0]
            st.markdown(f"- **{top_positive}** has the strongest positive relationship with sleep score (r={top_pos_value:.3f})")
            st.markdown(f"  - Increasing this metric is associated with better sleep scores")
        if len(negative_corr) > 0:
            top_negative = negative_corr.index[-1]
            top_neg_value = negative_corr.iloc[-1]
            st.markdown(f"- **{top_negative}** has the strongest negative relationship with sleep score (r={top_neg_value:.3f})")
            st.markdown(f"  - Higher values of this metric are associated with lower sleep scores")
        
        # Section 2: Regression Analysis
        st.subheader("2. Multiple Linear Regression - Sleep Score Prediction")
        
        # Prepare features
        feature_cols = ['Total Sleep Duration (hours)', 'Deep Sleep %', 'REM Sleep %', 'Average Stress Level']
        available_features = [col for col in feature_cols if col in df_stats.columns]
        
        df_regression = df_stats[available_features + ['Sleep Score']].dropna()
        
        if len(df_regression) > 5:
            X = df_regression[available_features]
            y = df_regression['Sleep Score']
            
            # Fit model
            model = LinearRegression()
            model.fit(X, y)
            y_pred = model.predict(X)
            
            # Metrics
            r2 = r2_score(y, y_pred)
            rmse = np.sqrt(mean_squared_error(y, y_pred))
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("R² Score", f"{r2:.3f}")
            with col2:
                st.metric("RMSE", f"{rmse:.2f}")
            with col3:
                st.metric("Model Accuracy", f"{r2*100:.1f}%")
            
            # Feature importance
            st.write("**Feature Coefficients (Impact on Sleep Score):**")
            
            coefficients = pd.DataFrame({
                'Feature': available_features,
                'Coefficient': model.coef_,
                'Abs Coefficient': np.abs(model.coef_)
            }).sort_values('Abs Coefficient', ascending=False)
            
            fig_coef = px.bar(
                coefficients,
                x='Coefficient',
                y='Feature',
                orientation='h',
                title='Feature Impact on Sleep Score',
                color='Coefficient',
                color_continuous_scale='Reds'
            )
            fig_coef.update_layout(height=300)
            st.plotly_chart(fig_coef, width='stretch')
            
            # Actual vs Predicted
            fig_pred = go.Figure()
            
            fig_pred.add_trace(go.Scatter(
                x=y,
                y=y_pred,
                mode='markers',
                name='Predictions',
                marker=dict(size=8, color="#10A0AF")
            ))
            
            # Perfect prediction line
            min_val = min(y.min(), y_pred.min())
            max_val = max(y.max(), y_pred.max())
            fig_pred.add_trace(go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                name='Perfect Prediction',
                line=dict(color='red', dash='dash')
            ))
            
            fig_pred.update_layout(
                title=f'Actual vs Predicted Sleep Score (R²={r2:.3f})',
                xaxis_title='Actual Score',
                yaxis_title='Predicted Score',
                height=400
            )
            
            st.plotly_chart(fig_pred, width='stretch')
            
            st.markdown("**Model Interpretation:**")
            st.markdown(f"- **R² Score ({r2:.3f})**: The model explains {r2*100:.1f}% of the variance in sleep scores")
            if r2 > 0.7:
                st.markdown("  - **Strong predictive power** - the selected features are good predictors of sleep quality")
            elif r2 > 0.4:
                st.markdown("  - **Moderate predictive power** - other factors also influence sleep quality")
            else:
                st.markdown("  - **Weak predictive power** - sleep score is influenced by factors not captured here")
            
            st.markdown(f"- **RMSE ({rmse:.2f})**: Average prediction error is {rmse:.2f} points")
            st.markdown("- **Feature Importance**: The chart above shows which factors have the biggest impact on your sleep score")
            st.markdown("  - Positive coefficients = increasing this improves your score")
            st.markdown("  - Negative coefficients = increasing this decreases your score")
        
        else:
            st.warning("Not enough data for regression analysis")
        
        # Section 3: A/B Testing - Weekday vs Weekend
        st.subheader("3. Statistical Testing: Weekday vs Weekend")
        
        weekday_data = df_stats[~df_stats['Is Weekend']]
        weekend_data = df_stats[df_stats['Is Weekend']]
        
        if len(weekday_data) > 0 and len(weekend_data) > 0:
            # T-tests
            duration_ttest = stats.ttest_ind(
                weekday_data['Total Sleep Duration (hours)'].dropna(),
                weekend_data['Total Sleep Duration (hours)'].dropna()
            )
            
            score_ttest = stats.ttest_ind(
                weekday_data['Sleep Score'].dropna(),
                weekend_data['Sleep Score'].dropna()
            )
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Sleep Duration Test:**")
                st.metric("Weekday Average", f"{weekday_data['Total Sleep Duration (hours)'].mean():.2f}h")
                st.metric("Weekend Average", f"{weekend_data['Total Sleep Duration (hours)'].mean():.2f}h")
                st.metric("P-value", f"{duration_ttest.pvalue:.4f}")
                
                if duration_ttest.pvalue < 0.05:
                    st.success("✓ Statistically significant difference (p < 0.05)")
                else:
                    st.info("No statistically significant difference")
            
            with col2:
                st.write("**Sleep Score Test:**")
                st.metric("Weekday Average", f"{weekday_data['Sleep Score'].mean():.1f}")
                st.metric("Weekend Average", f"{weekend_data['Sleep Score'].mean():.1f}")
                st.metric("P-value", f"{score_ttest.pvalue:.4f}")
                
                if score_ttest.pvalue < 0.05:
                    st.success("✓ Statistically significant difference (p < 0.05)")
                else:
                    st.info("No statistically significant difference")
            
            st.markdown("**Statistical Significance Explained:**")
            st.markdown("- **P-value < 0.05**: The difference is statistically significant (not due to random chance)")
            st.markdown("- **P-value ≥ 0.05**: The difference could be due to random variation")
            
            weekday_dur_avg = weekday_data['Total Sleep Duration (hours)'].mean()
            weekend_dur_avg = weekend_data['Total Sleep Duration (hours)'].mean()
            dur_diff = weekend_dur_avg - weekday_dur_avg
            
            st.markdown("**Sleep Pattern Conclusion:**")
            if duration_ttest.pvalue < 0.05:
                if dur_diff > 0:
                    st.markdown(f"- You sleep **significantly more** on weekends (+{dur_diff:.2f}h on average)")
                    st.markdown("  - This suggests you may be accumulating sleep debt during the week")
                else:
                    st.markdown(f"- You sleep **significantly less** on weekends ({dur_diff:.2f}h on average)")
                    st.markdown("  - This is unusual - consider maintaining consistent sleep schedule")
            else:
                st.markdown(f"- Your weekday/weekend sleep duration is **consistent** (difference: {abs(dur_diff):.2f}h)")
                st.markdown("  - Good sleep hygiene - maintaining regular schedule")
        
        # Section 4: Trend Analysis & Moving Averages
        st.subheader("4. Trend Analysis & Moving Averages")
        
        df_trend = df_stats.sort_values('Date').copy()
        df_trend['MA_7'] = df_trend['Sleep Score'].rolling(window=7, min_periods=1).mean()
        df_trend['MA_14'] = df_trend['Sleep Score'].rolling(window=14, min_periods=1).mean()
        
        # Linear trend
        if len(df_trend) > 2:
            X_trend = df_trend['Days Since Start'].values.reshape(-1, 1)
            y_trend = df_trend['Sleep Score'].values
            
            trend_model = LinearRegression()
            trend_model.fit(X_trend, y_trend)
            df_trend['Trend'] = trend_model.predict(X_trend)
            
            slope = trend_model.coef_[0]
            
            fig_trend = go.Figure()
            
            fig_trend.add_trace(go.Scatter(
                x=df_trend['Date'],
                y=df_trend['Sleep Score'],
                mode='markers',
                name='Actual Score',
                marker=dict(size=6, color='lightgray')
            ))
            
            fig_trend.add_trace(go.Scatter(
                x=df_trend['Date'],
                y=df_trend['MA_7'],
                mode='lines',
                name='7-Day MA',
                line=dict(color="#0B7186", width=2)
            ))
            
            fig_trend.add_trace(go.Scatter(
                x=df_trend['Date'],
                y=df_trend['MA_14'],
                mode='lines',
                name='14-Day MA',
                line=dict(color="#A01404", width=2)
            ))
            
            fig_trend.add_trace(go.Scatter(
                x=df_trend['Date'],
                y=df_trend['Trend'],
                mode='lines',
                name='Linear Trend',
                line=dict(color='red', width=2, dash='dash')
            ))
            
            fig_trend.update_layout(
                title='Sleep Score Trends & Moving Averages',
                xaxis_title='Date',
                yaxis_title='Sleep Score',
                height=500,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig_trend, width='stretch')
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Trend Direction", "Improving" if slope > 0 else "Declining")
            with col2:
                st.metric("Daily Change", f"{slope:.3f} points/day")
            with col3:
                monthly_change = slope * 30
                st.metric("Monthly Change", f"{monthly_change:+.1f} points")
            
            st.markdown("**Trend Analysis Insights:**")
            if slope > 0.1:
                st.markdown(f"- **Improving trend**: Your sleep quality is getting better over time (+{slope:.3f} points/day)")
                st.markdown(f"- If this continues, you'll gain **{monthly_change:.1f} points** per month")
            elif slope < -0.1:
                st.markdown(f"- **Declining trend**: Your sleep quality is decreasing over time ({slope:.3f} points/day)")
                st.markdown(f"- This translates to **{monthly_change:.1f} points** loss per month - consider lifestyle adjustments")
            else:
                st.markdown(f"- **Stable trend**: Your sleep quality is relatively consistent (change: {slope:.3f} points/day)")
                st.markdown("- Maintaining your current sleep habits is working well")
            
            st.markdown("**Moving Averages:**")
            st.markdown("- **7-Day MA**: Short-term trends - shows recent sleep pattern changes")
            st.markdown("- **14-Day MA**: Medium-term trends - smooths out weekly variations")
        
        # Section 5: Sleep Debt & Consistency
        st.subheader("5. Sleep Debt & Consistency Metrics")
        
        target_sleep = 7.5  # hours
        df_debt = df_stats.copy()
        df_debt['Sleep Debt'] = target_sleep - df_debt['Total Sleep Duration (hours)']
        df_debt['Cumulative Debt'] = df_debt['Sleep Debt'].cumsum()
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Sleep debt over time
            fig_debt = go.Figure()
            
            fig_debt.add_trace(go.Scatter(
                x=df_debt['Date'],
                y=df_debt['Cumulative Debt'],
                mode='lines+markers',
                fill='tozeroy',
                name='Cumulative Sleep Debt',
                line=dict(color="#7E160A", width=2),
                fillcolor='rgba(231, 76, 60, 0.2)'
            ))
            
            fig_debt.update_layout(
                title='Cumulative Sleep Debt',
                xaxis_title='Date',
                yaxis_title='Hours of Debt',
                height=400
            )
            
            st.plotly_chart(fig_debt, width='stretch')
            
            total_debt = df_debt['Cumulative Debt'].iloc[-1]
            st.metric("Current Sleep Debt", f"{total_debt:+.1f} hours")
        
        with col2:
            # Sleep consistency
            std_duration = df_stats['Total Sleep Duration (hours)'].std()
            mean_duration = df_stats['Total Sleep Duration (hours)'].mean()
            cv = (std_duration / mean_duration) * 100  # Coefficient of variation
            
            consistency_score = max(0, 100 - cv * 10)  # Higher is better
            
            fig_consistency = go.Figure(go.Indicator(
                mode="gauge+number",
                value=consistency_score,
                title={'text': "Sleep Consistency Score"},
                domain={'x': [0, 1], 'y': [0, 1]},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#3498DB"},
                    'steps': [
                        {'range': [0, 50], 'color': "#931103"},
                        {'range': [50, 75], 'color': "#096F88"},
                        {'range': [75, 100], 'color': "#DFE6EE"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 80
                    }
                }
            ))
            
            fig_consistency.update_layout(height=400)
            st.plotly_chart(fig_consistency, width='stretch')
            
            st.metric("Sleep Duration Std Dev", f"{std_duration:.2f} hours")
            st.metric("Coefficient of Variation", f"{cv:.1f}%")
        
        st.markdown("**Sleep Health Interpretation:**")
        st.markdown(f"**Sleep Debt:** {total_debt:+.1f} hours")
        if total_debt > 5:
            st.markdown("- **Significant sleep debt** - you're chronically under-sleeping")
            st.markdown("- Consider going to bed earlier or sleeping in when possible")
        elif total_debt > 0:
            st.markdown("- **Mild sleep debt** - slightly below your target")
            st.markdown("- Try to catch up with extra sleep on recovery days")
        elif total_debt < -5:
            st.markdown("- **Sleep surplus** - consistently getting more than target")
            st.markdown("- Great job maintaining healthy sleep duration!")
        else:
            st.markdown("- **Balanced** - meeting your sleep needs")
        
        st.markdown(f"\n**Consistency Score:** {consistency_score:.0f}/100")
        if consistency_score >= 75:
            st.markdown("- **Excellent consistency** - regular sleep schedule")
            st.markdown("- This promotes better sleep quality and health")
        elif consistency_score >= 50:
            st.markdown("- **Moderate consistency** - some variation in sleep duration")
            st.markdown("- Try to maintain more regular sleep/wake times")
        else:
            st.markdown("- **Poor consistency** - highly variable sleep patterns")
            st.markdown("- Irregular sleep can impact sleep quality - aim for more consistency")
        
        # Section 6: Distribution & Outliers
        st.subheader("6. Distribution Analysis & Outlier Detection")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Distribution with normal curve
            fig_dist = go.Figure()
            
            # Histogram
            fig_dist.add_trace(go.Histogram(
                x=df_stats['Sleep Score'],
                name='Sleep Score Distribution',
                nbinsx=15,
                marker_color="#21ACCB",
                opacity=0.7
            ))
            
            # Normal distribution overlay
            mean_score = df_stats['Sleep Score'].mean()
            std_score = df_stats['Sleep Score'].std()
            x_norm = np.linspace(df_stats['Sleep Score'].min(), df_stats['Sleep Score'].max(), 100)
            y_norm = stats.norm.pdf(x_norm, mean_score, std_score) * len(df_stats) * (df_stats['Sleep Score'].max() - df_stats['Sleep Score'].min()) / 15
            
            fig_dist.add_trace(go.Scatter(
                x=x_norm,
                y=y_norm,
                mode='lines',
                name='Normal Distribution',
                line=dict(color='red', width=2)
            ))
            
            fig_dist.update_layout(
                title='Sleep Score Distribution vs Normal',
                xaxis_title='Sleep Score',
                yaxis_title='Frequency',
                height=400
            )
            
            st.plotly_chart(fig_dist, width='stretch')
            
            # Normality test
            shapiro_stat, shapiro_p = stats.shapiro(df_stats['Sleep Score'])
            st.metric("Shapiro-Wilk p-value", f"{shapiro_p:.4f}")
            if shapiro_p > 0.05:
                st.info("Distribution appears normal (p > 0.05)")
            else:
                st.warning("Distribution is not normal (p < 0.05)")
        
        with col2:
            # Box plot with outliers
            fig_box = go.Figure()
            
            fig_box.add_trace(go.Box(
                y=df_stats['Sleep Score'],
                name='Sleep Score',
                boxmean='sd',
                marker_color="#28B8D5"
            ))
            
            fig_box.update_layout(
                title='Box Plot - Outlier Detection',
                yaxis_title='Sleep Score',
                height=400,
                showlegend=False
            )
            
            st.plotly_chart(fig_box, width='stretch')
            
            # Calculate percentiles
            q1 = df_stats['Sleep Score'].quantile(0.25)
            q3 = df_stats['Sleep Score'].quantile(0.75)
            iqr = q3 - q1
            
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            outliers = df_stats[(df_stats['Sleep Score'] < lower_bound) | (df_stats['Sleep Score'] > upper_bound)]
            
            st.metric("Number of Outliers", len(outliers))
            st.metric("IQR", f"{iqr:.1f}")
        
        st.markdown("**Distribution Analysis:**")
        st.markdown(f"- **Normality Test (Shapiro-Wilk p={shapiro_p:.4f})**:")
        if shapiro_p > 0.05:
            st.markdown("  - Your sleep scores follow a normal distribution")
            st.markdown("  - Most nights cluster around your average - predictable sleep pattern")
        else:
            st.markdown("  - Your sleep scores don't follow a normal distribution")
            st.markdown("  - You have more extreme variation or skewed patterns")
        
        st.markdown(f"- **Outliers:** {len(outliers)} night(s) detected")
        if len(outliers) > 0:
            st.markdown(f"  - These are unusually good or poor sleep nights")
            st.markdown(f"  - Review these dates to identify special circumstances")
        else:
            st.markdown("  - No extreme outliers - consistent sleep quality")
        
        # Section 7: Lag Correlation
        st.subheader("7. Lag Correlation - Does Yesterday's Sleep Affect Today?")
        
        df_lag = df_stats.sort_values('Date').copy()
        df_lag['Score_Lag1'] = df_lag['Sleep Score'].shift(1)
        df_lag['Duration_Lag1'] = df_lag['Total Sleep Duration (hours)'].shift(1)
        
        lag_corr_score = df_lag[['Sleep Score', 'Score_Lag1']].corr().iloc[0, 1]
        lag_corr_duration = df_lag[['Total Sleep Duration (hours)', 'Duration_Lag1']].corr().iloc[0, 1]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Score Autocorrelation (1-day lag)", f"{lag_corr_score:.3f}")
            if abs(lag_corr_score) > 0.3:
                st.info("Moderate to strong correlation - yesterday's score affects today")
            else:
                st.info("Weak correlation - each night is independent")
        
        with col2:
            st.metric("Duration Autocorrelation (1-day lag)", f"{lag_corr_duration:.3f}")
            if abs(lag_corr_duration) > 0.3:
                st.info("Your sleep duration shows a pattern")
            else:
                st.info("Sleep duration varies independently")
        
        st.markdown("**Autocorrelation Insights:**")
        st.markdown("- **Autocorrelation** measures if yesterday's sleep predicts today's sleep")
        
        if abs(lag_corr_score) > 0.5:
            st.markdown(f"  - **Strong dependency**: Sleep Score (r={lag_corr_score:.3f})")
            st.markdown("  - Good sleep yesterday → likely good sleep today")
            st.markdown("  - Poor sleep yesterday → likely poor sleep today")
            st.markdown("  - Your sleep quality has strong momentum/carry-over effects")
        elif abs(lag_corr_score) > 0.3:
            st.markdown(f"  - **Moderate dependency**: Sleep Score (r={lag_corr_score:.3f})")
            st.markdown("  - Yesterday's sleep has some influence on today")
        else:
            st.markdown(f"  - **Independent nights**: Sleep Score (r={lag_corr_score:.3f})")
            st.markdown("  - Each night is independent - good recovery ability")
            st.markdown("  - A bad night doesn't doom the next one")
        
        # Section 8: Variance Analysis
        st.subheader("8. Variance Analysis - Sleep Stability Over Time")
        
        # Calculate rolling variance
        df_variance = df_stats.sort_values('Date').copy()
        df_variance['Rolling Variance (7d)'] = df_variance['Sleep Score'].rolling(window=7, min_periods=3).var()
        df_variance['Rolling Std (7d)'] = df_variance['Sleep Score'].rolling(window=7, min_periods=3).std()
        df_variance['Rolling CV (7d)'] = (df_variance['Rolling Std (7d)'] / df_variance['Sleep Score'].rolling(window=7, min_periods=3).mean()) * 100
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Variance over time
            fig_var = go.Figure()
            
            fig_var.add_trace(go.Scatter(
                x=df_variance['Date'],
                y=df_variance['Rolling Variance (7d)'],
                mode='lines',
                name='7-Day Rolling Variance',
                line=dict(color="#991B0D", width=2),
                fill='tozeroy',
                fillcolor='rgba(231, 76, 60, 0.2)'
            ))
            
            fig_var.update_layout(
                title='Sleep Score Variance Over Time (7-Day Window)',
                xaxis_title='Date',
                yaxis_title='Variance',
                template='plotly_white',
                height=400,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig_var, width='stretch')
        
        with col2:
            # Standard deviation over time
            fig_std = go.Figure()
            
            fig_std.add_trace(go.Scatter(
                x=df_variance['Date'],
                y=df_variance['Rolling Std (7d)'],
                mode='lines',
                name='7-Day Rolling Std Dev',
                line=dict(color="#1EABCE", width=2),
                fill='tozeroy',
                fillcolor='rgba(52, 152, 219, 0.2)'
            ))
            
            fig_std.update_layout(
                title='Sleep Score Standard Deviation (7-Day Window)',
                xaxis_title='Date',
                yaxis_title='Standard Deviation',
                template='plotly_white',
                height=400,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig_std, width='stretch')
        
        # Coefficient of variation
        st.subheader("Coefficient of Variation Over Time")
        
        fig_cv = go.Figure()
        
        fig_cv.add_trace(go.Scatter(
            x=df_variance['Date'],
            y=df_variance['Rolling CV (7d)'],
            mode='lines',
            name='7-Day Rolling CV',
            line=dict(color="#039598", width=2),
            fill='tozeroy',
            fillcolor='rgba(0, 204, 204, 0.2)'
        ))
        
        # Add threshold line for stable sleep
        fig_cv.add_hline(
            y=10,
            line_dash="dash",
            line_color="green",
            annotation_text="Stable Sleep Threshold (CV < 10%)"
        )
        
        fig_cv.update_layout(
            title='Coefficient of Variation - Relative Sleep Stability',
            xaxis_title='Date',
            yaxis_title='Coefficient of Variation (%)',
            template='plotly_white',
            height=400,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig_cv, width='stretch')
        
        # Statistics summary
        col1, col2, col3 = st.columns(3)
        
        with col1:
            overall_variance = df_stats['Sleep Score'].var()
            st.metric("Overall Variance", f"{overall_variance:.2f}")
        
        with col2:
            overall_std = df_stats['Sleep Score'].std()
            st.metric("Overall Std Dev", f"{overall_std:.2f}")
        
        with col3:
            overall_cv = (overall_std / df_stats['Sleep Score'].mean()) * 100
            st.metric("Overall CV", f"{overall_cv:.1f}%")
        
        st.markdown("**Variance Analysis Insights:**")
        st.markdown("- **Variance** measures how spread out your sleep scores are from the average")
        st.markdown("- **Standard Deviation** is the square root of variance (same concept, different scale)")
        st.markdown("- **Coefficient of Variation (CV)** is standardized - allows comparison regardless of average score")
        
        st.markdown("\n**Connection to Sleep Consistency:**")
        st.markdown(f"- Your **Overall CV ({overall_cv:.1f}%)** directly determines your **Sleep Consistency Score** from Section 5")
        st.markdown(f"- Sleep Consistency Score = {100 - overall_cv * 10:.0f}/100 (inverse relationship)")
        st.markdown("- **Lower variance/CV** = **Higher consistency** = More predictable sleep")
        st.markdown("- **Higher variance/CV** = **Lower consistency** = More unpredictable sleep")
        
        st.markdown("\n**Interpretation:**")
        
        if overall_cv < 10:
            st.markdown(f"- **Very stable sleep** (CV = {overall_cv:.1f}%)")
            st.markdown("  - Your sleep quality is highly consistent")
            st.markdown("  - Low night-to-night variation")
        elif overall_cv < 15:
            st.markdown(f"- **Moderately stable sleep** (CV = {overall_cv:.1f}%)")
            st.markdown("  - Some variation but generally predictable")
            st.markdown("  - Normal variability for most people")
        else:
            st.markdown(f"- **Variable sleep** (CV = {overall_cv:.1f}%)")
            st.markdown("  - High night-to-night variation in sleep quality")
            st.markdown("  - Consider identifying factors causing variability")
        
        # Identify periods of high/low variance
        high_var_periods = df_variance[df_variance['Rolling Variance (7d)'] > df_variance['Rolling Variance (7d)'].quantile(0.75)].dropna()
        low_var_periods = df_variance[df_variance['Rolling Variance (7d)'] < df_variance['Rolling Variance (7d)'].quantile(0.25)].dropna()
        
        if len(high_var_periods) > 0:
            st.markdown(f"\n**Periods of High Variability:** {len(high_var_periods)} periods detected")
            st.markdown("  - These are times when your sleep was most unpredictable")
            st.markdown("  - Review these dates for potential stress, travel, or schedule changes")
        
        if len(low_var_periods) > 0:
            st.markdown(f"\n**Periods of Low Variability:** {len(low_var_periods)} periods detected")
            st.markdown("  - These are your most stable sleep periods")
            st.markdown("  - Identify what you did consistently during these times")
    
    # Footer
    st.markdown("---")
    st.markdown(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    st.markdown("Data source: Garmin Connect")

except FileNotFoundError:
    st.error("Sleep data file not found. Please run sleep.py first to extract data from Garmin Connect.")
except Exception as e:
    st.error(f"An error occurred: {str(e)}")
    st.exception(e)
