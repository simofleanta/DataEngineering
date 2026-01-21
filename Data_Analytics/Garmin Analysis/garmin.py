import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path
from datetime import datetime
import numpy as np
import subprocess
import sys
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Configurare pagină
st.set_page_config(
    page_title="Garmin Activity Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title
st.title("Garmin Activity Dashboard")
st.markdown("---")

# load data
@st.cache_data
def load_data():
    garmin_folder = Path(__file__).parent
    excel_file = garmin_folder / 'activitati_garmin.xlsx'
    
    if not excel_file.exists():
        st.error(f"Fisierul {excel_file} no such file! Run garminpy.py")
        return None
    
    df = pd.read_csv(garmin_folder / 'activitati_garmin.csv')
    df['Data'] = pd.to_datetime(df['Data'])
    df = df.sort_values('Data')
    return df

df = load_data()

if df is not None:
    # Sidebar - Filtre
    st.sidebar.header("Filtre")
    
    # Buton pentru sincronizare cu Garmin Connect
    st.sidebar.subheader("Sync from Garmin")
    if st.sidebar.button("Download New Activities", use_container_width=True):
        with st.spinner("Connecting to Garmin Connect..."):
            try:
                # Get Python executable path
                python_path = sys.executable
                garminpy_path = Path(__file__).parent / "garminpy.py"
                
                # Run garminpy.py
                result = subprocess.run(
                    [python_path, str(garminpy_path)],
                    capture_output=True,
                    text=True,
                    timeout=60
                )
                
                if result.returncode == 0:
                    st.sidebar.success("New activities downloaded!")
                    # Clear cache and reload
                    st.cache_data.clear()
                    st.rerun()
                else:
                    st.sidebar.error(f"Error: {result.stderr}")
            except subprocess.TimeoutExpired:
                st.sidebar.error("Timeout: Garmin sync took too long")
            except Exception as e:
                st.sidebar.error(f"Error: {str(e)}")
    
    st.sidebar.divider()
    
    # Filtrare după tip activitate
    tipuri = ['Toate'] + list(df['Tip'].unique())
    tip_selectat = st.sidebar.selectbox("Tip Activitate", tipuri)
    
    # Filtrare per data
    data_min = df['Data'].min().date()
    data_max = df['Data'].max().date()
    
    date_range = st.sidebar.date_input(
        "Interval Date",
        value=(data_min, data_max),
        min_value=data_min,
        max_value=data_max
    )
    
    # Aplicare filtre
    df_filtered = df.copy()
    
    if tip_selectat != 'Toate':
        df_filtered = df_filtered[df_filtered['Tip'] == tip_selectat]
    
    if len(date_range) == 2:
        df_filtered = df_filtered[
            (df_filtered['Data'].dt.date >= date_range[0]) &
            (df_filtered['Data'].dt.date <= date_range[1])
        ]
    
    # Stats
    st.header("General Stats")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            label="Total Activities",
            value=len(df_filtered),
            delta=f"{len(df_filtered) - len(df)} filtered" if len(df_filtered) != len(df) else None
        )
    
    with col2:
        total_calorii = df_filtered['Calorii'].sum()
        st.metric(
            label="Calories Burned",
            value=f"{total_calorii:,.0f}",
            delta=f"{total_calorii - df['Calorii'].sum():,.0f}" if len(df_filtered) != len(df) else None
        )
    
    with col3:
        total_distanta = df_filtered['Distanță (km)'].sum()
        st.metric(
            label="Distance (km)",
            value=f"{total_distanta:.1f}",
            delta=f"{total_distanta - df['Distanță (km)'].sum():.1f}" if len(df_filtered) != len(df) else None
        )
    
    with col4:
        total_timp = df_filtered['Durată (min)'].sum()
        st.metric(
            label="Total Time (hours)",
            value=f"{total_timp/60:.1f}",
            delta=f"{(total_timp - df['Durată (min)'].sum())/60:.1f}" if len(df_filtered) != len(df) else None
        )
    
    with col5:
        medie_calorii = df_filtered['Calorii'].mean()
        st.metric(
            label="Average Calories",
            value=f"{medie_calorii:.0f}"
        )
    
    st.markdown("---")
    
    # Row 1: Calories and Distance
    st.header("Calories and Distance")
    col1, col2 = st.columns(2)
    
    with col1:
        # Calories chart
        fig_calorii = go.Figure()
        
        fig_calorii.add_trace(go.Bar(
            x=df_filtered['Data'],
            y=df_filtered['Calorii'],
            name='Calories',
            marker_color='rgb(25, 106, 181)',
            text=df_filtered['Calorii'],
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>Calories: %{y}<extra></extra>'
        ))
        
        # Moving average
        df_filtered['Calorii_MA'] = df_filtered['Calorii'].rolling(window=3, min_periods=1).mean()
        fig_calorii.add_trace(go.Scatter(
            x=df_filtered['Data'],
            y=df_filtered['Calorii_MA'],
            mode='lines',
            name='Moving Average',
            line=dict(color='gold', width=2, dash='dash')
        ))
        
        fig_calorii.update_layout(
            title='Calories Burned per Activity',
            xaxis_title='Date',
            yaxis_title='Calories',
            hovermode='x unified',
            height=400,
            showlegend=True
        )
        
        st.plotly_chart(fig_calorii, width='stretch')
    
    with col2:
        # Grafic distanță
        fig_distanta = go.Figure()
        
        fig_distanta.add_trace(go.Bar(
            x=df_filtered['Data'],
            y=df_filtered['Distanță (km)'],
            name='Distance',
            marker_color='rgb(44, 119, 160)',
            text=df_filtered['Distanță (km)'],
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>Distance: %{y} km<extra></extra>'
        ))
        
        fig_distanta.update_layout(
            title='Distance per Activity',
            xaxis_title='Date',
            yaxis_title='Distance (km)',
            hovermode='x unified',
            height=400
        )
        
        st.plotly_chart(fig_distanta, width='stretch')
    
    st.markdown("---")
    
    # Row 2: Scatter and Pie
    st.header("Further Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        # Scatter: Duration vs Calories
        fig_scatter = go.Figure()
        
        fig_scatter.add_trace(go.Scatter(
            x=df_filtered['Durată (min)'],
            y=df_filtered['Calorii'],
            mode='markers',
            marker=dict(
                size=df_filtered['Distanță (km)'] * 5,
                color=df_filtered['Calorii'],
                colorscale='Blues',
                showscale=True,
                colorbar=dict(title="Calories")
            ),
            text=df_filtered['Tip'],
            hovertemplate='<b>%{text}</b><br>Duration: %{x} min<br>Calories: %{y}<extra></extra>'
        ))
        
        fig_scatter.update_layout(
            title='Duration vs Calories (size = distance)',
            xaxis_title='Duration (min)',
            yaxis_title='Calories',
            height=400
        )
        
        st.plotly_chart(fig_scatter, width='stretch')
    
    with col2:
        # Pie chart - Distribution of Types
        type_counts = df_filtered['Tip'].value_counts()
        
        fig_pie = go.Figure(data=[go.Pie(
            labels=type_counts.index,
            values=type_counts.values,
            hole=0.3,
            textinfo='label+percent',
            hovertemplate='<b>%{label}</b><br>Activities: %{value}<br>%{percent}<extra></extra>',
            marker=dict(colors=px.colors.qualitative.Pastel)
        )])
        
        fig_pie.update_layout(
            title='Distribution of Activity Types',
            height=400
        )
        
        st.plotly_chart(fig_pie, width='stretch')
    
    st.markdown("---")
    
    # Row 3: Heart Rate and Altitude
    st.header("Heart Rate and Altitude")
    col1, col2 = st.columns(2)
    
    with col1:
        # Heart Rate
        df_heart = df_filtered[df_filtered['Ritm cardiac mediu'] > 0]
        
        if not df_heart.empty:
            fig_heart = go.Figure()
            
            fig_heart.add_trace(go.Scatter(
                x=df_heart['Data'],
                y=df_heart['Ritm cardiac mediu'],
                mode='lines+markers',
                line=dict(color='steelblue', width=2),
                marker=dict(size=8),
                fill='tozeroy',
                fillcolor='rgba(112, 194, 224, 0.3 )',
                hovertemplate='<b>%{x}</b><br>BPM: %{y}<extra></extra>'
            ))
            
            fig_heart.update_layout(
                title='Average Heart Rate Over Time',
                xaxis_title='Date',
                yaxis_title='BPM',
                height=400
            )
            
            st.plotly_chart(fig_heart, width='stretch')
    
    with col2:
        # Altitude
        df_alt = df_filtered[df_filtered['Altitudine max (m)'] > 0]
        
        if not df_alt.empty:
            fig_alt = go.Figure()
            
            fig_alt.add_trace(go.Bar(
                x=df_alt['Data'],
                y=df_alt['Altitudine max (m)'],
                marker_color='rgb(148, 103, 189)',
                hovertemplate='<b>%{x}</b><br>Altitude: %{y} m<extra></extra>'
            ))
            
            fig_alt.update_layout(
                title='Max Altitude',
                xaxis_title='Date',
                yaxis_title='Altitude (m)',
                height=400
            )
            
            st.plotly_chart(fig_alt, width='stretch')
    
    st.markdown("---")
    
    # Swim Comparison Section
    df_swims = df[df['Tip'] == 'lap_swimming'].copy()
    
    if not df_swims.empty:
        st.header("Swim Performance Comparison")
        
        # Calculate swim metrics
        df_swims['Pace (min/km)'] = df_swims['Durată (min)'] / df_swims['Distanță (km)']
        df_swims['Calorii/km'] = df_swims['Calorii'] / df_swims['Distanță (km)']
        
        # Swim stats overview
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Total Swims",
                value=len(df_swims)
            )
        
        with col2:
            total_swim_distance = df_swims['Distanță (km)'].sum()
            st.metric(
                label="Total Distance (km)",
                value=f"{total_swim_distance:.2f}"
            )
        
        with col3:
            avg_pace = df_swims['Pace (min/km)'].mean()
            st.metric(
                label="Avg Pace (min/km)",
                value=f"{avg_pace:.2f}"
            )
        
        with col4:
            best_pace = df_swims['Pace (min/km)'].min()
            st.metric(
                label="Best Pace (min/km)",
                value=f"{best_pace:.2f}"
            )
        
        # Swim comparison charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Distance vs Pace scatter
            fig_swim_pace = go.Figure()
            
            fig_swim_pace.add_trace(go.Scatter(
                x=df_swims['Distanță (km)'],
                y=df_swims['Pace (min/km)'],
                mode='markers+text',
                marker=dict(
                    size=12,
                    color=df_swims['Calorii'],
                    colorscale='Blues',
                    showscale=True,
                    colorbar=dict(title="Calories")
                ),
                text=df_swims['Pace (min/km)'].round(2),
                textposition='top center',
                hovertemplate='<b>Date: %{customdata}</b><br>Distance: %{x:.2f} km<br>Pace: %{y:.2f} min/km<extra></extra>',
                customdata=df_swims['Data'].dt.strftime('%d/%m/%Y')
            ))
            
            fig_swim_pace.update_layout(
                title='Swim Distance vs Pace',
                xaxis_title='Distance (km)',
                yaxis_title='Pace (min/km)',
                height=400
            )
            
            st.plotly_chart(fig_swim_pace, width='stretch')
        
        with col2:
            # Progress over time
            fig_swim_progress = go.Figure()
            
            fig_swim_progress.add_trace(go.Scatter(
                x=df_swims['Data'],
                y=df_swims['Pace (min/km)'],
                mode='lines+markers',
                name='Pace',
                line=dict(color='steelblue', width=2),
                marker=dict(size=8),
                hovertemplate='<b>%{x}</b><br>Pace: %{y:.2f} min/km<extra></extra>'
            ))
            
            # Add trend line
            z = np.polyfit(range(len(df_swims)), df_swims['Pace (min/km)'], 1)
            p = np.poly1d(z)
            fig_swim_progress.add_trace(go.Scatter(
                x=df_swims['Data'],
                y=p(range(len(df_swims))),
                mode='lines',
                name='Trend',
                line=dict(color='darkred', dash='dash', width=2)
            ))
            
            fig_swim_progress.update_layout(
                title='Swim Pace Progress Over Time',
                xaxis_title='Date',
                yaxis_title='Pace (min/km)',
                height=400,
                showlegend=True
            )
            
            st.plotly_chart(fig_swim_progress, width='stretch')
        
        st.markdown("---")
    
    # Cycle Comparison Section
    df_cycles = df[df['Tip'] == 'cycling'].copy()
    
    if not df_cycles.empty:
        st.header("Cycle Performance Comparison")
        
        # Calculate cycle metrics
        df_cycles['Speed (km/h)'] = (df_cycles['Distanță (km)'] / df_cycles['Durată (min)']) * 60
        df_cycles['Calorii/km'] = df_cycles['Calorii'] / df_cycles['Distanță (km)']
        
        # Cycle stats overview
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Total Rides",
                value=len(df_cycles)
            )
        
        with col2:
            total_cycle_distance = df_cycles['Distanță (km)'].sum()
            st.metric(
                label="Total Distance (km)",
                value=f"{total_cycle_distance:.2f}"
            )
        
        with col3:
            avg_speed = df_cycles['Speed (km/h)'].mean()
            st.metric(
                label="Avg Speed (km/h)",
                value=f"{avg_speed:.2f}"
            )
        
        with col4:
            max_speed = df_cycles['Speed (km/h)'].max()
            st.metric(
                label="Max Speed (km/h)",
                value=f"{max_speed:.2f}"
            )
        
        # Cycle comparison charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Distance vs Speed scatter
            fig_cycle_speed = go.Figure()
            
            fig_cycle_speed.add_trace(go.Scatter(
                x=df_cycles['Distanță (km)'],
                y=df_cycles['Speed (km/h)'],
                mode='markers+text',
                marker=dict(
                    size=12,
                    color=df_cycles['Calorii'],
                    colorscale='Greens',
                    showscale=True,
                    colorbar=dict(title="Calories")
                ),
                text=df_cycles['Speed (km/h)'].round(1),
                textposition='top center',
                hovertemplate='<b>Date: %{customdata}</b><br>Distance: %{x:.2f} km<br>Speed: %{y:.2f} km/h<extra></extra>',
                customdata=df_cycles['Data'].dt.strftime('%d/%m/%Y')
            ))
            
            fig_cycle_speed.update_layout(
                title='Cycling Distance vs Speed',
                xaxis_title='Distance (km)',
                yaxis_title='Speed (km/h)',
                height=400
            )
            
            st.plotly_chart(fig_cycle_speed, width='stretch')
        
        with col2:
            # Speed progress over time
            fig_cycle_progress = go.Figure()
            
            fig_cycle_progress.add_trace(go.Scatter(
                x=df_cycles['Data'],
                y=df_cycles['Speed (km/h)'],
                mode='lines+markers',
                name='Speed',
                line=dict(color='green', width=2),
                marker=dict(size=8),
                hovertemplate='<b>%{x}</b><br>Speed: %{y:.2f} km/h<extra></extra>'
            ))
            
            # Add trend line
            z = np.polyfit(range(len(df_cycles)), df_cycles['Speed (km/h)'], 1)
            p = np.poly1d(z)
            fig_cycle_progress.add_trace(go.Scatter(
                x=df_cycles['Data'],
                y=p(range(len(df_cycles))),
                mode='lines',
                name='Trend',
                line=dict(color='darkred', dash='dash', width=2)
            ))
            
            fig_cycle_progress.update_layout(
                title='Cycling Speed Progress Over Time',
                xaxis_title='Date',
                yaxis_title='Speed (km/h)',
                height=400,
                showlegend=True
            )
            
            st.plotly_chart(fig_cycle_progress, width='stretch')
        
        st.markdown("---")
    
    # Machine Learning: Calorie Prediction
    st.header("Machine Learning: Calorie Prediction")
    
    st.markdown("""
    This section uses **Linear Regression** to predict calories burned based on:
    - Distance (km)
    - Duration (min) 
    - Average Heart Rate (BPM)
    """)
    
    # Prepare data for ML
    df_ml = df_filtered[['Distanță (km)', 'Durată (min)', 'Ritm cardiac mediu', 'Calorii']].copy()
    df_ml = df_ml[df_ml['Ritm cardiac mediu'] > 0]
    df_ml = df_ml.dropna()
    
    if len(df_ml) >= 15:
        # Features and target
        X = df_ml[['Distanță (km)', 'Durată (min)', 'Ritm cardiac mediu']].values
        y = df_ml['Calorii'].values
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train Linear Regression Model
        lr_model = LinearRegression()
        lr_model.fit(X_train, y_train)
        lr_pred_train = lr_model.predict(X_train)
        lr_pred_test = lr_model.predict(X_test)
        
        lr_r2_train = r2_score(y_train, lr_pred_train)
        lr_r2_test = r2_score(y_test, lr_pred_test)
        lr_mae = mean_absolute_error(y_test, lr_pred_test)
        lr_rmse = np.sqrt(mean_squared_error(y_test, lr_pred_test))
        
        # Model Performance
        st.subheader("Model Performance")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("R² Score (Train)", f"{lr_r2_train:.3f}")
        with col2:
            st.metric("R² Score (Test)", f"{lr_r2_test:.3f}")
        with col3:
            st.metric("MAE (Test)", f"{lr_mae:.1f} cal")
        with col4:
            st.metric("RMSE (Test)", f"{lr_rmse:.1f} cal")
        
        st.markdown("---")
        
        # Prediction Quality and Analysis
        st.subheader("Model Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Actual vs Predicted
            fig_lr = go.Figure()
            
            fig_lr.add_trace(go.Scatter(
                x=y_test,
                y=lr_pred_test,
                mode='markers',
                marker=dict(size=10, color='steelblue', opacity=0.7),
                name='Predictions',
                hovertemplate='Actual: %{x:.0f}<br>Predicted: %{y:.0f}<extra></extra>'
            ))
            
            min_val = min(y_test.min(), lr_pred_test.min())
            max_val = max(y_test.max(), lr_pred_test.max())
            fig_lr.add_trace(go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                line=dict(color='red', dash='dash', width=2),
                name='Perfect Prediction'
            ))
            
            fig_lr.update_layout(
                title=f'Actual vs Predicted (R²={lr_r2_test:.3f})',
                xaxis_title='Actual Calories',
                yaxis_title='Predicted Calories',
                height=400,
                showlegend=True
            )
            
            st.plotly_chart(fig_lr, use_container_width=True)
        
        with col2:
            # Feature Importance - Horizontal Bar Chart
            feature_names = ['Distance (km)', 'Duration (min)', 'Heart Rate (BPM)']
            coefficients = lr_model.coef_
            
            # Normalize coefficients for color mapping (0 to max)
            max_coef = max(abs(coefficients))
            normalized_coefs = [c / max_coef if max_coef > 0 else 0 for c in coefficients]
            
            fig_coef = go.Figure()
            
            fig_coef.add_trace(go.Bar(
                y=feature_names,
                x=coefficients,
                orientation='h',
                marker=dict(
                    color=coefficients,
                    colorscale='Reds',
                    showscale=True,
                    colorbar=dict(title="Coefficient")
                ),
                text=[f'{c:.2f}' for c in coefficients],
                textposition='outside',
                hovertemplate='<b>%{y}</b><br>Coefficient: %{x:.2f}<extra></extra>'
            ))
            
            fig_coef.update_layout(
                title='Feature Impact on Calorie Prediction',
                xaxis_title='Coefficient',
                yaxis_title='Feature',
                height=400,
                showlegend=False
            )
            
            st.plotly_chart(fig_coef, use_container_width=True)
        
        # Residuals Plot
        st.subheader("Residual Analysis")
        
        residuals = y_test - lr_pred_test
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Residuals vs Predicted
            fig_res = go.Figure()
            
            fig_res.add_trace(go.Scatter(
                x=lr_pred_test,
                y=residuals,
                mode='markers',
                marker=dict(size=8, color='coral', opacity=0.7),
                hovertemplate='Predicted: %{x:.0f}<br>Residual: %{y:.0f}<extra></extra>'
            ))
            
            # Add zero line
            fig_res.add_hline(y=0, line_dash="dash", line_color="red", line_width=2)
            
            fig_res.update_layout(
                title='Residuals vs Predicted Values',
                xaxis_title='Predicted Calories',
                yaxis_title='Residuals (Actual - Predicted)',
                height=400
            )
            
            st.plotly_chart(fig_res, use_container_width=True)
        
        with col2:
            # Residuals Distribution
            fig_hist = go.Figure()
            
            fig_hist.add_trace(go.Histogram(
                x=residuals,
                nbinsx=15,
                marker=dict(color='lightblue', line=dict(color='darkblue', width=1)),
                name='Residuals'
            ))
            
            fig_hist.update_layout(
                title='Residuals Distribution',
                xaxis_title='Residuals',
                yaxis_title='Frequency',
                height=400,
                showlegend=False
            )
            
            st.plotly_chart(fig_hist, use_container_width=True)
        
        st.markdown("---")
        
        # Interactive Prediction
        st.subheader("Predict Your Calories")
        st.markdown("Use the sliders below to estimate calories burned:")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            pred_distance = st.slider("Distance (km)", 0.5, 25.0, 5.0, 0.5)
        with col2:
            pred_duration = st.slider("Duration (min)", 10, 200, 60, 5)
        with col3:
            pred_hr = st.slider("Avg Heart Rate (BPM)", 60, 180, 120, 5)
        
        # Make prediction
        custom_input = np.array([[pred_distance, pred_duration, pred_hr]])
        lr_custom_pred = lr_model.predict(custom_input)[0]
        
        # Display prediction
        st.markdown(f"### Estimated Calories: **{lr_custom_pred:.0f} kcal**")
        
        # Model insights
        with st.expander("Model Insights"):
            st.markdown(f"""
            **Linear Regression Coefficients:**
            - Intercept: {lr_model.intercept_:.2f}
            - Distance coefficient: {lr_model.coef_[0]:.2f} (cal/km)
            - Duration coefficient: {lr_model.coef_[1]:.2f} (cal/min)
            - Heart Rate coefficient: {lr_model.coef_[2]:.2f} (cal/BPM)
            
            **Interpretation:**
            - Each km adds ~{lr_model.coef_[0]:.1f} calories
            - Each minute adds ~{lr_model.coef_[1]:.1f} calories
            - Each BPM adds ~{lr_model.coef_[2]:.1f} calories
            """)
        
    else:
        st.warning("Not enough data for machine learning. Need at least 15 activities with heart rate data.")
    
    st.markdown("---")
    
    # Conclusions
    st.header("Key Insights and Conclusions")
    
    # Calculate summary statistics
    total_activities = len(df_filtered)
    total_distance = df_filtered['Distanță (km)'].sum()
    total_duration = df_filtered['Durată (min)'].sum()
    total_calories = df_filtered['Calorii'].sum()
    avg_hr = df_filtered[df_filtered['Ritm cardiac mediu'] > 0]['Ritm cardiac mediu'].mean()
    
    # Activity breakdown
    activity_counts = df_filtered['Tip'].value_counts()
    most_common_activity = activity_counts.index[0] if len(activity_counts) > 0 else "N/A"
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Performance Summary")
        st.markdown(f"""
        **Overall Statistics:**
        - Total Activities Analyzed: **{total_activities}**
        - Total Distance Covered: **{total_distance:.2f} km**
        - Total Time Spent: **{total_duration:.0f} minutes** ({total_duration/60:.1f} hours)
        - Total Calories Burned: **{total_calories:.0f} kcal**
        - Average Heart Rate: **{avg_hr:.0f} BPM**
        
        **Activity Distribution:**
        - Most Frequent Activity: **{most_common_activity}**
        - Activity Types: **{len(activity_counts)}**
        """)
    
    with col2:
        st.subheader("Machine Learning Insights")
        if len(df_ml) >= 15:
            st.markdown(f"""
            **Model Performance:**
            - Algorithm: **Linear Regression**
            - Test R² Score: **{lr_r2_test:.3f}**
            - Mean Absolute Error: **{lr_mae:.1f} calories**
            - Model Accuracy: **{lr_r2_test*100:.1f}%**
            
            **Key Predictors:**
            - Distance contributes **{abs(lr_model.coef_[0]):.1f} cal/km**
            - Duration contributes **{abs(lr_model.coef_[1]):.1f} cal/min**
            - Heart Rate contributes **{abs(lr_model.coef_[2]):.1f} cal/BPM**
            
            **Interpretation:**
            The model successfully predicts calorie expenditure with reasonable accuracy. 
            Heart rate and duration are strong indicators of energy expenditure.
            """)
        else:
            st.markdown("""
            **Data Requirements:**
            - Insufficient data for machine learning analysis
            - Minimum required: 15 activities with heart rate data
            - Current available: Less than required threshold
            
            **Recommendation:**
            Continue tracking activities to enable predictive modeling and deeper insights.
            """)
