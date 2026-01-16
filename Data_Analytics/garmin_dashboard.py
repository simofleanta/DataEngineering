import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path
from datetime import datetime
import numpy as np

# Configurare pagină
st.set_page_config(
    page_title="Garmin Activity Dashboard",
    page_icon="",
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
        else:
            st.info("No heart rate data available for the filtered activities")
    
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
        else:
            st.info("No altitude data available for the filtered activities")
    
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
        
        # Detailed swim comparison table
        st.subheader("Swim Comparison Table")
        
        df_swims_display = df_swims[['Data', 'Nume', 'Distanță (km)', 'Durată (min)', 
                                      'Pace (min/km)', 'Calorii', 'Calorii/km', 
                                      'Ritm cardiac mediu']].copy()
        df_swims_display['Data'] = df_swims_display['Data'].dt.strftime('%Y-%m-%d')
        df_swims_display = df_swims_display.sort_values('Data', ascending=False)
        
        # Format numeric columns
        df_swims_display['Pace (min/km)'] = df_swims_display['Pace (min/km)'].round(2)
        df_swims_display['Calorii/km'] = df_swims_display['Calorii/km'].round(0)
        
        st.dataframe(
            df_swims_display,
            width='stretch',
            hide_index=True
        )
        
        st.markdown("---")
    
    # Data table
    st.header("Activity Details")
    
    # Sorting option
    col1, col2 = st.columns([1, 3])
    with col1:
        sort_by = st.selectbox(
            "Sort by:",
            ['Data', 'Calorii', 'Distanță (km)', 'Durată (min)', 'Tip']
        )
    
    with col2:
        sort_order = st.radio("Order:", ['Descending', 'Ascending'], horizontal=True)
    
    # Apply sorting
    df_display = df_filtered.sort_values(
        by=sort_by,
        ascending=(sort_order == 'Ascending')
    )
    
    # Format table
    st.dataframe(
        df_display[['Data', 'Nume', 'Tip', 'Distanță (km)', 'Durată (min)', 'Calorii', 'Ritm cardiac mediu']],
        width='stretch',
        hide_index=True
    )
    
    # Download data
    st.markdown("---")
    st.header("Export Data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        csv = df_filtered.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f'activities_garmin_{datetime.now().strftime("%Y%m%d")}.csv',
            mime='text/csv'
        )
    
    with col2:
        st.info(f"Filtered activities: {len(df_filtered)} out of {len(df)} activities")
