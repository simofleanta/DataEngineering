import streamlit as st
import pandas as pd
import os
import random
import string
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

#We need to evaluate the ghost job detection model's performance using a confusion matrix to measure whether 
#if it reliably identifies fraudulent job postings or fails to detect critical cases.
#This analysis will help us understand the model's strengths and weaknesses, 
#guiding improvements to enhance its accuracy and reliability.


# Page configuration
st.set_page_config(
    page_title="Confusion Matrix Analysis",
    layout="wide"
)

# Get the directory of the current file
current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the path to JOB_T.xlsx
file_path = os.path.join(current_dir, 'JOB_T.xlsx')

# Read the Excel file
df = pd.read_excel(file_path)

# Create confusion matrix
y_true = (df['Job_manual'] == 'T').astype(int)
y_pred = (df['Job_predicted'] == 'T').astype(int)

cm = confusion_matrix(y_true, y_pred)
tn, fp, fn, tp = cm.ravel()

# Calculate metrics
accuracy = (tp + tn) / (tp + tn + fp + fn)
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

# Calculate percentages
total = cm.sum()
cm_percent = (cm / total) * 100

# Create annotations with labels, count and percentage
labels = [['TN', 'FP'], ['FN', 'TP']]
annotations = [[f'{labels[i][j]}={cm[i][j]}\n({cm_percent[i][j]:.1f}%)' 
                for j in range(2)] 
               for i in range(2)]

# Create heatmap
fig, ax = plt.subplots(figsize=(6, 4))
sns.heatmap(cm, annot=annotations, fmt='', cmap='Blues',
            xticklabels=['False', 'True'], 
            yticklabels=['False', 'True'],
            cbar_kws={'label': 'Count'},
            annot_kws={'fontsize': 8},
            ax=ax)
ax.tick_params(axis='both', labelsize=8)
ax.set_xlabel('Predicted', fontsize=8)
ax.set_ylabel('Actual', fontsize=8)
ax.set_title('Job Predicted vs Job Manual', 
             fontsize=8)

plt.tight_layout()
st.pyplot(fig)

st.markdown("<br><br>", unsafe_allow_html=True)

# Display metrics in columns
st.subheader("Performance Metrics")
metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

with metric_col1:
    st.metric("Accuracy", f"{accuracy:.2%}")
with metric_col2:
    st.metric("Precision", f"{precision:.2%}")
with metric_col3:
    st.metric("Recall", f"{recall:.2%}")
with metric_col4:
    st.metric("F1-Score", f"{f1:.4f}")

st.markdown("<br>", unsafe_allow_html=True)

# Interpretation
st.subheader("Interpretation")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Key Findings:**")
    st.markdown(f"""
    - **False Negatives (FN={fn})**: {fn/(fn+tp)*100:.1f}% of ghost jobs are missed
    - **False Positives (FP={fp})**: {fp/(fp+tn)*100:.1f}% of real jobs incorrectly flagged
    - Model accuracy is only {accuracy:.1%}, suggesting poor performance
    """)

with col2:
    st.markdown("**Model Behavior:**")
    if precision > 0.9:
        st.markdown("- Very conservative - high confidence when flagging ghost jobs")
    if recall < 0.6:
        st.markdown(f"- **Low Recall ({recall:.1%})**: Model misses critical information - only detects {recall:.1%} of ghost jobs, allowing most fraudulent postings through")
    if accuracy < 0.6:
        st.markdown("- Overall performance below acceptable threshold")

st.markdown("<br>", unsafe_allow_html=True)

# Extract problematic jobs
st.subheader("Problematic Cases")

# False Positives - Real jobs flagged as ghost
fp_jobs = df[(df['Job_manual'] == 'F') & (df['Job_predicted'] == 'T')]

# False Negatives - Ghost jobs missed
fn_jobs = df[(df['Job_manual'] == 'T') & (df['Job_predicted'] == 'F')]

tab1, tab2 = st.tabs([f"False Positives ({len(fp_jobs)})", f"Missed Ghost Jobs ({len(fn_jobs)})"])

with tab1:
    st.markdown("**Real jobs incorrectly flagged as ghost jobs (False Alarms)**")
    if len(fp_jobs) > 0:
        st.dataframe(fp_jobs, width="stretch")
        
        st.markdown("---")
        st.markdown("**Solutions to Reduce False Alarms:**")
        st.markdown(f"""
        - **Increase Classification Threshold**: Raise confidence threshold to reduce false positives
        - **Improve Feature Quality**: Verify legitimate company indicators (verified domains, company age, location)
        - **Review These {len(fp_jobs)} Cases**: Analyze common patterns in misclassified real jobs
        - **Add Whitelist**: Maintain verified companies/recruiters to avoid false flagging
        - **Balance Precision-Recall**: Currently at {precision:.1%} precision - find optimal threshold
        - **Human-in-the-Loop**: Flag borderline cases for manual review instead of auto-blocking
        """)
    else:
        st.info("No false positives detected")

with tab2:
    st.markdown("**Ghost jobs that were missed (False Negatives)**")
    if len(fn_jobs) > 0:
        st.dataframe(fn_jobs, width="stretch")
        
        st.markdown("---")
        st.markdown("**Solutions to Improve Performance:**")
        st.markdown(f"""
        - **Adjust Classification Threshold**: Lower the threshold to increase sensitivity and catch more ghost jobs
        - **Feature Engineering**: Add more discriminative features (posting patterns, company verification, job description quality)
        - **Retrain with More Data**: Especially focusing on the {len(fn_jobs)} missed cases to learn their patterns
        - **Ensemble Methods**: Combine multiple models to improve recall
        - **Cost-Sensitive Learning**: Penalize false negatives more heavily during training
        - **Target Recall > 80%**: Currently at {recall:.1%}, aim for higher detection rate even if precision drops slightly
        """)
    else:
        st.info("No missed ghost jobs")
