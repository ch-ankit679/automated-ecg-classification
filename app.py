import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.interpolate import interp1d
import altair as alt

from predict import predict_ecg
from utils.preprocessing import inject_gaussian_noise, inject_baseline_wander

# -------------------------
# Page Config
# -------------------------
st.set_page_config(
    page_title="ECG Classification System",
    layout="wide" # Set to wide for better visualization of side-by-side graphs
)

# -------------------------
# Helper Function: XAI Plot
# -------------------------
def plot_attention_heatmap(signal_1d, attn_1d):
    """Generates the Explainable AI heatmap overlay."""
    x_old = np.linspace(0, 1, len(attn_1d))
    x_new = np.linspace(0, 1, len(signal_1d))
    f = interp1d(x_old, attn_1d, kind='cubic')
    attn_upsampled = f(x_new)
    
    # Normalize between 0 and 1
    attn_norm = (attn_upsampled - attn_upsampled.min()) / (attn_upsampled.max() - attn_upsampled.min() + 1e-8)
    
    fig, ax = plt.subplots(figsize=(10, 3))
    time_axis = np.arange(len(signal_1d))
    
    # Plot black ECG line
    ax.plot(time_axis, signal_1d, color='black', linewidth=1.5, zorder=2)
    
    # Plot red Heatmap
    cmap = mcolors.LinearSegmentedColormap.from_list('attention', ['#e0f3ff', '#ffffff', '#ff4d4d', '#b30000'])
    heatmap = np.expand_dims(attn_norm, axis=0)
    
    im = ax.imshow(heatmap, aspect='auto', cmap=cmap, 
                   extent=[time_axis[0], time_axis[-1], signal_1d.min()-0.2, signal_1d.max()+0.2], 
                   alpha=0.6, zorder=1)
    
    ax.set_xlabel("Time (Samples)")
    ax.set_ylabel("Amplitude")
    fig.colorbar(im, ax=ax, orientation='horizontal', label='AI Attention Intensity', pad=0.25)
    return fig

# -------------------------
# Helper Function: Mini Signal Plot
# -------------------------
def plot_signal(signal_1d, color="#1f77b4"):
    """Generates an Altair line chart with proper axis labels."""
    df = pd.DataFrame({
        "Time (samples)": np.arange(len(signal_1d)),
        "Amplitude (mV)": signal_1d
    })
    chart = alt.Chart(df).mark_line(color=color).encode(
        x=alt.X("Time (samples):Q", title="Time (samples)"),
        y=alt.Y("Amplitude (mV):Q", title="Amplitude (mV)")
    ).properties(height=180)
    return chart

# -------------------------
# Title
# -------------------------
st.title("💓 ECG Classification System")
st.markdown("### CNN + LSTM + Attention Model")
st.write("Upload a CSV file containing ECG signal data")

# -------------------------
# File Upload
# -------------------------
uploaded_file = st.file_uploader("📂 Upload ECG CSV File", type=["csv"])

# -------------------------
# Main Logic
# -------------------------
if uploaded_file is not None:
    st.success("✅ File uploaded successfully")

    try:
        # Read file and Validate input
        df = pd.read_csv(uploaded_file, header=None)
        if df.shape[1] < 50:
            st.warning("⚠️ ECG signal seems too short. Expected ~187 values.")
        
        # Original Interactive Chart
        data = df.T.reset_index()
        data.columns = ["Time"] + [f"Signal_{i}" for i in range(data.shape[1]-1)]
        data_melted = data.melt("Time", var_name="Signal", value_name="Amplitude")
        chart = alt.Chart(data_melted).mark_line().encode(
            x=alt.X("Time:Q", title="Time (samples)"),
            y=alt.Y("Amplitude:Q", title="Amplitude (mV)"),
            color="Signal:N"
        ).interactive()
        st.altair_chart(chart, use_container_width=True)
        
        uploaded_file.seek(0)

        # -------------------------
        # Prediction
        # -------------------------
        with st.spinner("🔍 Analyzing ECG..."):
            label, confidence, probs, attn_1d, clean_data = predict_ecg(file=uploaded_file)

        # -------------------------
        # Result Display
        # -------------------------
        st.markdown("---")
        st.subheader("🧠 Baseline Prediction Result")

        if label == "Normal":
            st.success(f"**🟢 {label} ECG** (Probability: {confidence*100:.2f}%)")
        else:
            st.error(f"**🔴 {label}** (Probability: {confidence*100:.2f}%)")

        # -------------------------
        # Explainable AI (XAI)
        # -------------------------
        st.markdown("---")
        st.subheader("👁️ Explainable AI (Model Attention)")
        st.write("The red heatmap shows exactly which morphological features (like the QRS complex) the AI is looking at to make its diagnosis.")
        
        xai_fig = plot_attention_heatmap(clean_data.flatten(), attn_1d)
        st.pyplot(xai_fig)

        # -------------------------
        # Clinical Stress Test (Noise)
        # -------------------------
        st.markdown("---")
        st.subheader("🏥 Clinical Stress Test (Noise Robustness)")
        st.write("Testing if the model's result changes when exposed to real-world hospital noise.")

        # Generate Noise
        noisy_gaussian = inject_gaussian_noise(clean_data)
        noisy_wander = inject_baseline_wander(clean_data)

        # Predict Noise
        g_label, g_conf, g_probs, _, _ = predict_ecg(data_array=noisy_gaussian)
        w_label, w_conf, w_probs, _, _ = predict_ecg(data_array=noisy_wander)

        # Layout for Noise tests
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**1. Clean Signal**")
            st.altair_chart(plot_signal(clean_data.flatten(), color="#1f77b4"), use_container_width=True)
            st.info(f"**{label}**\n\nProb: {confidence*100:.1f}%")

        with col2:
            st.markdown("**2. Gaussian Static**")
            st.altair_chart(plot_signal(noisy_gaussian.flatten(), color="#ff7f0e"), use_container_width=True)
            if g_label == label:
                st.success(f"**{g_label}**\n\nProb: {g_conf*100:.1f}%")
            else:
                st.error(f"**{g_label}**\n\nProb: {g_conf*100:.1f}%")

        with col3:
            st.markdown("**3. Baseline Wander**")
            st.altair_chart(plot_signal(noisy_wander.flatten(), color="#2ca02c"), use_container_width=True)
            if w_label == label:
                st.success(f"**{w_label}**\n\nProb: {w_conf*100:.1f}%")
            else:
                st.error(f"**{w_label}**\n\nProb: {w_conf*100:.1f}%")

        # -------------------------
        # Result Shift Comparison Chart
        # -------------------------
        st.markdown("#### 📊 Prediction Result Shifts Across Conditions")
        st.write("Compare how the noise affects the model's decision across all 5 heartbeat classifications.")
        
        class_names = ["Normal", "Atrial Premature", "PVC", "Fusion (V+N)", "Fusion (Paced)"]

        # Create a DataFrame combining all probabilities
        comparison_df = pd.DataFrame({
            "Environment": ["1. Clean"]*5 + ["2. Gaussian Noise"]*5 + ["3. Baseline Wander"]*5,
            "Class": class_names * 3,
            "Probability (%)": np.concatenate([probs * 100, g_probs * 100, w_probs * 100])
        })

        # Plot the comparison using Altair (Grouped Bar Chart)
        bar_chart = alt.Chart(comparison_df).mark_bar().encode(
            x=alt.X('Environment:N', axis=alt.Axis(labels=False, title=None, ticks=False)),
            y=alt.Y('Probability (%):Q', title='Probability (%)', scale=alt.Scale(domain=[0, 100])),
            color=alt.Color('Environment:N', scale=alt.Scale(range=['#1f77b4', '#ff7f0e', '#2ca02c'])),
            column=alt.Column('Class:N', header=alt.Header(titleOrient='bottom', labelOrient='bottom', title='Heartbeat Classification'))
        ).properties(width=110, height=300)

        st.altair_chart(bar_chart, use_container_width=False)

    except Exception as e:
        st.error("❌ Error processing file")
        st.exception(e)

# -------------------------
# Footer
# -------------------------
st.markdown("---")
