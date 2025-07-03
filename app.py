### app.py
import streamlit as st
from utils import simulate
from plots import plot_downtime_distribution, plot_cost_comparison, plot_sensitivity_analysis
import pandas as pd
import numpy as np

# Page configuration
st.set_page_config(
    page_title="IT Infrastructure Reliability Analysis",
    layout="wide"
)

st.title("RAMS en SLA van Optical Repeater incl. SLA kosten")
st.markdown("Monte Carlo simulatie voor infrastructuur betrouwbaarheidsanalyse met SLA vergelijking")

# Sidebar configuration
st.sidebar.header("Simulatie Parameters")

# Component defaults
if 'components' not in st.session_state:
    st.session_state.components = {
        "Repeater": {"MTBF": 100_000, "MTTR_no_SLA": 72, "MTTR_with_SLA": 12},
        "Voeding/UPS": {"MTBF": 50_000, "MTTR_no_SLA": 48, "MTTR_with_SLA": 8},
        "Glasvezel": {"MTBF": 100_000, "MTTR_no_SLA": 24, "MTTR_with_SLA": 24},
        "Omgevingsfactoren": {"MTBF": 50_000, "MTTR_no_SLA": 48, "MTTR_with_SLA": 10}
    }

components = st.session_state.components

# Sidebar controls
for name, params in components.items():
    st.sidebar.subheader(name)
    params["MTBF"] = st.sidebar.number_input(f"{name} - MTBF (uren)", 1000, 1_000_000, params["MTBF"], step=1000)
    params["MTTR_no_SLA"] = st.sidebar.number_input(f"{name} - MTTR zonder SLA (uren)", 1, 200, params["MTTR_no_SLA"])
    params["MTTR_with_SLA"] = st.sidebar.number_input(f"{name} - MTTR met SLA (uren)", 1, 100, params["MTTR_with_SLA"])

# Cost parameters
sla_cost_total = st.sidebar.number_input("Totale SLA kosten (5 jaar) (€)", 100, 20_000, 8_665, step=50)
inflation_rate = st.sidebar.number_input("Inflatie per jaar (%)", 0.0, 10.0, 4.5, step=0.1)

# Simulation parameters
n_simulations = st.sidebar.number_input("Aantal simulaties", 1_000, 50_000, 10_000, step=1_000)

# Start simulation
if st.sidebar.button("Start Simulatie", type="primary"):
    with st.spinner("Monte Carlo simulatie wordt uitgevoerd..."):
        
        # Inflation correction
        inflation_factor = (1 + inflation_rate / 100) ** 3
        sla_cost_per_year = (sla_cost_total / 5) * inflation_factor
        
        # Run simulations
        downtime_no_sla, costs_no_sla = simulate(
            components, "MTTR_no_SLA", False, n_simulations, sla_cost_per_year, seasonal_factor=1.0
        )
        downtime_with_sla, costs_with_sla = simulate(
            components, "MTTR_with_SLA", True, n_simulations, sla_cost_per_year, seasonal_factor=1.0
        )

        # Store in session
        st.session_state["results"] = {
            "downtime_no_sla": downtime_no_sla,
            "costs_no_sla": costs_no_sla,
            "downtime_with_sla": downtime_with_sla,
            "costs_with_sla": costs_with_sla
        }

# Display results
if "results" in st.session_state:
    res = st.session_state["results"]
    
    st.header("Belangrijkste Resultaten")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Gem. Downtime (Zonder SLA)", f"{np.mean(res['downtime_no_sla']):.1f} uren")
        st.metric("Gem. Kosten (Zonder SLA)", f"€{np.mean(res['costs_no_sla']):,.0f}")
    with col2:
        st.metric("Gem. Downtime (Met SLA)", f"{np.mean(res['downtime_with_sla']):.1f} uren")
        st.metric("Gem. Kosten (Met SLA)", f"€{np.mean(res['costs_with_sla']):,.0f}")

    # Plots
    st.subheader("Downtime Verdeling")
    st.plotly_chart(plot_downtime_distribution(res), use_container_width=True)

    st.subheader("Kosten Vergelijking")
    st.plotly_chart(plot_cost_comparison(res), use_container_width=True)

    st.subheader("Gevoeligheidsanalyse")
    st.plotly_chart(plot_sensitivity_analysis(components, sla_cost_per_year), use_container_width=True)
