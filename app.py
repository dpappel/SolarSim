# --------------------------------------------------------
# Solar Farm Financial Model App
# --------------------------------------------------------
import streamlit as st
import pandas as pd
import numpy as np
import numpy_financial as nf

import plotly.express as px

# --- Sidebar Inputs ---
st.set_page_config(page_title="Solar Farm Financial Model", layout="wide")
st.title("Eichzöpfen Solar Park Financial Model")

# 1. Project Data
st.sidebar.header("1. Project Data")
st.sidebar.write("**Project name:** Eichzöpfen solar park")
st.sidebar.write("**Location:** Bad Camberg, Hesse")
st.sidebar.write("**System type:** Ground-mounted photovoltaics")
st.sidebar.write("**Client / Operator:** ECOWAY GmbH")
st.sidebar.write("**Global irradiation:** 1,086.72 kWh/m²/yr")

# 2. Site Characteristics
st.sidebar.header("2. Site Characteristics")
# Global irradiation is fixed for the site
global_irr = 1086.72  # kWh/m²/yr
specific_yield  = st.sidebar.slider("Specific yield (kWh/kWp/yr)", 900, 1100, 1000)
num_modules     = st.sidebar.number_input("Number of modules", value=1482, step=1)
power_per_mod   = st.sidebar.number_input("Power per module (Wp)", value=710, step=1)

# 3. Income & Remuneration
st.sidebar.header("3. Income & Remuneration")
feed1           = st.sidebar.number_input("Feed‑in tariff low variant (€/kWh)", value=0.065, step=0.001, format="%.3f")
feed2           = st.sidebar.number_input("Feed‑in tariff mid variant (€/kWh)", value=0.075, step=0.001, format="%.3f")
feed3           = st.sidebar.number_input("Feed‑in tariff high variant (€/kWh)", value=0.085, step=0.001, format="%.3f")
tariff_choice   = st.sidebar.selectbox("Select tariff variant", ["Low variant", "Mid variant", "High variant"])
tariffs         = {"Low variant": feed1, "Mid variant": feed2, "High variant": feed3}
selected_tariff = tariffs[tariff_choice]

# 4. Operating Costs
st.sidebar.header("4. Operating Costs (€/kWp/yr)")
ins_cost        = st.sidebar.number_input("Insurance", value=0.80)
maint_cost      = st.sidebar.number_input("Maintenance", value=5.00)
lease_cost      = st.sidebar.number_input("Lease for area (€/yr)", value=12000)

# 5. Construction Costs (CAPEX, €/kWp)
st.sidebar.header("5. Construction Costs (CAPEX, €/kWp)")
capex_trans     = st.sidebar.number_input("Transformer + cable", value=152)
capex_assembly  = st.sidebar.number_input("Assembly + material", value=650)
total_capex     = capex_trans + capex_assembly

st.sidebar.header("6. Financing")
debt_pct      = st.sidebar.slider("Debt %", 0, 100, 70)  # unchanged
interest_rate = float(st.sidebar.text_input("Debt interest rate (%)", "5.0")) / 100
debt_term     = int(st.sidebar.text_input("Debt term (years)", "15"))
tax_rate      = float(st.sidebar.text_input("Tax rate (%)", "21.0")) / 100

st.sidebar.header("7. Model Assumptions")
horizon = st.sidebar.number_input("Projection horizon (years)", min_value=1, value=20, step=1)
depr_period = st.sidebar.number_input("Depreciation period (years)", min_value=1, value=5, step=1)
degrade_rate = st.sidebar.number_input("Annual degradation (%)", min_value=0.0, value=2.0, step=0.1) / 100
cost_of_equity = st.sidebar.number_input(
    "Cost of equity (%)",
    value=10.0,
    step=0.1,
    format="%.1f"
) / 100

# --- Derived Metrics ---
total_capacity    = num_modules * power_per_mod / 1000  # kWp
annual_generation = total_capacity * specific_yield
annual_revenue    = annual_generation * selected_tariff
annual_om         = total_capacity * (ins_cost + maint_cost)
annual_lease      = lease_cost
capex_total       = total_capacity * total_capex

debt_amount       = capex_total * debt_pct / 100
equity_amount     = capex_total - debt_amount
annual_debt_srv   = nf.pmt(interest_rate, debt_term, -debt_amount)
depreciation      = capex_total / depr_period  # straight-line over chosen period

# --- Cash Flow Projection ---
years       = np.arange(1, horizon + 1)
revenue_ser = np.array([
    annual_revenue * (1 - degrade_rate)**(y-1)
    for y in years
])
om_ser       = np.full_like(years, annual_om + annual_lease)
debt_ser = np.array([
    annual_debt_srv if y <= debt_term else 0
    for y in years
])
depr_ser = np.array([
    depreciation if y <= depr_period else 0
    for y in years
])
taxable      = revenue_ser - om_ser - debt_ser - depr_ser
taxes        = np.where(taxable > 0, taxable * tax_rate, 0)
cashflow     = revenue_ser - om_ser - debt_ser - taxes
cashflow[0] -= equity_amount  # account for equity outlay in year 1

# Cumulative cash flow
cumulative = np.cumsum(cashflow)

df = pd.DataFrame({
    "Year": years,
    "Revenue": revenue_ser,
    "O&M + Lease": om_ser,
    "Debt Service": debt_ser,
    "Depreciation": depr_ser,
    "Tax": taxes,
    "Net Cash Flow": cashflow
})

# --- Outputs & Visualizations ---
st.header("Model Outputs")

# --- WACC Calculation ---
debt_fraction   = debt_pct / 100
equity_fraction = 1 - debt_fraction

# After‑tax cost of debt
after_tax_rd = interest_rate * (1 - tax_rate)

# Weighted Average Cost of Capital
wacc = debt_fraction * after_tax_rd + equity_fraction * cost_of_equity

# Use WACC as discount rate for NPV
npv = nf.npv(wacc, cashflow)
irr = nf.irr(cashflow)

# Average annual revenue over the projection horizon
avg_revenue = revenue_ser.mean()


# Key metrics: capacity, CapEx, Year 1 & average revenue, WACC
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total Capacity (kWp)", f"{total_capacity:.2f}")
col2.metric("CapEx Total (€)", f"{capex_total:,.0f}")
col3.metric("Year 1 Revenue (€)", f"{annual_revenue:,.0f}")
col4.metric("Avg Annual Revenue (€)", f"{avg_revenue:,.0f}")
col5.metric("WACC (%)", f"{wacc*100:.2f}")

c5, c6, c7, c8 = st.columns(4)
c5.metric("NPV (€)", f"{npv:,.0f}")
c6.metric("IRR (%)", f"{irr*100:.2f}")
c7.metric("Equity Invested (€)", f"{equity_amount:,.0f}")
# End-of-horizon cumulative profit
c8.metric("Profit at Horizon (€)", f"{cumulative[-1]:,.0f}")

# Scenario Comparison: NPV & IRR for each tariff variant
scenarios = []
for name, tariff in tariffs.items():
    # Compute revenue series for this tariff
    rev_gen = annual_generation * tariff
    rev_ser_scenario = np.array([
        rev_gen * (1 - degrade_rate)**(y-1)
        for y in years
    ])
    # Cashflow series for this scenario
    cashflow_scenario = rev_ser_scenario - om_ser - debt_ser - np.where(
        rev_ser_scenario - om_ser - debt_ser - depr_ser > 0,
        (rev_ser_scenario - om_ser - debt_ser - depr_ser) * tax_rate,
        0
    )
    cashflow_scenario[0] -= equity_amount
    label = {
        "Low variant": "Low variant",
        "Mid variant": "Mid variant",
        "High variant": "High variant"
    }.get(name, name)
    # Store metrics
    scenarios.append({
        "Scenario": label,
        "NPV (€)": nf.npv(interest_rate, cashflow_scenario),
        "IRR (%)": nf.irr(cashflow_scenario) * 100
    })


# Scenario Comparison: Table view
df_scen = pd.DataFrame(scenarios)
st.subheader("Scenario Comparison (by Tariff Variant)")
st.dataframe(
    df_scen.style.format({"NPV (€)": "{:,.0f}", "IRR (%)": "{:.2f}"}),
    use_container_width=True
)

st.subheader("Cumulative Cash Flow Over Time")
cum_df = pd.DataFrame({"Cumulative Cash Flow": cumulative}, index=years).reset_index()
cum_df.columns = ["Year", "Cumulative Cash Flow"]
fig_cum = px.line(
    cum_df,
    x="Year",
    y="Cumulative Cash Flow",
    title="Cumulative Cash Flow Over Time",
    template="plotly_white"
)
fig_cum.update_xaxes(dtick=1)
st.plotly_chart(fig_cum, use_container_width=True)


# Cash flow components over time (stacked)

st.subheader("Cash Flow Components Over Time")
# Make costs negative for plotting
df_plot = df.copy()
df_plot["O&M + Lease"] = -df_plot["O&M + Lease"]
df_plot["Debt Service"] = -df_plot["Debt Service"]
df_plot["Tax"] = -df_plot["Tax"]
fig_cf = px.bar(
    df_plot,
    x="Year",
    y=["Revenue", "O&M + Lease", "Debt Service", "Tax"],
    title="Cash Flow Components Over Time",
    template="plotly_white",
    barmode="relative"
)
fig_cf.update_yaxes(autorange=True)
fig_cf.update_xaxes(dtick=1)
st.plotly_chart(fig_cf, use_container_width=True)

# --- Insert Total Cash Flow Over Time Bar Chart ---
st.subheader("Total Cash Flow Over Time")
fig_total = px.bar(
    df,
    x="Year",
    y="Net Cash Flow",
    title="Total Cash Flow Over Time",
    template="plotly_white"
)
fig_total.update_xaxes(dtick=1)
st.plotly_chart(fig_total, use_container_width=True)


# --- Tax Paid Over Time Bar Chart ---
st.subheader("Tax Paid Over Time")
fig_tax = px.bar(
    df_plot if "df_plot" in locals() else df,
    x="Year",
    y="Tax",
    title="Tax Paid Over Time",
    template="plotly_white"
)
fig_tax.update_xaxes(dtick=1)
st.plotly_chart(fig_tax, use_container_width=True)

st.subheader("CapEx Component Breakdown")
capex_components = pd.DataFrame({
    "Component": ["Transformer + cable", "Assembly + material"],
    "Cost (€)": [capex_trans * total_capacity, capex_assembly * total_capacity]
}).set_index("Component")
st.bar_chart(capex_components)

st.subheader("Detailed Cash Flow Table")
st.dataframe(
    df.style.format({
        "Year": "{:.0f}",
        "Revenue": "{:.0f}",
        "O&M + Lease": "{:.0f}",
        "Debt Service": "{:.0f}",
        "Depreciation": "{:.0f}",
        "Tax": "{:.0f}",
        "Net Cash Flow": "{:.0f}"
    }),
    use_container_width=True
)