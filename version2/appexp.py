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
feed_eeg         = st.sidebar.number_input("Feed‑in tariff EEG (€/kWh)", value=0.11470, step=0.00001, format="%.5f")
feed_augmented   = st.sidebar.number_input("Feed‑in tariff EEG Augmented (€/kWh)", value=0.15000, step=0.00001, format="%.5f")
tariff_choice    = st.sidebar.selectbox("Select tariff variant", ["EEG", "EEG Augmented"])
tariffs          = {"EEG": feed_eeg, "EEG Augmented": feed_augmented}
selected_tariff  = tariffs[tariff_choice]

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
debt_pct      = st.sidebar.slider("Debt %", 0, 100, 0)
interest_rate = float(st.sidebar.text_input("Debt interest rate (%)", "0")) / 100
debt_term     = int(st.sidebar.text_input("Debt term (years)", "0"))
tax_rate      = float(st.sidebar.text_input("Tax rate (%)", "0")) / 100

st.sidebar.header("7. Model Assumptions")
horizon = st.sidebar.number_input("Projection horizon (years)", min_value=1, value=20, step=1)
depr_period = st.sidebar.number_input("Depreciation period (years)", min_value=1, value=40, step=1)
degrade_rate = st.sidebar.number_input("Annual degradation (%)", min_value=0.0, value=0.0, step=0.1) / 100
cost_of_equity = st.sidebar.number_input(
    "Cost of equity (%)",
    value=0.0,
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

 # Derived values for EEG and EEG Augmented variants
yield_per_year_eeg = (annual_generation * feed_eeg) - annual_om - annual_lease
yield_per_year_aug = (annual_generation * feed_augmented) - annual_om - annual_lease

monthly_yield_eeg = yield_per_year_eeg / 12
monthly_yield_aug = yield_per_year_aug / 12

# Additional first-year metrics
gross_profit_eeg = annual_generation * feed_eeg
gross_yield_15ct = annual_generation * 0.15

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


col_extra1, col_extra2 = st.columns(2)
col_extra1.metric("Yield/year EEG (€)", f"{yield_per_year_eeg:,.2f}")
col_extra1.metric("Yield/month EEG (€)", f"{monthly_yield_eeg:,.2f}")
col_extra2.metric("Yield/year EEG Aug. (€)", f"{yield_per_year_aug:,.2f}")
col_extra2.metric("Yield/month EEG Aug. (€)", f"{monthly_yield_aug:,.2f}")

# Gross yield metrics
col_gross1, col_gross2 = st.columns(2)
col_gross1.metric("Gross profit (EEG) (€)", f"{gross_profit_eeg:,.2f}")
col_gross2.metric("Gross yield EEG Aug. (€)", f"{gross_yield_15ct:,.2f}")


# ---- Additional metrics used in dashboard ----
revenue_aug = np.array([
    annual_generation * feed_augmented * (1 - degrade_rate)**(y-1)
    for y in years
])
tax_base_aug = revenue_aug - om_ser - debt_ser - depr_ser
tax_aug = np.where(tax_base_aug > 0, tax_base_aug * tax_rate, 0)
cashflow_aug = revenue_aug - om_ser - debt_ser - tax_aug
cashflow_aug[0] -= equity_amount

npv_eeg = nf.npv(wacc, cashflow)
npv_aug = nf.npv(wacc, cashflow_aug)
irr_eeg = nf.irr(cashflow) * 100
irr_aug = nf.irr(cashflow_aug) * 100
profit_eeg = cumulative[-1]
profit_aug = np.cumsum(cashflow_aug)[-1]

c7, c8 = st.columns(2)
c7.metric("Equity Invested (€)", f"{equity_amount:,.0f}")
# End-of-horizon cumulative profit
c8.metric("Profit at Horizon (€)", f"{cumulative[-1]:,.0f}")

# --- Additional Metrics for EEG and EEG Augmented ---
col9, col10, col11 = st.columns(3)
col9.metric("IRR EEG (%)", f"{irr_eeg:.2f}")
col9.metric("IRR EEG Aug. (%)", f"{irr_aug:.2f}")
col10.metric("NPV EEG (€)", f"{npv_eeg:,.0f}")
col10.metric("NPV EEG Aug. (€)", f"{npv_aug:,.0f}")
col11.metric("Profit at Horizon EEG (€)", f"{profit_eeg:,.0f}")
col11.metric("Profit at Horizon EEG Aug. (€)", f"{profit_aug:,.0f}")

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
        "EEG": "EEG",
        "EEG Augmented": "EEG Augmented"
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

st.subheader("Download Scenario Summary")

selected_row = df_scen[df_scen["Scenario"] == tariff_choice]
if not selected_row.empty:
    annual_prod = annual_generation

# --- PDF Export using fpdf ---
from fpdf import FPDF

class PDF(FPDF):
    def header(self):
        self.set_font("Arial", 'B', 14)
        self.cell(0, 10, "Solar Project Summary", ln=True, align="C")
        self.ln(10)

def create_pdf():
    pdf = PDF()
    pdf.alias_nb_pages()
    pdf.set_font("Arial", 'B', 12)
    pdf.set_font("Arial", size=12)
    rows = []

    # --- Cash Flow projections for EEG and EEG Augmented scenarios ---
    # Full cash flow for EEG Augmented
    revenue_aug = np.array([
        annual_generation * feed_augmented * (1 - degrade_rate)**(y-1)
        for y in years
    ])
    tax_base_aug = revenue_aug - om_ser - debt_ser - depr_ser
    tax_aug = np.where(tax_base_aug > 0, tax_base_aug * tax_rate, 0)
    cashflow_aug = revenue_aug - om_ser - debt_ser - tax_aug
    cashflow_aug[0] -= equity_amount

    npv_eeg = nf.npv(wacc, cashflow)
    npv_aug = nf.npv(wacc, cashflow_aug)
    irr_eeg = nf.irr(cashflow) * 100
    irr_aug = nf.irr(cashflow_aug) * 100
    profit_eeg = cumulative[-1]
    profit_aug = np.cumsum(cashflow_aug)[-1]

    # Scenario Parameters
    rows += [
        ("--- Scenario Parameters ---", ""),
        ("Project Name", "Eichzoepfen solar park"),
        ("Location", "Bad Camberg, Hesse"),
        ("System Type", "Ground-mounted PV"),
        ("Client / Operator", "ECOWAY GmbH"),
        ("Global irradiation", "1086.72 kWh/m2/yr"),
        ("Modules", f"{num_modules} units"),
        ("Power per Module", f"{power_per_mod} Wp"),
        ("Feed-in Tariff EEG", f"{feed_eeg:.3f} EUR/kWh"),
        ("Feed-in Tariff EEG Augmented", f"{feed_augmented:.3f} EUR/kWh"),
        ("Selected Tariff", tariff_choice),
        ("Insurance (per kWp/yr)", f"{ins_cost:.2f}"),
        ("Maintenance (per kWp/yr)", f"{maint_cost:.2f}"),
        ("Lease Cost (per yr)", f"{lease_cost:,.0f}"),
        ("CAPEX Transformer", f"{capex_trans:.0f} per kWp"),
        ("CAPEX Assembly", f"{capex_assembly:.0f} per kWp"),
        ("Debt percent", f"{debt_pct}%"),
        ("Interest Rate", f"{interest_rate*100:.2f}%"),
        ("Debt Term (years)", f"{debt_term}"),
        ("Tax Rate", f"{tax_rate*100:.2f}%"),
        ("Depreciation Period", f"{depr_period} years"),
        ("Degradation Rate", f"{degrade_rate*100:.1f}%"),
        ("Cost of Equity", f"{cost_of_equity*100:.1f}%"),
        ("Projection Horizon", f"{horizon} years")
    ]

    # Technical Values
    rows += [
        ("--- Technical Values ---", ""),
        ("Output in kWp", f"{total_capacity:.3f}"),
        ("kWh per kWp per year", f"{specific_yield:.2f}"),
        ("Annual electricity prod. in kWh", f"{annual_generation:,.0f}"),
    ]

    # Revenue & Tariffs
    rows += [
        ("--- Revenue & Tariffs ---", ""),
        ("Feed-in tariff", f"{selected_tariff:.2f} EUR/kWh"),
        ("Annual income from feed-in", f"{annual_revenue:,.0f} EUR"),
    ]

    # Operating Costs
    rows += [
        ("--- Operating Costs ---", ""),
        ("Maintenance", f"{maint_cost * total_capacity:,.0f} EUR"),
        ("Insurance", f"{ins_cost * total_capacity:,.0f} EUR"),
        ("Total yield per year", f"{annual_revenue - (maint_cost + ins_cost) * total_capacity:,.0f} EUR"),
    ]

    # Yield Calculations
    rows += [
        ("--- Yield Calculations ---", ""),
        ("Yield/year EEG", f"{yield_per_year_eeg:,.2f} EUR"),
        ("Yield/month EEG", f"{monthly_yield_eeg:,.2f} EUR"),
        ("Yield/year EEG Augmented", f"{yield_per_year_aug:,.2f} EUR"),
        ("Yield/month EEG Augmented", f"{monthly_yield_aug:,.2f} EUR"),
    ]

    # Gross Yield Estimates
    rows += [
        ("--- Gross Yield Estimates ---", ""),
        ("Gross profit with EEG remuneration", f"{gross_profit_eeg:,.2f} EUR"),
        ("Gross yield with 15 ct/kWh", f"{gross_yield_15ct:,.2f} EUR")
    ]

    # Financial Overview
    rows += [
        ("--- Financial Overview ---", ""),
        ("Investment (CapEx)", f"{capex_total:,.0f} EUR"),
        ("Equity capital", f"{equity_amount:,.0f} EUR"),
        ("NPV", f"{npv:,.0f} EUR"),
        ("IRR", f"{irr * 100:.2f} %"),
        ("Profit at Horizon", f"{cumulative[-1]:,.0f} EUR"),
        ("NPV EEG", f"{npv_eeg:,.0f} EUR"),
        ("NPV EEG Augmented", f"{npv_aug:,.0f} EUR"),
        ("Profit at Horizon EEG", f"{profit_eeg:,.2f} EUR"),
        ("Profit at Horizon EEG Augmented", f"{profit_aug:,.2f} EUR")
    ]

    # Calculate possible results for EEG Augmented scenario
    possible_result_15ct = yield_per_year_aug * horizon
    possible_overall_15ct = yield_per_year_aug * horizon

    pdf.add_page()  # new page before Summary Template
    # Summary Template
    rows += [
        ("--- Summary Template ---", ""),
        ("Output in kWp", f"{total_capacity:.3f}"),
        ("kWh per kWp per year", f"{specific_yield:.2f}"),
        ("Annual electricity prod. in kWh", f"{annual_generation:,.0f}"),
        ("Annual CO2 savings (kg)", f"{annual_generation * 0.576:,.0f}"),
        ("Total investment", f"{capex_total:,.0f} EUR"),
        ("Annual EEG result (Year 1, no deg)", f"{gross_profit_eeg:,.2f} EUR"),
        ("EEG result (monthly, no deg)", f"{monthly_yield_eeg:,.2f} EUR"),
        ("Annual EEG Augmented result (Year 1, no deg)", f"{yield_per_year_aug:,.2f} EUR"),
        ("EEG Augmented result (monthly, no deg)", f"{monthly_yield_aug:,.2f} EUR"),
        ("Maintenance", f"{maint_cost * total_capacity:,.2f} EUR"),
        ("Insurance", f"{ins_cost * total_capacity:,.2f} EUR"),
        ("Total yield per year EEG", f"{yield_per_year_eeg:,.2f} EUR"),
        ("Total yield per year Augmented", f"{yield_per_year_aug:,.2f} EUR"),
        ("Possible result in EEG runtime", f"{cumulative[-1]:,.2f} EUR"),
        ("Possible overall result over lease term", f"{cumulative[-1]:,.2f} EUR"),
        ("Possible result in EEG Augmented runtime", f"{possible_result_15ct:,.2f} EUR"),
        ("Possible overall result with EEG Augmented", f"{possible_overall_15ct:,.2f} EUR"),
        ("Gross profit with EEG remuneration (%)", f"{(gross_profit_eeg / capex_total * 100):.2f} %"),
        ("Gross yield EEG Augmented (%)", f"{(gross_yield_15ct / capex_total * 100):.2f} %"),
        # --- Insert IRR for EEG and EEG Augmented scenarios ---
        ("IRR EEG (%)", f"{irr_eeg:.2f} %"),
        ("IRR EEG Augmented (%)", f"{irr_aug:.2f} %")
    ]

    col1_width = 90
    col2_width = 100
    line_height = 8
    for label, value in rows:
        if label.startswith("---"):
            pdf.set_font("Arial", 'B', 12)
            pdf.cell(0, line_height + 2, label.replace("-", "").strip(), ln=True)
            pdf.set_font("Arial", size=12)
        else:
            pdf.cell(col1_width, line_height, label, border=0)
            pdf.cell(col2_width, line_height, value, ln=True, border=0)
    return bytes(pdf.output(dest='S').encode('latin1', errors='replace'))

st.subheader("Download PDF Summary (Offline Mode)")
pdf_bytes = create_pdf()
st.download_button("📄 Download Summary as PDF", data=pdf_bytes, file_name="solar_summary.pdf", mime="application/pdf")
