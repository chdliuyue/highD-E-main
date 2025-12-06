# VT-CPFM-1 Model Notes

## 1. Model Formulation
We adopt the VT-CPFM-1 (Virginia Tech Comprehensive Power-Based Fuel Consumption Model, type 1) to estimate instantaneous fuel use and CO₂ emissions at the frame level of the highD dataset. The traction power demand at time $t$ (in kW) follows the simplified longitudinal dynamics in Fahmin et al. (2022, Eq. 2):

$$
P(t) = \frac{\big(m(1+\lambda) \cdot a(t) + m g \frac{\mathrm{Cr}}{1000} (c_1 v_{\mathrm{km/h}}(t) + c_2) + 0.5\,\rho\,A_f C_d\, v(t)^2\big)\, v(t)}{1000\,\eta_d}
$$

Where:
- $m$ is vehicle mass [kg]; $\lambda$ is the rotational mass factor (dimensionless);
- $a(t)$ is longitudinal acceleration [m/s²]; $v(t)$ is speed [m/s]; $v_{\mathrm{km/h}}$ is speed converted to km/h;
- $g=9.80665$ m/s² is gravity; $\rho=1.2256$ kg/m³ is air density;
- $\mathrm{Cr}, c_1, c_2$ are rolling resistance parameters (Fahmin et al., 2022);
- $A_f$ is frontal area [m²]; $C_d$ is drag coefficient;
- $\eta_d$ is drivetrain efficiency.

Fuel consumption rate $\mathrm{FC}(t)$ (L/s) uses the concave quadratic VT-CPFM-1 fuel map (Rakha et al., 2011; Park et al., 2013):

$$
\mathrm{FC}(t) = \begin{cases}
\alpha_0 + \alpha_1 P(t) + \alpha_2 P(t)^2, & P(t) \ge 0 \\
\alpha_0, & P(t) < 0
\end{cases}
$$

CO₂ emission rate (g/s) is derived from carbon-content factors: gasoline 2310 g/L, diesel 2680 g/L (U.S. EPA, 2021):
- LDV (gasoline): $E_{CO2} = \mathrm{FC} \times 2310$
- HDDT (diesel): $E_{CO2} = \mathrm{FC} \times 2680$

## 2. Representative Vehicle Parameters
All highD vehicles are mapped to two representative powertrain/shape classes.

### 2.1 LDV (Honda Civic, gasoline)
Parameters primarily from Kamalanathsharma (2014) Table 7.3 with rolling resistance terms from Fahmin et al. (2022) Table 1:

| Parameter | Value | Source |
| --- | --- | --- |
| Mass $m$ | 1453 kg | Kamalanathsharma 2014, Table 7.3 |
| Drag coefficient $C_d$ | 0.30 | Kamalanathsharma 2014 |
| Frontal area $A_f$ | 2.32 m² | Kamalanathsharma 2014 |
| Drivetrain efficiency $\eta_d$ | 0.92 | Kamalanathsharma 2014 |
| Rotational mass factor $\lambda$ | 0.04 | VT-CPFM recommended (Rakha 2011) |
| Rolling resistance $\mathrm{Cr}$ | 1.75 | Fahmin 2022, Table 1 (Honda Accord) |
| $c_1$ | 0.0328 | Fahmin 2022, Table 1 |
| $c_2$ | 4.575 | Fahmin 2022, Table 1 |
| $\alpha_0$ | $4.7738\times10^{-4}$ L/s | Kamalanathsharma 2014 |
| $\alpha_1$ | $5.363\times10^{-5}$ L/(s·kW) | Kamalanathsharma 2014 |
| $\alpha_2$ | $1.0\times10^{-6}$ L/(s·kW²) | Kamalanathsharma 2014 |

### 2.2 HDDT (International 9800 SBA, diesel)
Parameters from Fahmin et al. (2022) Table 1; fuel map coefficients temporarily reuse LDV values pending heavy-duty calibration:

| Parameter | Value | Source/Note |
| --- | --- | --- |
| Mass $m$ | 7239 kg | Fahmin 2022, Table 1 |
| Drag coefficient $C_d$ | 0.78 | Fahmin 2022, Table 1 |
| Frontal area $A_f$ | 8.90 m² | Fahmin 2022, Table 1 |
| Drivetrain efficiency $\eta_d$ | 0.95 | Typical HDDT VT-CPFM assumption |
| Rotational mass factor $\lambda$ | 0.10 | VT-CPFM heavy-duty default |
| Rolling resistance $\mathrm{Cr}$ | 1.75 | Fahmin 2022, Table 1 |
| $c_1$ | 0.0328 | Fahmin 2022, Table 1 |
| $c_2$ | 4.575 | Fahmin 2022, Table 1 |
| $\alpha_0$ | $4.7738\times10^{-4}$ L/s | Reused from LDV (simplifying assumption) |
| $\alpha_1$ | $5.363\times10^{-5}$ L/(s·kW) | Reused from LDV |
| $\alpha_2$ | $1.0\times10^{-6}$ L/(s·kW²) | Reused from LDV |

## 3. CO2 and Fuel Outputs in the Pipeline
The VT-CPFM-1 output in this project includes per-frame power ($P$), fuel rate ($\mathrm{FC}$), and CO₂ rate columns:
- `cpf_power_kw`: traction power demand [kW]
- `cpf_fuel_rate_lps`: fuel consumption rate [L/s]
- `cpf_co2_rate_gps`: CO₂ emission rate [g/s]

CO₂ and fuel totals over a time window are obtained by integrating the corresponding rates over time (sum of rate · $\Delta t$).

## 4. Assumptions and Limitations
- All highD `Car` records are treated as Honda Civic-like LDVs; all `Truck` records as International 9800 SBA HDDTs.
- HDDT fuel map coefficients ($\alpha_0, \alpha_1, \alpha_2$) reuse the LDV set as a conservative placeholder. The study focuses on relative marginal emission costs across behaviors/events rather than absolute fleet inventories.
- Road grade is assumed zero (flat highway); no gear-shift dynamics are modeled.
- Future work can replace HDDT fuel coefficients with calibrated heavy-duty VT-CPFM values once available, improving absolute emission estimates without changing the pipeline API.
