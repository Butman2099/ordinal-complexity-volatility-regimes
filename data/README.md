# Input data

Daily spot prices of four energy commodities published by the U.S. Energy
Information Administration (EIA) and distributed through FRED, Federal Reserve
Bank of St. Louis. All four series are marked on FRED as
"Public Domain: Citation Requested". Retrieved on 18 August 2026.

| Market | File | Title on FRED | Units | Observations | Period |
|---|---|---|---|---|---|
| WTI | `DCOILWTICO.csv` | Crude Oil Prices: West Texas Intermediate (WTI) - Cushing, Oklahoma | Dollars per Barrel | 10,226 | 1986-01-02 to 2026-08-18 |
| Brent | `DCOILBRENTEU.csv` | Crude Oil Prices: Brent - Europe | Dollars per Barrel | 9,958 | 1987-05-20 to 2026-08-18 |
| Gasoline | `DGASUSGULF.csv` | Conventional Gasoline Prices: U.S. Gulf Coast, Regular | Dollars per Gallon | 10,099 | 1986-06-02 to 2026-08-18 |
| Henry Hub | `DHHNGSP.csv` | Henry Hub Natural Gas Spot Price | Dollars per Million BTU | 7,436 | 1997-01-07 to 2026-08-18 |

Each file has two columns, `DATE` (YYYY-MM-DD) and `price`. Frequency: daily,
not seasonally adjusted.

The values are the FRED values used in the study, frozen on the retrieval date.
Dates without a quotation on FRED are not included (missing values are removed,
not forward-filled); no other change was made. FRED series are revised and
extended over time, so a fresh download will in general differ from these files.
`features_gen.ipynb` builds the feature tables from these files by default.

Returns are simple returns, because WTI has one negative price (-36.98 on
2020-04-20).

## Citation

U.S. Energy Information Administration, Crude Oil Prices: West Texas
Intermediate (WTI) - Cushing, Oklahoma [DCOILWTICO], retrieved from FRED,
Federal Reserve Bank of St. Louis; https://fred.stlouisfed.org/series/DCOILWTICO,
August 18, 2026.

U.S. Energy Information Administration, Crude Oil Prices: Brent - Europe
[DCOILBRENTEU], retrieved from FRED, Federal Reserve Bank of St. Louis;
https://fred.stlouisfed.org/series/DCOILBRENTEU, August 18, 2026.

U.S. Energy Information Administration, Conventional Gasoline Prices: U.S. Gulf
Coast, Regular [DGASUSGULF], retrieved from FRED, Federal Reserve Bank of
St. Louis; https://fred.stlouisfed.org/series/DGASUSGULF, August 18, 2026.

U.S. Energy Information Administration, Henry Hub Natural Gas Spot Price
[DHHNGSP], retrieved from FRED, Federal Reserve Bank of St. Louis;
https://fred.stlouisfed.org/series/DHHNGSP, August 18, 2026.
