# API Reference

Overview of the core `cfr` classes organized by module.

| Module | Classes | Description |
|--------|---------|-------------|
| [Proxy](proxy.md) | `ProxyRecord`, `ProxyDatabase` | Individual records and collections of proxy data |
| [Climate](climate.md) | `ClimateField` | Gridded climate data (wraps xarray) |
| [Timeseries](timeseries.md) | `EnsTS` | Ensemble timeseries with uncertainty |
| [PSM](psm.md) | `Linear`, `Bilinear`, `VSLite`, `Coral_SrCa`, ... | Proxy System Models |
| [Data Assimilation](da.md) | `EnKF` | Ensemble Kalman Filter |
| [Reconstruction](reconstruction.md) | `ReconJob`, `ReconRes` | Reconstruction workflows and results |
