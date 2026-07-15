# Data layout

`raw_data/` contains the small example PCAP used by the command-line smoke
workflow. `processed_dataset/` contains:

- `X_data.npy`: `(flows, packets, 3)` float array;
- `y_label.npy`: integer class labels;
- `classes.npy`: label-to-name mapping;
- `data_distribution.csv`: class summary.

Feature order is packet size in bytes, direction (`+1` initiator to responder,
`-1` reverse), and non-negative inter-arrival time in seconds. Zero rows are
padding. The bundled matrix retains 100 packets per flow from an earlier data
export; current experiments truncate to the first 30 packets by default.
