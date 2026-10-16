# Isolated Paderborn (KAt) MATLAB 5.0 Reader

This utility is **isolated**. It is not integrated into training,
preprocessing, resampling, or evaluation pipelines.

## Dependencies

- Python 3.9+
- `numpy`, `scipy` (for `scipy.io.loadmat`)

Install (isolated venv recommended):

```bash
python -m venv .venv && . .venv/bin/activate
pip install numpy scipy
```

## Usage

Text report:

```bash
python -m tools.paderborn_mat_reader.cli path/to/file.mat
python -m tools.paderborn_mat_reader.cli a.mat b.mat
```

JSON output:

```bash
python -m tools.paderborn_mat_reader.cli --json path/to/file.mat
```

Exit code is non-zero on any parse/validation failure.

## Supported MAT-file structure

A single top-level variable (the recording name) that is a MATLAB
struct with fields:

- `Info` (optional)
- `X` (cell array of time-axis structs)
- `Y` (cell array of channel structs)
- `Description` (optional)

Each `X` entry must contain `Name` and `Data` (time values), and may
contain `Unit`. Each `Y` entry must contain `Name`, `Data` (values),
and `XIndex` (1-based index into `X`), and may contain `Unit`.

Channels are discovered by their `Name` metadata, never by array
position. At least `vibration_1` and `speed` are supported.

## XIndex interpretation

`XIndex` is treated as **1-based** and resolved as `X[XIndex - 1]`.
Out-of-range or non-integer values raise an error.

## Validation

- Each selected channel must have a matching time vector of equal length.
- Timestamps must be finite and strictly increasing.
- NaN/Inf, duplicate, or decreasing timestamps are rejected.
- Missing target channels, malformed root/entries, and invalid XIndex
  raise informative errors identifying the file and field/channel.

## Units

Units are taken verbatim from metadata. Missing/empty units become
`UNKNOWN`. Speed is never converted to Hz and is never assumed to be
RPM; its unit is reported as stored.

## Sample rate

Two conditional rates are reported (valid only if timestamps are in
seconds):

- `endpoint_rate_hz` = `(N - 1) / (t_last - t_first)` (approximate
  endpoint-derived rate).
- `median_interval_rate_hz` = `1 / median(diff(t))` (diagnostic rate).

These may differ when timestamps are non-uniform. The maximum absolute
residual against the endpoint-derived uniform grid is also reported.
Neither value is a verified physical sampling rate.

## Common time support

`support_start = max(vib_first, speed_first)`,
`support_end   = min(vib_last,  speed_last)`.
Overlap duration is reported only when `support_end > support_start`.
No alignment or resampling is performed.

## Test coverage vs. real data

Synthetic tests validate parser behavior against the fixture schema
only. They do **not** prove compatibility with actual KA01 files.
Actual KA01 compatibility remains unverified until a separate local
smoke test opens the real recordings (not possible in this environment
due to network restrictions).
