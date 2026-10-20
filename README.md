# uwreplay-gate

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23493739.svg)](https://doi.org/10.5281/zenodo.23493739)

Offloading and freshness decisions for AUV networks on replayed sea-trial channels: the frames, receivers, replay
protocol, rebuilt published designs, short-probe study, result files and number-pinning tests behind the paper of
the same name (submitted to the IEEE Transactions on Vehicular Technology Special Section on Edge Intelligence and
In-Network Computing for Connected Autonomous Underwater Vehicles).

Two standard frame families are replayed through seven measured channels of the Underwater Acoustic Channel
Library (Zenodo 10.5281/zenodo.21287414, CC BY 4.0) on one hydrophone: a JANUS-like frequency-hopped BFSK frame
with a rate-1/2 convolutional code and its rate ladder, and a coherent single-carrier frame received by a
fractionally spaced decision-feedback equalizer with RLS adaptation, Doppler compensation and a phase-locked loop.

- Packet success reaches a plateau that no signal-to-noise ratio removes. With this single-hydrophone receiver the
  goodput with retransmissions is at most about 700 bit/s with the robust frame and 2.7 kbit/s with the coherent
  one, 65 times below the per-device rate at which a published multi-tier offloading economy breaks even, so the
  published ordering of its schemes inverts, and a published multi-AUV data-collection planner spends most of its
  mission hovering while sensors upload.
- Packet errors arrive in runs. The max-age-first scheduling rule that is optimal for independent erasures loses to
  round-robin wherever the failures cluster relative to independent errors, and a one-slot burst-aware rule
  recovers it.
- A 30 s probe on a moored platform places the link against the economy's break-even success from its own success
  rate, and the clustering of its failures relative to independent errors recovers the scheduling decision once it
  has seen failures. On a moving platform the probe describes the next minute, not the mission.

Every number the paper quotes is written by a script under `scripts/` into `results/`, collected by
`manuscript_numbers.py`, and asserted by `tests/test_findings.py`.

## Layout

| Path | Holds |
|---|---|
| `scripts/modem.py`, `scripts/coherent.py` | the two frame families and their receivers |
| `scripts/linkbudget.py` | the published link budgets (Thorp, four-component noise, Shannon rate) |
| `scripts/fetch_library.py` | downloads and verifies the channel library from Zenodo |
| `scripts/psr_replay.py`, `psr_replay_long.py`, `psr_coherent.py`, `run_queue.py`, `run_coherent_queue.py`, `launch_detached.py` | the replay sweeps and their job runners |
| `scripts/gate_offloading.py`, `gate_offloading2.py` | the multi-tier economy and the data-collection planner, rebuilt |
| `scripts/gate_aoi.py`, `aoi_long.py` | the age-of-information comparison on fitted chains and replayed sequences |
| `scripts/cheap_stats.py` | the short-probe study |
| `scripts/summarise.py`, `summarise_coherent.py`, `figures.py` | summaries, goodput ladders and the four figures |
| `results/` | every result file: gate sweeps, `long/` (160 robust-frame replays), `coherent/` (168 coherent replays and their summary), the rebuilt designs, the probe study, `manuscript_numbers.json` |
| `figs/` | the four figures as 600-dpi PNG and vector PDF, with the numbers they draw as JSON |
| `tests/` | `test_findings.py` pins the paper's numbers to `results/`; `test_packaging.py` guards the released tree |

## Running it

The replays need the channel library (about 19 GB for the seven sites) and the `uwa-channels` package; the
analyses, summaries, numbers, figures and tests need only the result files in this repository.

```
pip install -r requirements.txt
export UWA_LIBRARY_ROOT=/path/to/UWChannelLibrary       # where fetch_library.py puts the .mat files
cd scripts
python fetch_library.py                                 # downloads and MD5-verifies the Zenodo record
python psr_replay.py blue 160                           # one gate sweep: site and chip rate
python -u run_queue.py                                  # every long replay of the robust frame (hours; background jobs)
python psr_coherent.py blue blue_1                      # coherent ladder on one record
python -u run_coherent_queue.py                         # the coherent sweep over every record
python linkbudget.py && python gate_offloading.py && python gate_aoi.py && python aoi_long.py
python gate_offloading2.py && python cheap_stats.py
python summarise.py && python summarise_coherent.py && python figures.py
cd .. && python manuscript_numbers.py
python -m pytest tests -q
```

The job runners start detached single-threaded Windows processes; on another platform run the sweeps directly or
adapt `launch_detached.py`. Nominal signal-to-noise ratio is defined per record (`psr_replay.py` docstring). Green
exceeds the equalizer's 160-tap feedback cap at every coherent rate and Pink above 1000 symbols per second; both
frames exceed Pink's 4 kHz probe band at their highest rates; those rows are reported as such in the result files.

## Citing

See `CITATION.cff`. Cite the concept DOI 10.5281/zenodo.23493739, which resolves to the latest release.

## License

MIT. The channel library is the work of its authors and is distributed under CC BY 4.0 from its own Zenodo record;
it is not included here.
