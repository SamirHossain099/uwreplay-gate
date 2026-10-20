"""Second published scheme on the measured link (PLAN item 4): the multi-AUV data-collection scheme of Cao, Chen
and Cheng, Frontiers in Marine Science 2025 (10.3389/fmars.2025.1580751, "MADCS"), reduced to the parts the link
touches.

What the paper does. 150 sensors in a 5 km x 5 km x 1 km volume are clustered into 25 hovering points; K AUVs
(1 to 25) tour the hovering points from a surface data centre at 6 m/s, hover while each member sensor uploads
its D_n packets of 1024 bits in a scheduled train, and return. The average AoI of the delivered data and the
total energy (AUV navigation at 42 W, hovering at 15 W, sensor transmission at 2.8 W) are the objectives; the
scheme's contribution is clustering plus trajectory planning, and its headline is an average-AoI improvement of
10.23% over three baselines (abstract). The link is the Shannon capacity of its own budget (26 kHz carrier,
16 kHz band, 154 dB source level, Thorp absorption, Stojanovic noise): 204 kbit/s at 1 km, 315 kbit/s at 100 m
(src/linkbudget.py), so a 1024-bit packet uploads in 3 to 5 ms, no packet is lost, and hovering is a negligible
part of the mission.

What this script does. It rebuilds that pipeline with the paper's parameters and stated readings (below), plans
the tours with a nearest-neighbour start and 2-opt, and evaluates every K from 1 to 25 over random layouts under
(a) the paper's link and (b) each measured link: a goodput per site that does not depend on distance (the library
ranges, 0.06 to 14 km, bracket the paper's 1 km cells), taken from the FH-BFSK pooled ceilings (results/long/) and,
when results/coherent/summary_coherent.json exists, the coherent-frame ceilings. Where a site's pooled success
never reaches 0.9, the ARQ goodput at its lowest rate (rate x plateau PSR) is used. The quantities reported are
the ones the link changes: hover share of mission time, average AoI, AUV and sensor energy, network lifetime, the
operating K nearest the utopian point of the AoI-energy front (the paper's Fig. 9 rule), and the AoI gain that
trajectory planning buys over a random visiting order, which is the paper's own lever.

Readings the paper leaves open (each stated in the output): D_n is not given, so it is swept (1, 10, 100 packets
per sensor per mission); the sensor's AoI is the time from its upload to its AUV's return to the data centre (the
reading under which AoI "depends on upload periods and navigation periods" and Proposition 1 sums along the
trajectory); hovering time at a hovering point is the sum of its members' upload times; the data centre sits at
the surface above the volume's corner. Collection utility has no formula in the paper and is not reproduced.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from linkbudget import FRONTIERS, shannon_bps, snr_db_frontiers  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
OUT = os.path.join(RES, "gate_offloading2.json")

N_SENSORS, N_CLUSTERS = 150, 25
AREA = (5000.0, 5000.0, 1000.0)
DATA_CENTRE = np.array([0.0, 0.0, 1000.0])
SPEED = 6.0
C_SOUND = 1500.0
PACKET_BITS = 1024
P_NAV, P_HOVER, P_TX = 42.0, 15.0, 2.8
E_SENSOR0, E_AUV0 = 40e3, 600e3
HANDSHAKE_S = 0.0  # the paper's announcement/request/order exchange; propagation delays enter theta directly
D_N_LIST = (1, 10, 100)
K_LIST = list(range(1, 26))
N_LAYOUTS = 20
SEED = 11


def kmeans(x: np.ndarray, k: int, rng: np.random.Generator, iters: int = 50) -> tuple[np.ndarray, np.ndarray]:
    c = x[rng.choice(x.shape[0], k, replace=False)].copy()
    for _ in range(iters):
        lab = np.argmin(((x[:, None, :] - c[None, :, :]) ** 2).sum(-1), axis=1)
        for j in range(k):
            if np.any(lab == j):
                c[j] = x[lab == j].mean(0)
    return c, lab


def tour_length(order: list[int], pts: np.ndarray) -> float:
    path = [DATA_CENTRE] + [pts[i] for i in order] + [DATA_CENTRE]
    return float(sum(np.linalg.norm(path[i + 1] - path[i]) for i in range(len(path) - 1)))


def plan_tour(pts: np.ndarray, rng: np.random.Generator, planned: bool) -> list[int]:
    n = pts.shape[0]
    if n == 0:
        return []
    if not planned:
        return list(rng.permutation(n))
    # nearest neighbour from the data centre, then 2-opt
    left = set(range(n))
    cur = DATA_CENTRE
    order = []
    while left:
        j = min(left, key=lambda i: np.linalg.norm(pts[i] - cur))
        order.append(j)
        left.remove(j)
        cur = pts[j]
    improved = True
    while improved and n > 3:
        improved = False
        for a in range(n - 1):
            for b in range(a + 2, n):
                cand = order[:a + 1] + order[a + 1:b + 1][::-1] + order[b + 1:]
                if tour_length(cand, pts) < tour_length(order, pts) - 1e-9:
                    order = cand
                    improved = True
    return order


def paper_rate_bps(d_m: float) -> float:
    snr = snr_db_frontiers(max(d_m, 1.0), FRONTIERS["sl_db"], FRONTIERS["f_khz"], FRONTIERS["bw_hz"], FRONTIERS["k"],
                           FRONTIERS["s"], FRONTIERS["w"])
    return float(shannon_bps(snr, FRONTIERS["bw_hz"]))


def mission(sensors: np.ndarray, hp: np.ndarray, lab: np.ndarray, k: int, rate_fn, d_n: int, rng, planned: bool) -> dict:
    """One mission with k AUVs. Returns AoI, energies, times."""
    # Split hovering points among AUVs by k-means on their positions (k <= number of HPs), then tour each group.
    if k == 1:
        groups = [list(range(hp.shape[0]))]
    else:
        _, g = kmeans(hp, k, rng)
        groups = [list(np.where(g == j)[0]) for j in range(k)]
    aoi_sum, n_served = 0.0, 0
    e_nav = e_hover = e_sensor = 0.0
    t_nav_total = t_hover_total = 0.0
    e_sensor_max = 0.0
    t_return_max = 0.0
    for grp in groups:
        if not grp:
            continue
        pts = hp[grp]
        order = plan_tour(pts, rng, planned)
        t = 0.0
        cur = DATA_CENTRE
        uploads = []  # (upload completion time) per sensor served by this AUV
        for idx in order:
            h = grp[idx]
            leg = float(np.linalg.norm(pts[idx] - cur))
            t += leg / SPEED
            t_nav_total += leg / SPEED
            cur = pts[idx]
            members = np.where(lab == h)[0]
            hover = HANDSHAKE_S
            for m in members:
                d = float(np.linalg.norm(sensors[m] - hp[h]))
                theta = d_n * PACKET_BITS / rate_fn(d) + d / C_SOUND
                hover += theta
                uploads.append(t + hover)
                e_tx = P_TX * d_n * PACKET_BITS / rate_fn(d)
                e_sensor += e_tx
                e_sensor_max = max(e_sensor_max, e_tx)
            t += hover
            t_hover_total += hover
        leg = float(np.linalg.norm(DATA_CENTRE - cur))
        t += leg / SPEED
        t_nav_total += leg / SPEED
        t_return_max = max(t_return_max, t)
        aoi_sum += sum(t - u for u in uploads)
        n_served += len(uploads)
    e_nav = P_NAV * t_nav_total
    e_hover = P_HOVER * t_hover_total
    return dict(aoi_mean_s=aoi_sum / n_served, t_nav_s=t_nav_total, t_hover_s=t_hover_total, mission_s=t_return_max,
                e_auv_j=e_nav + e_hover, e_nav_j=e_nav, e_hover_j=e_hover, e_sensor_j=e_sensor, e_sensor_max_j=e_sensor_max,
                lifetime_missions=(E_SENSOR0 / e_sensor_max if e_sensor_max > 0 else float("inf")))


def measured_links() -> dict:
    """Goodput per site (bit/s), distance-independent: FH-BFSK pooled ceilings and coherent ceilings when present."""
    links = {}
    # FH-BFSK, pooled over results/long (Black from the gate sweeps); ARQ goodput at the lowest rate where no rate passes.
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from summarise_coherent import fh_ceilings_pooled  # noqa: E402
    fh = fh_ceilings_pooled()
    for site, v in fh.items():
        # Goodput with retransmissions (payload rate times pooled success, best rate), the same basis as the coherent
        # rows and as Figure 2; until 2026-10-20 the FH rows used the PSR >= 0.9 ceiling where one existed (review).
        links[f"fh_{site}"] = dict(bps=v["arq_bps"], basis=f"FH-BFSK goodput with retransmissions at {v['arq_rate']} chips/s, pooled PSR {v['arq_psr']:.3f}")
    coh_path = os.path.join(RES, "coherent", "summary_coherent.json")
    if os.path.exists(coh_path):
        lad = json.load(open(coh_path, encoding="utf-8"))["ladder"]
        for site, v in lad.items():
            # ARQ goodput of the best scheme evaluated on every file of the site (rate times pooled success), the
            # same quantity the FH rows use where no rate reaches 0.9.
            links[f"coh_{site}"] = dict(bps=v["coherent_arq_bps"],
                                        basis=f"coherent ARQ goodput, {v['coherent_arq_scheme']} at pooled PSR {v['coherent_arq_psr']:.3f} over {v['n_files']} files")
    return links


def main() -> None:
    rng = np.random.default_rng(SEED)
    links = {"paper": dict(bps=None, basis="Shannon capacity of the paper's link budget at the sensor-to-HP distance")}
    links.update(measured_links())
    layouts = []
    for _ in range(N_LAYOUTS):
        s = rng.random((N_SENSORS, 3)) * np.array(AREA)
        hp, lab = kmeans(s, N_CLUSTERS, rng)
        layouts.append((s, hp, lab))
    out = dict(params=dict(n_sensors=N_SENSORS, n_clusters=N_CLUSTERS, area_m=AREA, speed_mps=SPEED, packet_bits=PACKET_BITS,
                           p_nav_w=P_NAV, p_hover_w=P_HOVER, p_tx_w=P_TX, e_sensor0_j=E_SENSOR0, n_layouts=N_LAYOUTS, d_n_list=D_N_LIST,
                           paper_rate_bps_at={d: paper_rate_bps(d) for d in (100, 500, 1000)}),
               links=links, results={})
    for lname, link in links.items():
        if link["bps"] is not None and link["bps"] <= 0:
            continue
        rate_fn = paper_rate_bps if link["bps"] is None else (lambda d, b=link["bps"]: b)
        out["results"][lname] = {}
        for d_n in D_N_LIST:
            rows = []
            for k in K_LIST:
                per = {key: [] for key in ("aoi_mean_s", "t_nav_s", "t_hover_s", "mission_s", "e_auv_j", "e_sensor_j", "lifetime_missions", "aoi_random_s")}
                for s, hp, lab in layouts:
                    lrng = np.random.default_rng([SEED, k, d_n])
                    m = mission(s, hp, lab, k, rate_fn, d_n, lrng, planned=True)
                    mr = mission(s, hp, lab, k, rate_fn, d_n, np.random.default_rng([SEED, k, d_n, 1]), planned=False)
                    for key in per:
                        per[key].append(mr["aoi_mean_s"] if key == "aoi_random_s" else m[key])
                row = {key: float(np.mean(v)) for key, v in per.items()}
                row["k"] = k
                row["hover_share"] = row["t_hover_s"] / (row["t_hover_s"] + row["t_nav_s"])
                row["planning_gain_pct"] = 100 * (row["aoi_random_s"] - row["aoi_mean_s"]) / row["aoi_random_s"]
                rows.append(row)
            # Operating K: nearest to the utopian point of the (AoI, AUV energy) front, each axis normalised to its range.
            a = np.array([r["aoi_mean_s"] for r in rows])
            e = np.array([r["e_auv_j"] for r in rows])
            an = (a - a.min()) / max(a.max() - a.min(), 1e-9)
            en = (e - e.min()) / max(e.max() - e.min(), 1e-9)
            k_star = int(K_LIST[int(np.argmin(np.hypot(an, en)))])
            out["results"][lname][str(d_n)] = dict(rows=rows, k_star=k_star)
            r5 = rows[4]
            print(f"{lname:10s} D_n {d_n:3d}: K*={k_star:2d} | at K=5: AoI {r5['aoi_mean_s']:8.0f} s, hover share {100 * r5['hover_share']:5.1f}%, "
                  f"mission {r5['mission_s']:8.0f} s, AUV energy {r5['e_auv_j'] / 1e3:7.1f} kJ, sensor energy {r5['e_sensor_j']:9.1f} J, "
                  f"lifetime {r5['lifetime_missions']:9.0f} missions, planning gain {r5['planning_gain_pct']:5.1f}%", flush=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
