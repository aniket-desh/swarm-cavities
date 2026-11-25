"""analysis/poa_data.py: orientation and belief mapping on a tiny synthetic file, then the real file if present."""

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from analysis.poa_data import DATA_PATH, load_poa, summarize


def test_loader_transposes_J_and_maps_votes(tmp_path):
    N, T1, K = 3, 2, 5
    J_poa = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 0]])     # PoA: sender 0 -> receiver 1 with +1
    votes = np.ones((T1, N, K), dtype=int)
    votes[1, 0] = [1, -1, -1, 0, 0]                          # 3 parsed votes: mean -1/3
    votes[1, 2] = 0                                          # nothing parsed -> NaN
    spins = np.ones((T1, N), dtype=int)
    rows = dict(model=["m", "m"], mode=["objective"] * 2, statement=["q", "q"],
                J=[J_poa.tolist(), J_poa.T.tolist()], spins_history=[spins.tolist()] * 2,
                spins_raw_history=[votes.tolist()] * 2, replica=[0, 0],
                ground_truth=["A", "A"], political_lean=[None, None])
    path = tmp_path / "poa.parquet"
    pq.write_table(pa.table(rows), path)
    d = load_poa(path)[("m", "objective", "q")]
    assert d["J"].shape == (2, N, N)
    # Two distinct graphs; ours is receiver-row, so J_ours[1, 0] = J_poa[0, 1] = +1 (0 influences 1).
    Js = {d["J"][i].tobytes(): d["J"][i] for i in range(2)}
    assert J_poa.T.astype(np.int8).tobytes() in Js and J_poa.astype(np.int8).tobytes() in Js
    assert sorted(d["graph"].tolist()) == [0, 1]
    b = d["beliefs"][0]
    assert b[1, 0] == pytest.approx(-1 / 3) and np.isnan(b[1, 2]) and b[0, 1] == 1.0
    assert d["ground_truth"] == "A" and d["political_lean"] is None


@pytest.mark.skipif(not DATA_PATH.exists(), reason="PoA parquet not present")
def test_real_poa_file_summary():
    data = load_poa()
    s = summarize(data)
    assert s["communities"] == 9600 and s["N"] == 32 and s["rounds_T"] == 8 and s["votes_K"] == 5
    assert s["all_symmetric"] and s["zero_diagonal"]
    assert s["spins_equal_majority_of_votes"] == 1.0
    d = next(iter(data.values()))
    b, sp = d["beliefs"], d["spins"]
    ok = np.isfinite(b) & (b != 0)
    assert np.all(np.sign(b[ok]) == sp[ok])                  # majority sign agrees with the vote mean
