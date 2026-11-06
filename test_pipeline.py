import warnings
import pandas as pd
from src.analysis import run_all
from src.generate_data import generate
from src.viability import Assumptions, scorecard

warnings.filterwarnings("ignore")


def test_generator_has_noise_and_outliers():
    df = generate(n=300, seed=1, noise=1.0)
    assert df.isna().sum().sum() > 0
    assert df.duplicated(subset=[c for c in df.columns if c != "respondent_id"]).sum() > 0
    assert df["q07_clothing_spend_aed"].max() > 5 * df["q07_clothing_spend_aed"].median()


def test_pipeline_runs_and_beats_coin_flip():
    R = run_all(generate(n=400, seed=7))
    assert R["clf"]["metrics"]["CV AUC (5-fold)"].max() > 0.7
    assert R["reg"]["metrics"]["CV R2 (log AED)"].max() > 0.3
    assert set(R["clu"]["summary"].index) == {"High probability", "Medium probability", "Low probability"}
    assert len(R["asr"]["curve"]) == 8
    sc, n, verdict, _ = scorecard(R, Assumptions())
    assert len(sc) == 7 and 0 <= n <= 7


def test_cleaning_removes_extreme_outliers():
    raw = generate(n=400, seed=3)
    R = run_all(raw)
    assert R["data"]["q07_clothing_spend_aed"].max() < raw["q07_clothing_spend_aed"].max()
