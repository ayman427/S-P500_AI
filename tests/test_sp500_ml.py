import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from dashboard_server import _project_python
from vercel_app import app as vercel_app
from sp500_ml import (
    FUNDAMENTAL_FEATURES,
    FEATURES,
    PRICE_COLUMN_ORDER,
    PRICE_FEATURES,
    PREDICTIONS_PATH,
    _align_fundamental_events,
    _atomic_write_prices,
    _drop_incomplete_session,
    _extract_ticker,
    _final_holdout_metrics,
    _full_refresh_due,
    _extract_headline_relationships,
    _fundamental_events_from_companyfacts,
    _get_sec_user_agent,
    _load_fundamentals,
    _load_training_fundamentals,
    _merge_price_history,
    _metrics_for_console,
    _model_artifact_paths,
    _new_model,
    _parse_yahoo_news,
    _prediction_output_path,
    _parse_wikipedia_entity_context,
    _paper_ledger_path,
    _paper_summary,
    _save_news_cache,
    _settle_paper_rows,
    _summarize_universe_comparisons,
    _trade_metrics,
    _trade_results,
    paper_trade,
    update_data,
    _walk_forward_intraday_comparison,
    _walk_forward_metrics,
    build_features,
    compare_all_models,
    paper_status,
)


def make_prices(dates: pd.DatetimeIndex, start: float) -> pd.DataFrame:
    close = np.arange(start, start + len(dates), dtype=float)
    return pd.DataFrame(
        {
            "Open": close - 0.25,
            "High": close + 1,
            "Low": close - 1,
            "Close": close,
            "Volume": np.arange(1_000, 1_000 + len(dates)),
        },
        index=dates,
    )


class BuildFeaturesTests(unittest.TestCase):
    def test_vercel_asgi_entrypoint_exports_dashboard_routes(self) -> None:
        route_paths = {route.path for route in vercel_app.routes}

        self.assertTrue({"/api/state", "/api/quote", "/api/action/{action}"}.issubset(route_paths))

        config = json.loads(
            (Path(__file__).resolve().parents[1] / "vercel.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(config["services"]["app"]["entrypoint"], "vercel_app:app")
        self.assertEqual(config["rewrites"][-1]["destination"]["service"], "frontend")

    def test_extract_ticker_uses_deterministic_column_order(self) -> None:
        dates = pd.date_range("2024-01-02", periods=3)
        # Build the raw download with columns in a shuffled, non-alphabetical order.
        download = pd.DataFrame(
            {
                "Volume": [100, 200, 300],
                "Close": [10.0, 11.0, 12.0],
                "Open": [9.5, 10.5, 11.5],
                "High": [10.5, 11.5, 12.5],
                "Low": [9.0, 10.0, 11.0],
            },
            index=dates,
        )

        extracted = _extract_ticker(download, "AMD")

        self.assertEqual(list(extracted.columns), PRICE_COLUMN_ORDER)

    def test_merge_price_history_flags_split_adjustment_drift(self) -> None:
        dates = pd.date_range("2024-01-02", periods=3)
        old = make_prices(dates, 100)
        # Simulate a 2:1 split adjustment retroactively halving old closes.
        fresh = old.copy()
        fresh["Close"] = fresh["Close"] / 2

        _, drifted = _merge_price_history(old, fresh)

        self.assertTrue(drifted)

    def test_merge_price_history_does_not_flag_unchanged_overlap(self) -> None:
        dates = pd.date_range("2024-01-02", periods=3)
        old = make_prices(dates, 100)
        fresh = old.copy()

        merged, drifted = _merge_price_history(old, fresh)

        self.assertFalse(drifted)
        self.assertEqual(len(merged), len(dates))

    def test_atomic_price_write_preserves_old_file_on_write_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "AMD.csv"
            path.write_text("old-price-data", encoding="utf-8")
            prices = make_prices(pd.bdate_range("2024-01-02", periods=3), 100)
            with patch.object(prices, "to_csv", side_effect=OSError("disk full")):
                with self.assertRaisesRegex(OSError, "disk full"):
                    _atomic_write_prices(prices, path)

            self.assertEqual(path.read_text(encoding="utf-8"), "old-price-data")
            self.assertEqual(list(Path(directory).glob("*.tmp")), [])

    def test_full_price_refresh_is_due_after_six_months(self) -> None:
        now = pd.Timestamp("2026-09-29", tz="UTC")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "AMD.csv"
            path.touch()
            old_time = (now - pd.Timedelta(days=181)).timestamp()
            os.utime(path, (old_time, old_time))
            self.assertTrue(_full_refresh_due(path, now))

            recent_time = (now - pd.Timedelta(days=179)).timestamp()
            os.utime(path, (recent_time, recent_time))
            self.assertFalse(_full_refresh_due(path, now))

            marker = path.with_suffix(".full_refresh")
            marker.write_text((now - pd.Timedelta(days=181)).isoformat())
            os.utime(path, (now.timestamp(), now.timestamp()))
            self.assertTrue(_full_refresh_due(path, now))

            marker.write_text((now - pd.Timedelta(days=179)).isoformat())
            os.utime(path, (old_time, old_time))
            self.assertFalse(_full_refresh_due(path, now))

    def test_failed_drift_rerefresh_does_not_write_mixed_adjusted_history(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data_dir = Path(directory)
            dates = pd.bdate_range(
                end=pd.Timestamp.now().normalize() - pd.Timedelta(days=20), periods=4
            )
            old_prices = make_prices(dates, 100)
            old_path = data_dir / "AMD.csv"
            old_prices.to_csv(old_path, index_label="Date", date_format="%Y-%m-%d")
            make_prices(dates, 400).to_csv(
                data_dir / "SPY.csv", index_label="Date", date_format="%Y-%m-%d"
            )
            original_bytes = old_path.read_bytes()
            fresh_prices = old_prices.copy()
            fresh_prices["Close"] = fresh_prices["Close"] * 0.5
            market_prices = make_prices(dates, 400)

            with patch("sp500_ml.data.DATA_DIR", data_dir), patch(
                "sp500_ml.data.get_sp500_tickers", return_value=["AMD"]
            ), patch(
                "sp500_ml.data._download_batch",
                side_effect=[pd.DataFrame(), RuntimeError("full refresh failed")],
            ), patch(
                "sp500_ml.data._extract_ticker",
                side_effect=lambda _download, ticker: fresh_prices if ticker == "AMD" else market_prices,
            ):
                update_data()

            self.assertEqual(old_path.read_bytes(), original_bytes)

    def test_paper_trade_persists_signal_decision_and_never_replaces_news(self) -> None:
        prediction = {
            "ticker": "AMD",
            "as_of": "2026-09-28",
            "last_close": 100.0,
            "predicted_return": 0.02,
            "predicted_next_close": 102.0,
        }
        with tempfile.TemporaryDirectory() as directory:
            ledger_dir = Path(directory)
            prices_path = ledger_dir / "AMD-prices.csv"
            pd.DataFrame(
                {"Open": [100.0], "Close": [100.0]},
                index=pd.DatetimeIndex([prediction["as_of"]]),
            ).to_csv(prices_path, index_label="Date")
            headlines = [{
                "published_at_utc": "2026-09-28T12:00:00+00:00",
                "publisher": "Example",
                "title": "Original headline",
                "url": "https://example.com/original",
                "relationships": [],
            }]
            with patch("sp500_ml.paper.PAPER_DIR", ledger_dir), patch(
                "sp500_ml.paper.ROOT", ledger_dir
            ), patch(
                "sp500_ml.paper._load_latest_prediction",
                return_value=(prediction, "0123456789abcdef"),
            ), patch("sp500_ml.paper._fetch_recent_news", return_value=headlines), patch(
                "sp500_ml.paper._save_news_cache"
            ), patch("sp500_ml.paper._ticker_file", return_value=prices_path):
                paper_trade("AMD", entry_threshold=0.005)
                headlines[:] = [{
                    "published_at_utc": "2026-09-29T12:00:00+00:00",
                    "publisher": "Example",
                    "title": "Changed headline",
                    "url": "https://example.com/changed",
                    "relationships": [],
                }]
                paper_trade("AMD", entry_threshold=0.005)

            ledger_path = ledger_dir / "AMD_hist_gradient_boosting_0123456789_v2.csv"
            ledger = pd.read_csv(ledger_path)

        self.assertTrue(bool(ledger.loc[0, "enter_long"]))
        self.assertAlmostEqual(ledger.loc[0, "predicted_return"], 0.02)
        self.assertNotIn("predicted_next_close", ledger.columns)
        self.assertEqual(json.loads(ledger.loc[0, "news_context_json"])[0]["title"], "Original headline")

    def test_drop_incomplete_session_trims_todays_partial_bar_during_market_hours(
        self,
    ) -> None:
        dates = pd.bdate_range("2024-01-02", periods=5)
        prices = make_prices(dates, 100)
        today = pd.Timestamp.now(tz=ZoneInfo("America/New_York")).normalize().tz_localize(None)
        prices.index = list(dates[:-1]) + [today]
        market_open = pd.Timestamp.now(tz=ZoneInfo("America/New_York")).normalize() + pd.Timedelta(
            hours=12
        )

        trimmed = _drop_incomplete_session(prices, now=market_open)

        self.assertEqual(len(trimmed), len(prices) - 1)
        self.assertNotIn(today, trimmed.index)

    def test_fundamental_events_merge_concepts_and_derive_q4(self) -> None:
        def entry(start, end, filed, value, form="10-Q"):
            return {"start": start, "end": end, "filed": filed, "val": value, "form": form}

        facts = {
            "facts": {
                "us-gaap": {
                    # Q1-Q3 reported under the old tag, annual filed under the new tag.
                    "Revenues": {
                        "units": {
                            "USD": [
                                entry("2024-01-01", "2024-03-31", "2024-04-15", 100),
                                entry("2024-04-01", "2024-06-30", "2024-07-15", 110),
                                entry("2024-07-01", "2024-09-30", "2024-10-15", 120),
                            ]
                        }
                    },
                    "RevenueFromContractWithCustomerExcludingAssessedTax": {
                        "units": {
                            "USD": [
                                entry(
                                    "2024-01-01",
                                    "2024-12-31",
                                    "2025-02-15",
                                    460,
                                    form="10-K",
                                ),
                            ]
                        }
                    },
                }
            }
        }

        events = _fundamental_events_from_companyfacts(facts)

        derived_q4_date = pd.Timestamp("2025-02-16")
        self.assertIn(derived_q4_date, events.index)
        self.assertAlmostEqual(
            events.loc[derived_q4_date, "fund_revenue_log"], np.sign(130) * np.log1p(130)
        )

    def test_metrics_for_console_hides_holdout_unless_revealed(self) -> None:
        metrics = {"ticker": "AMD"}
        holdout = {"mae": 1.23}

        hidden = _metrics_for_console(metrics, holdout, reveal_holdout=False)
        revealed = _metrics_for_console(metrics, holdout, reveal_holdout=True)

        self.assertNotIn("final_holdout", hidden)
        self.assertEqual(revealed["final_holdout"], holdout)
        self.assertNotIn("final_holdout", metrics)

    def test_feature_groups_are_explicit_and_partition_features(self) -> None:
        self.assertEqual(PRICE_FEATURES + FUNDAMENTAL_FEATURES, FEATURES)
        self.assertEqual(len(PRICE_FEATURES), 15)
        self.assertEqual(len(FUNDAMENTAL_FEATURES), 7)

    def test_hist_gradient_boosting_uses_absolute_error_loss(self) -> None:
        self.assertEqual(_new_model("hist_gradient_boosting").loss, "absolute_error")

    def test_final_holdout_reports_return_space_mae(self) -> None:
        dates = pd.bdate_range("2023-01-01", periods=360)
        features = build_features(make_prices(dates, 100), make_prices(dates, 400))

        metrics, _ = _final_holdout_metrics(features, holdout_sessions=60)

        self.assertIn("mean_absolute_return_error", metrics)
        self.assertIn("previous_close_baseline_mean_absolute_return_error", metrics)
        self.assertLess(metrics["mean_absolute_return_error"], 1)

    def test_compare_all_skips_data_errors_but_raises_real_bugs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data_dir = Path(directory)
            dates = pd.bdate_range("2024-01-01", periods=30)
            for symbol in ("AAA", "SPY"):
                make_prices(dates, 100).to_csv(
                    data_dir / f"{symbol}.csv", index_label="Date", date_format="%Y-%m-%d"
                )
            with patch("sp500_ml.data.DATA_DIR", data_dir), patch(
                "sp500_ml.backtest.MODEL_DIR", data_dir
            ):
                with self.assertRaisesRegex(RuntimeError, "No tickers had sufficient history"):
                    compare_all_models()
                with patch(
                    "sp500_ml.backtest.build_features", side_effect=KeyError("bug")
                ):
                    with self.assertRaises(KeyError):
                        compare_all_models()

    def test_paper_status_settles_pending_rows_from_local_prices(self) -> None:
        dates = pd.bdate_range("2026-01-05", periods=3)
        ledger = pd.DataFrame(
            [{
                "signal_date": dates[0].date().isoformat(),
                "predicted_return": 0.03,
                "transaction_cost_bps_per_side": 10.0,
                "signal_edge_after_costs": 0.028,
                "enter_long": True,
                "status": "pending",
            }]
        )
        with tempfile.TemporaryDirectory() as directory:
            ledger_path = Path(directory) / "ledger.csv"
            prices_path = Path(directory) / "prices.csv"
            ledger.to_csv(ledger_path, index=False)
            pd.DataFrame(
                {"Open": [100.0, 101.0, 102.0], "Close": [100.0, 102.0, 103.0]},
                index=dates,
            ).to_csv(prices_path, index_label="Date")
            with patch("sp500_ml.paper._paper_ledger_path", return_value=ledger_path), patch(
                "sp500_ml.paper._ticker_file", return_value=prices_path
            ), patch("sp500_ml.paper._drop_incomplete_session", side_effect=lambda p: p):
                paper_status("AMD")
            saved = pd.read_csv(ledger_path)

        self.assertEqual(saved.loc[0, "status"], "settled")

    def test_equal_priority_overlapping_tags_resolve_deterministically(self) -> None:
        def entry(value):
            return {
                "start": "2024-01-01", "end": "2024-03-31",
                "filed": "2024-04-15", "val": value, "form": "10-Q",
            }

        facts = {
            "facts": {"us-gaap": {
                "Revenues": {"units": {"USD": [entry(100)]}},
                "RevenueFromContractWithCustomerExcludingAssessedTax": {
                    "units": {"USD": [entry(120)]}
                },
            }}
        }

        events = _fundamental_events_from_companyfacts(facts)

        # The preferred (first-listed) tag wins the overlap.
        self.assertAlmostEqual(
            events.iloc[0]["fund_revenue_log"], np.log1p(120)
        )

    def test_q4_skip_is_logged(self) -> None:
        def entry(start, end, filed, value, form="10-Q"):
            return {"start": start, "end": end, "filed": filed, "val": value, "form": form}

        facts = {"facts": {"us-gaap": {"Revenues": {"units": {"USD": [
            entry("2024-01-01", "2024-03-31", "2024-04-15", 100),
            entry("2024-01-01", "2024-12-31", "2025-02-15", 460, form="10-K"),
        ]}}}}}

        with patch("builtins.print") as printed:
            _fundamental_events_from_companyfacts(facts, "AMD")

        self.assertIn("Q4 derivation skipped for AMD", printed.call_args_list[0].args[0])

    def test_ridge_predictions_do_not_replace_hist_gradient_boosting_output(self) -> None:
        self.assertEqual(
            _prediction_output_path("hist_gradient_boosting"), PREDICTIONS_PATH
        )
        self.assertEqual(
            _prediction_output_path("ridge"),
            PREDICTIONS_PATH.with_name("predictions_ridge.csv"),
        )

    def test_model_artifacts_are_separate_and_keep_legacy_hist_paths(self) -> None:
        hist_paths = _model_artifact_paths("AMD", "hist_gradient_boosting")
        ridge_paths = _model_artifact_paths("AMD", "ridge")

        self.assertEqual(hist_paths[0].name, "AMD_next_close.joblib")
        self.assertEqual(hist_paths[1].name, "AMD_metrics.json")
        self.assertEqual(hist_paths[2].name, "AMD_final_holdout_trades.csv")
        self.assertTrue(
            all(hist != ridge for hist, ridge in zip(hist_paths, ridge_paths))
        )
        self.assertEqual(ridge_paths[0].name, "AMD_ridge_next_close.joblib")

    def test_new_paper_ledger_schema_uses_separate_versioned_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory, patch(
            "sp500_ml.paper.PAPER_DIR", Path(directory)
        ):
            legacy_path = Path(directory) / "AMD.csv"
            pd.DataFrame({"model_sha256": ["fingerprint"]}).to_csv(
                legacy_path, index=False
            )

            new_path = _paper_ledger_path(
                "AMD", "fingerprint", schema_version=2
            )
            self.assertTrue(legacy_path.exists())

        self.assertEqual(new_path.name, "AMD_hist_gradient_boosting_fingerprin_v2.csv")

    def test_ridge_factory_uses_requested_alpha(self) -> None:
        model = _new_model("ridge", ridge_alpha=2.5)

        self.assertEqual(model.named_steps["ridge"].alpha, 2.5)

    def test_universe_summary_aggregates_candidate_metrics_across_tickers(self) -> None:
        ticker_results = {
            "AAA": {
                "candidates": {
                    "ridge": {
                        "return_mae": 0.01,
                        "directional_accuracy": 0.52,
                        "trades": 10,
                        "wins": 6,
                        "losses": 4,
                        "win_rate": 0.6,
                        "net_return": 0.1,
                    }
                }
            },
            "BBB": {
                "candidates": {
                    "ridge": {
                        "return_mae": 0.03,
                        "directional_accuracy": 0.48,
                        "trades": 10,
                        "wins": 4,
                        "losses": 6,
                        "win_rate": 0.4,
                        "net_return": -0.1,
                    }
                }
            },
        }

        summary = _summarize_universe_comparisons(ticker_results)["ridge"]

        self.assertEqual(summary["tickers_compared"], 2)
        self.assertEqual(summary["total_trades"], 20)
        self.assertEqual(summary["wins"], 10)
        self.assertEqual(summary["losses"], 10)
        self.assertEqual(summary["pooled_win_rate"], 0.5)
        self.assertEqual(summary["profitable_tickers"], 1)
        self.assertAlmostEqual(summary["mean_ticker_return_mae"], 0.02)

    def test_dashboard_actions_prefer_project_virtual_environment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            python_path = Path(directory) / ".venv" / "Scripts" / "python.exe"
            python_path.parent.mkdir(parents=True)
            python_path.touch()
            with patch("dashboard_server.ROOT", Path(directory)):
                self.assertEqual(_project_python(), str(python_path))

    def test_retrained_model_gets_separate_paper_ledger(self) -> None:
        with tempfile.TemporaryDirectory() as directory, patch(
            "sp500_ml.paper.PAPER_DIR", Path(directory)
        ):
            existing_path = Path(directory) / "AMD.csv"
            pd.DataFrame({"model_sha256": ["old-model"]}).to_csv(
                existing_path, index=False
            )

            active_path = _paper_ledger_path("AMD", "new-model-fingerprint")

            self.assertEqual(
                active_path.name,
                f"AMD_{'new-model-fingerprint'[:10]}.csv",
            )
            self.assertTrue(existing_path.exists())

    def test_matching_model_keeps_existing_paper_ledger(self) -> None:
        with tempfile.TemporaryDirectory() as directory, patch(
            "sp500_ml.paper.PAPER_DIR", Path(directory)
        ):
            existing_path = Path(directory) / "AMD.csv"
            pd.DataFrame({"model_sha256": ["same-model"]}).to_csv(
                existing_path, index=False
            )

            active_path = _paper_ledger_path("AMD", "same-model")

            self.assertEqual(active_path, existing_path)

    def test_news_cache_is_saved_per_ticker_without_model_artifacts(self) -> None:
        headlines = [{"title": "CCI company update", "url": "https://example.com/cci"}]
        with tempfile.TemporaryDirectory() as directory, patch(
            "sp500_ml.news.NEWS_DIR", Path(directory)
        ):
            payload = _save_news_cache("CCI", headlines)
            saved = json.loads(
                (Path(directory) / "CCI.json").read_text(encoding="utf-8")
            )

        self.assertEqual(payload["ticker"], "CCI")
        self.assertEqual(saved["headlines"], headlines)
        self.assertTrue(saved["fetched_at_utc"])

    def test_sec_user_agent_reads_saved_value_when_process_environment_is_empty(self) -> None:
        with patch("sp500_ml.fundamentals.os.environ", {}), patch(
            "sp500_ml.fundamentals._read_windows_sec_user_agent", return_value="saved app contact"
        ) as read_saved:
            value = _get_sec_user_agent()

        self.assertEqual(value, "saved app contact")
        read_saved.assert_called_once_with()

    def test_process_sec_user_agent_takes_precedence(self) -> None:
        with patch(
            "sp500_ml.fundamentals.os.environ", {"SEC_USER_AGENT": "process app contact"}
        ), patch("sp500_ml.fundamentals._read_windows_sec_user_agent") as read_saved:
            value = _get_sec_user_agent()

        self.assertEqual(value, "process app contact")
        read_saved.assert_not_called()

    def test_training_refreshes_sec_fundamentals_before_reading(self) -> None:
        dates = pd.bdate_range("2024-01-01", periods=2)
        cached = pd.DataFrame(
            {feature: [1.0] for feature in FUNDAMENTAL_FEATURES},
            index=pd.DatetimeIndex([dates[0]]),
        )
        cache_path = Mock()
        cache_path.exists.return_value = False
        with patch("sp500_ml.fundamentals._fundamental_file", return_value=cache_path), patch(
            "sp500_ml.fundamentals.update_fundamentals"
        ) as refresh, patch("sp500_ml.fundamentals._read_fundamentals", return_value=cached) as read:
            fundamentals, warning = _load_training_fundamentals("AMD", dates)

        refresh.assert_called_once_with("AMD")
        read.assert_called_once_with("AMD", dates)
        self.assertIs(fundamentals, cached)
        self.assertIsNone(warning)

    def test_fresh_fundamentals_cache_skips_refresh(self) -> None:
        dates = pd.bdate_range("2024-01-01", periods=2)
        cached = pd.DataFrame(
            {feature: [1.0] for feature in FUNDAMENTAL_FEATURES},
            index=pd.DatetimeIndex([dates[0]]),
        )
        cache_path = Mock()
        cache_path.exists.return_value = True
        cache_path.stat.return_value.st_mtime = pd.Timestamp.now(tz="UTC").timestamp() - 3600
        with patch("sp500_ml.fundamentals._fundamental_file", return_value=cache_path), patch(
            "sp500_ml.fundamentals.update_fundamentals"
        ) as refresh, patch("sp500_ml.fundamentals._read_fundamentals", return_value=cached):
            fundamentals, warning = _load_training_fundamentals("AMD", dates)

        refresh.assert_not_called()
        self.assertIs(fundamentals, cached)
        self.assertIsNone(warning)

    def test_training_uses_cached_sec_fundamentals_when_refresh_fails(self) -> None:
        dates = pd.bdate_range("2024-01-01", periods=2)
        cached = pd.DataFrame(
            {feature: [1.0] for feature in FUNDAMENTAL_FEATURES},
            index=pd.DatetimeIndex([dates[0]]),
        )
        cache_path = Mock()
        cache_path.exists.return_value = True
        cache_path.stat.return_value.st_mtime = 0
        with patch("sp500_ml.fundamentals._fundamental_file", return_value=cache_path), patch(
            "sp500_ml.fundamentals.update_fundamentals",
            side_effect=RuntimeError("SEC_USER_AGENT is missing"),
        ), patch("sp500_ml.fundamentals._read_fundamentals", return_value=cached):
            fundamentals, warning = _load_training_fundamentals("AMD", dates)

        self.assertIs(fundamentals, cached)
        self.assertIn("using cached fundamentals", warning)

    def test_training_without_cache_falls_back_when_sec_refresh_fails(self) -> None:
        dates = pd.bdate_range("2024-01-01", periods=2)
        cache_path = Mock()
        cache_path.exists.return_value = False
        with patch("sp500_ml.fundamentals._fundamental_file", return_value=cache_path), patch(
            "sp500_ml.fundamentals.update_fundamentals",
            side_effect=RuntimeError("SEC_USER_AGENT is missing"),
        ), patch("sp500_ml.fundamentals._read_fundamentals") as read:
            fundamentals, warning = _load_training_fundamentals("AMD", dates)

        self.assertIsNone(fundamentals)
        self.assertIn("training without them", warning)
        read.assert_not_called()

    def test_target_is_next_session_close(self) -> None:
        dates = pd.bdate_range("2024-01-01", periods=120)
        prices = make_prices(dates, 100)
        market = make_prices(dates, 400)

        result = build_features(prices, market)
        expected = prices["Close"].iloc[81]

        self.assertEqual(result.loc[dates[80], "next_close"], expected)
        self.assertEqual(result.index.max(), dates[-1])
        self.assertTrue(result["next_close"].tail(1).isna().all())

    def test_feature_rows_do_not_contain_non_finite_values(self) -> None:
        dates = pd.bdate_range("2024-01-01", periods=120)
        prices = make_prices(dates, 100)
        market = make_prices(dates, 400)

        result = build_features(prices, market)

        self.assertTrue(result.dropna(subset=["next_close"])[PRICE_FEATURES].notna().all().all())
        self.assertTrue(result[PRICE_FEATURES].notna().all().all())
        self.assertTrue((result["volatility_20d"] >= 0).all())

    def test_fundamental_availability_is_never_the_filing_day_itself(self) -> None:
        # A same-day available_date would leak same-session fundamentals into training.
        facts = {
            "facts": {
                "us-gaap": {
                    "Revenues": {
                        "units": {
                            "USD": [
                                {
                                    "form": "10-Q",
                                    "filed": "2024-04-30",
                                    "start": "2024-01-01",
                                    "end": "2024-03-31",
                                    "val": 100,
                                }
                            ]
                        }
                    },
                }
            }
        }

        events = _fundamental_events_from_companyfacts(facts)

        self.assertNotIn(pd.Timestamp("2024-04-30"), events.index)
        self.assertIn(pd.Timestamp("2024-05-01"), events.index)

    def test_q4_derivation_ignores_quarters_filed_after_annual_report(self) -> None:
        def entry(start, end, filed, value, form="10-Q"):
            return {"start": start, "end": end, "filed": filed, "val": value, "form": form}

        facts = {
            "facts": {
                "us-gaap": {
                    "Revenues": {
                        "units": {
                            "USD": [
                                entry("2024-01-01", "2024-03-31", "2024-04-15", 100),
                                entry("2024-04-01", "2024-06-30", "2024-07-15", 110),
                                entry("2024-07-01", "2024-09-30", "2025-03-01", 120),
                            ]
                        }
                    },
                    "RevenueFromContractWithCustomerExcludingAssessedTax": {
                        "units": {
                            "USD": [
                                entry(
                                    "2024-01-01", "2024-12-31", "2025-02-15", 460,
                                    form="10-K",
                                )
                            ]
                        }
                    },
                }
            }
        }

        events = _fundamental_events_from_companyfacts(facts)

        self.assertNotIn(pd.Timestamp("2025-02-16"), events.index)

    def test_prediction_fails_closed_when_trained_fundamentals_are_unavailable(self) -> None:
        cache_path = Mock()
        cache_path.exists.return_value = False
        with patch("sp500_ml.fundamentals._fundamental_file", return_value=cache_path), patch(
            "sp500_ml.fundamentals.update_fundamentals",
            side_effect=RuntimeError("SEC_USER_AGENT is missing"),
        ):
            with self.assertRaisesRegex(RuntimeError, "requires SEC fundamentals"):
                _load_fundamentals("AMD", pd.bdate_range("2026-01-01", periods=2))

    def test_walk_forward_folds_never_train_on_future_validation_dates(self) -> None:
        dates = pd.bdate_range("2024-01-01", periods=300)
        features = build_features(make_prices(dates, 100), make_prices(dates, 400))
        boundaries = []

        class _RecordingStub:
            def fit(self, X, y):
                boundaries.append({"train_max": X.index.max()})
                return self

            def predict(self, X):
                boundaries[-1]["validation_min"] = X.index.min()
                return np.zeros(len(X))

        with patch("sp500_ml.backtest._new_model", return_value=_RecordingStub()):
            _walk_forward_metrics(features)

        self.assertEqual(len(boundaries), 3)
        for fold in boundaries:
            self.assertLess(fold["train_max"], fold["validation_min"])

    def test_sec_fundamentals_only_become_available_after_filing(self) -> None:
        observation = {
            "form": "10-Q",
            "filed": "2024-04-30",
            "start": "2024-01-01",
            "end": "2024-03-31",
            "val": 100,
        }
        instant_observation = {
            "form": "10-Q",
            "filed": "2024-04-30",
            "end": "2024-03-31",
            "val": 500,
        }
        facts = {
            "facts": {
                "us-gaap": {
                    "Revenues": {"units": {"USD": [observation]}},
                    "NetIncomeLoss": {"units": {"USD": [observation | {"val": 10}]}},
                    "OperatingIncomeLoss": {"units": {"USD": [observation | {"val": 20}]}},
                    "Assets": {"units": {"USD": [instant_observation]}},
                    "Liabilities": {"units": {"USD": [instant_observation | {"val": 300}]}},
                    "StockholdersEquity": {"units": {"USD": [instant_observation | {"val": 200}]}},
                }
            }
        }

        events = _fundamental_events_from_companyfacts(facts)
        dates = pd.DatetimeIndex(["2024-04-30", "2024-05-01", "2024-05-02"])
        aligned = _align_fundamental_events(events, dates)

        self.assertTrue(aligned.iloc[0].isna().all())
        self.assertAlmostEqual(aligned.iloc[1]["fund_net_margin"], 0.1)
        self.assertEqual(aligned.iloc[2]["fund_assets_log"], aligned.iloc[1]["fund_assets_log"])
        self.assertEqual(set(events.columns), set(FUNDAMENTAL_FEATURES))

    def test_walk_forward_metrics_use_three_ordered_folds(self) -> None:
        dates = pd.bdate_range("2024-01-01", periods=300)
        features = build_features(make_prices(dates, 100), make_prices(dates, 400))

        metrics = _walk_forward_metrics(features)

        folds = metrics["validation_folds"]
        self.assertEqual(len(folds), 3)
        self.assertEqual(
            [fold["validation_start"] for fold in folds],
            sorted(fold["validation_start"] for fold in folds),
        )
        self.assertGreater(metrics["validation_rows"], 0)

    def test_walk_forward_metrics_support_ridge_model(self) -> None:
        dates = pd.bdate_range("2024-01-01", periods=300)
        features = build_features(make_prices(dates, 100), make_prices(dates, 400))

        metrics = _walk_forward_metrics(features, model_type="ridge", ridge_alpha=1.0)

        self.assertEqual(metrics["model_type"], "ridge")
        self.assertEqual(len(metrics["validation_folds"]), 3)
        self.assertGreater(metrics["mean_absolute_return_error"], 0)

    def test_intraday_model_comparison_uses_three_development_folds(self) -> None:
        dates = pd.bdate_range("2024-01-01", periods=320)
        features = build_features(make_prices(dates, 100), make_prices(dates, 400))

        comparison = _walk_forward_intraday_comparison(features)

        self.assertEqual(
            set(comparison),
            {"ridge", "hist_gradient_boosting"},
        )
        self.assertEqual(
            len(comparison["ridge"]["validation_folds"]),
            3,
        )
        self.assertEqual(
            len(comparison["hist_gradient_boosting"]["validation_folds"]),
            3,
        )
        self.assertEqual(
            comparison["ridge"]["target"],
            "next-session open-to-close return",
        )
        self.assertGreater(comparison["ridge"]["validation_rows"], 0)

    def test_final_holdout_is_separate_and_writes_each_session(self) -> None:
        dates = pd.bdate_range("2023-01-01", periods=360)
        features = build_features(make_prices(dates, 100), make_prices(dates, 400))
        labeled_dates = features.dropna(subset=["next_close"]).index
        expected_start = labeled_dates[-60]

        metrics, trade_log = _final_holdout_metrics(features, holdout_sessions=60)

        self.assertEqual(metrics["period_start"], expected_start.date().isoformat())
        self.assertLess(metrics["training_through"], metrics["period_start"])
        self.assertEqual(metrics["sessions"], 60)
        self.assertEqual(len(trade_log), 60)
        self.assertEqual(metrics["trading"]["baselines"]["no_trade_cumulative_return"], 0.0)
        self.assertEqual(metrics["trading"]["baselines"]["always_long"]["trades"], 60)

    def test_walk_forward_handles_sparse_point_in_time_fundamentals(self) -> None:
        dates = pd.bdate_range("2024-01-01", periods=300)
        events = pd.DataFrame(
            {
                feature: [float(index + 1), float(index + 2)]
                for index, feature in enumerate(FUNDAMENTAL_FEATURES)
            },
            index=[dates[100], dates[200]],
        )
        features = build_features(
            make_prices(dates, 100), make_prices(dates, 400), events
        )

        metrics = _walk_forward_metrics(features)

        self.assertEqual(len(metrics["validation_folds"]), 3)

    def test_trade_metrics_count_net_wins_after_costs(self) -> None:
        validation = pd.DataFrame(
            {
                "next_open": [100.0, 100.0, 100.0, 100.0, 104.0],
                "next_close": [102.0, 99.0, 104.0, 100.0, 105.0],
            }
        )
        predicted_returns = np.array([0.02, 0.015, 0.009, 0.0, -0.01])

        metrics = _trade_metrics(predicted_returns, validation, 0.01, 10)

        self.assertEqual(metrics["trades"], 2)
        self.assertEqual(metrics["wins"], 1)
        self.assertEqual(metrics["losses"], 1)
        self.assertEqual(metrics["win_rate"], 0.5)
        self.assertAlmostEqual(metrics["mean_net_trade_return"], 0.003)
        self.assertFalse(metrics["more_wins_than_losses"])
        self.assertFalse(metrics["win_rate_confidently_above_50pct"])
        self.assertFalse(metrics["win_rate_confidently_below_50pct"])

    def test_entry_decision_uses_signal_return_not_next_open_gap(self) -> None:
        validation = pd.DataFrame(
            {
                "next_open": [150.0],
                "next_close": [151.0],
            },
            index=pd.DatetimeIndex(["2026-09-28"]),
        )

        trade = _trade_results(
            np.array([0.012]), validation, entry_threshold=0.005,
            transaction_cost_bps=10,
        )

        self.assertTrue(bool(trade.iloc[0]["enter_long"]))
        self.assertAlmostEqual(trade.iloc[0]["predicted_return"], 0.012)
        self.assertAlmostEqual(trade.iloc[0]["gross_open_to_close_return"], 1 / 150)

    def test_paper_ledger_settles_pending_signal_once_next_session_exists(self) -> None:
        dates = pd.bdate_range("2026-01-05", periods=3)
        ledger = pd.DataFrame(
            [
                {
                    "ticker": "TEST",
                    "signal_date": dates[0].date().isoformat(),
                    "predicted_return": 0.03,
                    "entry_threshold": 0.0,
                    "transaction_cost_bps_per_side": 10.0,
                    "enter_long": True,
                    "status": "pending",
                    "execution_date": np.nan,
                    "next_open": np.nan,
                    "next_close": np.nan,
                    "gross_open_to_close_return": np.nan,
                    "round_trip_cost_return": np.nan,
                    "net_return_if_traded": np.nan,
                    "strategy_daily_return": np.nan,
                }
            ]
        )
        prices = pd.DataFrame(
            {"Open": [100.0, 101.0, 102.0], "Close": [100.0, 102.0, 103.0]},
            index=dates,
        )

        settled = _settle_paper_rows(ledger, prices)
        summary = _paper_summary(settled)

        self.assertEqual(settled.loc[0, "status"], "settled")
        self.assertEqual(settled.loc[0, "execution_date"], dates[1].date().isoformat())
        self.assertTrue(settled.loc[0, "enter_long"])
        self.assertEqual(summary["trades"], 1)
        self.assertEqual(summary["wins"], 1)
        self.assertEqual(summary["pending_signals"], 0)

    def test_settlement_does_not_recalculate_logged_entry_decision(self) -> None:
        dates = pd.bdate_range("2026-01-05", periods=2)
        ledger = pd.DataFrame(
            [{
                "signal_date": dates[0].date().isoformat(),
                "predicted_return": -0.02,
                "enter_long": False,
                "transaction_cost_bps_per_side": 10.0,
                "status": "pending",
            }]
        )
        prices = pd.DataFrame(
            {"Open": [100.0, 90.0], "Close": [100.0, 99.0]}, index=dates
        )

        settled = _settle_paper_rows(ledger, prices)

        self.assertFalse(bool(settled.loc[0, "enter_long"]))
        self.assertEqual(settled.loc[0, "strategy_daily_return"], 0.0)

    def test_yahoo_news_parser_filters_wrong_ticker_and_future_articles(self) -> None:
        cutoff = pd.Timestamp("2026-09-28T20:00:00Z")
        payload = {
            "news": [
                {
                    "title": "AMD market update",
                    "publisher": "Example News",
                    "link": "https://example.com/amd",
                    "providerPublishTime": int(pd.Timestamp("2026-09-28T19:00:00Z").timestamp()),
                    "relatedTickers": ["AMD"],
                },
                {
                    "title": "AMD after-close announcement",
                    "publisher": "Example News",
                    "link": "https://example.com/after-close",
                    "providerPublishTime": int(pd.Timestamp("2026-09-28T20:05:00Z").timestamp()),
                    "relatedTickers": ["AMD"],
                },
                {
                    "title": "Other company announcement",
                    "publisher": "Example News",
                    "link": "https://example.com/other",
                    "providerPublishTime": int(pd.Timestamp("2026-09-28T19:30:00Z").timestamp()),
                    "relatedTickers": ["OTHER"],
                },
            ]
        }

        headlines = _parse_yahoo_news(payload, "AMD", now=cutoff)

        self.assertEqual(len(headlines), 1)
        self.assertEqual(headlines[0]["title"], "AMD market update")
        self.assertEqual(headlines[0]["published_at_utc"], "2026-09-28T19:00:00+00:00")

    def test_headline_relationship_extracts_candidate_with_evidence(self) -> None:
        relationships = _extract_headline_relationships(
            "AMD to acquire Fei-Fei Li’s World Labs for $8.2 billion", "AMD"
        )

        acquisition = next(item for item in relationships if item["predicate"] == "acquires")
        self.assertEqual(acquisition["subject_ticker"], "AMD")
        self.assertEqual(acquisition["counterparty_candidate"], "World Labs")
        self.assertIn("candidate requires source verification", acquisition["extraction_method"])

    def test_founded_company_headline_does_not_mistake_ai_for_person(self) -> None:
        relationships = _extract_headline_relationships(
            "AMD buys firm founded by AI 'godmother' Fei-Fei Li", "AMD"
        )

        acquisition = next(
            item for item in relationships
            if item["predicate"] == "acquires_company_founded_by"
        )
        self.assertEqual(acquisition["counterparty_candidate"], "Fei-Fei Li")
        self.assertFalse(any(item["predicate"] == "founded_by" for item in relationships))

    def test_editorial_bet_wording_is_not_labeled_as_an_investment(self) -> None:
        relationships = _extract_headline_relationships(
            "AMD’s $8.2B bet on World Labs is a direct shot at Nvidia", "AMD"
        )

        self.assertEqual(relationships[0]["predicate"], "bet_on")
        self.assertNotEqual(relationships[0]["predicate"], "invests_in")

    def test_entity_context_search_parser_requires_counterparty_terms(self) -> None:
        payload = {
            "query": {
                "search": [
                    {
                        "title": "Fei-Fei Li",
                        "snippet": "Li co-founded World Labs, an AI company.",
                        "pageid": 42,
                    },
                    {
                        "title": "World War II",
                        "snippet": "A world conflict.",
                        "pageid": 43,
                    },
                ]
            }
        }

        results = _parse_wikipedia_entity_context(payload, "World Labs")

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["title"], "Fei-Fei Li")
        self.assertIn("co-founded World Labs", results[0]["snippet"])


if __name__ == "__main__":
    unittest.main()