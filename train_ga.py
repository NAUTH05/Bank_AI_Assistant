"""Chạy thí nghiệm GA tối ưu TF-IDF + Logistic Regression cho dự án DT51.

Ví dụ sử dụng::

    python train_ga.py --generations 12        # thí nghiệm chính thức
    python train_ga.py --generations 2         # smoke test (ghi vào ga_results_smoke/)

Script này thực thi **thật** toàn bộ quy trình: tái lập baseline, chạy GA với
đánh giá fitness bằng 7-fold CV trên tập Train, huấn luyện lại mô hình tốt nhất
trên toàn bộ Train, đánh giá trên tập Test và ghi toàn bộ bằng chứng vào
``ga_results/``.

Nguyên tắc bất biến:
    * Không dùng Test trong quá trình tối ưu (fitness chỉ nhận dữ liệu Train).
    * Không ghi đè ``Models/best_model.joblib`` của baseline.
    * Không sửa dữ liệu, nhãn intent hay hệ luật IF–THEN.
"""

from __future__ import annotations

import argparse
import platform
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

import genetic_algorithm as ga


METHODOLOGY = {
    "fitness": (
        "Macro F1 trung bình qua 7-fold StratifiedKFold (shuffle=True, "
        "random_state=45116) trên phân vùng Train; mỗi fold fit lại toàn bộ "
        "Pipeline (TF-IDF + Logistic Regression) nên không có TF-IDF fit toàn cục."
    ),
    "fold_policy": (
        "Các fold được tính một lần bằng StratifiedKFold trên tập Train và dùng "
        "chung cho mọi ứng viên, bảo đảm so sánh công bằng."
    ),
    "test_isolation": (
        "Tập Test chỉ được dùng sau khi GA kết thúc, để đánh giá mô hình cuối. "
        "FitnessEvaluator không nhận tham số Test."
    ),
    "selection": "Tournament selection, kích thước 3.",
    "crossover": "Uniform crossover, xác suất 0.80, mỗi gen lấy ngẫu nhiên từ bố hoặc mẹ.",
    "mutation": (
        "Đột biến độc lập theo từng gen với xác suất 0.01; giá trị mới được chọn "
        "trong không gian giá trị của gen và khác giá trị cũ."
    ),
    "elitism": "Giữ 2 cá thể tốt nhất mỗi thế hệ, không làm tăng kích thước quần thể.",
    "caching": "Mỗi nhiễm sắc thể chỉ được đánh giá một lần (cache theo tuple gen).",
    "early_stopping": (
        "Tuỳ chọn, mặc định tắt để chạy đủ 12 thế hệ đã đăng ký. Khi bật "
        "(--patience 5), dừng sau 5 thế hệ liên tiếp không cải thiện quá 1e-6."
    ),
}


# ---------------------------------------------------------------------------
# 1. Tham số dòng lệnh
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Thí nghiệm GA tối ưu pipeline TF-IDF + Logistic Regression (DT51).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--generations", type=int, default=ga.GA_GENERATIONS)
    parser.add_argument("--population-size", type=int, default=ga.GA_POPULATION_SIZE)
    parser.add_argument("--mutation-rate", type=float, default=ga.GA_MUTATION_RATE)
    parser.add_argument("--crossover-rate", type=float, default=ga.GA_CROSSOVER_RATE)
    parser.add_argument("--elite-size", type=int, default=ga.GA_ELITE_SIZE)
    parser.add_argument("--tournament-size", type=int, default=ga.GA_TOURNAMENT_SIZE)
    parser.add_argument("--seed", type=int, default=ga.SEED)
    parser.add_argument(
        "--patience",
        type=int,
        default=0,
        help=(
            "Bật dừng sớm: dừng sau N thế hệ liên tiếp không cải thiện quá 1e-6 "
            f"(0 = tắt, mặc định; dùng --patience {ga.GA_PATIENCE} cho cửa sổ đã đăng ký)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Thư mục ghi kết quả (mặc định ga_results, hoặc ga_results_smoke cho smoke test).",
    )
    parser.add_argument("--quiet", action="store_true", help="Ẩn log tiến trình.")
    return parser.parse_args(argv)


def resolve_output_dir(args: argparse.Namespace) -> Path:
    """Chọn thư mục kết quả; tách smoke test khỏi thí nghiệm chính thức."""
    if args.output_dir is not None:
        return Path(args.output_dir)
    if args.generations != ga.GA_GENERATIONS:
        return ga.BASE_DIR / "ga_results_smoke"
    return ga.RESULTS_DIR


def _relative_or_name(path: Path) -> str:
    """Đường dẫn tương đối so với gốc dự án (tránh ghi đường dẫn tuyệt đối)."""
    path = Path(path)
    try:
        return str(path.resolve().relative_to(ga.BASE_DIR))
    except ValueError:
        return path.name


# ---------------------------------------------------------------------------
# 2. Đánh giá trên tập Test
# ---------------------------------------------------------------------------


def evaluate_on_test(
    pipeline, test_questions: np.ndarray, test_labels: np.ndarray
) -> Dict[str, Any]:
    """Tính Accuracy, Macro F1, Weighted F1, báo cáo từng intent, ma trận nhầm lẫn."""
    predictions = pipeline.predict(test_questions)
    return {
        "accuracy": float(accuracy_score(test_labels, predictions)),
        "macro_f1": float(f1_score(test_labels, predictions, average="macro")),
        "weighted_f1": float(f1_score(test_labels, predictions, average="weighted")),
        "report": classification_report(
            test_labels,
            predictions,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(
            test_labels, predictions, labels=list(pipeline.classes_)
        ),
        "labels": [str(label) for label in pipeline.classes_],
        "predictions": predictions,
    }


def per_intent_frame(
    baseline_report: Dict[str, Any], ga_report: Dict[str, Any], labels: List[str]
) -> pd.DataFrame:
    """Bảng Precision/Recall/F1/support cho từng intent của hai mô hình."""
    rows = []
    for label in labels:
        base = baseline_report.get(label, {})
        optimised = ga_report.get(label, {})
        rows.append(
            {
                "intent": label,
                "support": int(base.get("support", 0)),
                "precision_baseline": round(float(base.get("precision", 0.0)), 4),
                "recall_baseline": round(float(base.get("recall", 0.0)), 4),
                "f1_baseline": round(float(base.get("f1-score", 0.0)), 4),
                "precision_ga": round(float(optimised.get("precision", 0.0)), 4),
                "recall_ga": round(float(optimised.get("recall", 0.0)), 4),
                "f1_ga": round(float(optimised.get("f1-score", 0.0)), 4),
                "delta_f1": round(
                    float(optimised.get("f1-score", 0.0)) - float(base.get("f1-score", 0.0)),
                    4,
                ),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3. Chương trình chính
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    output_dir = resolve_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)
    verbose = not args.quiet

    script_start = time.perf_counter()
    started_at = datetime.now().astimezone()

    print("=" * 78)
    print("THÍ NGHIỆM GA – TỐI ƯU TF-IDF + LOGISTIC REGRESSION (DT51)")
    print("=" * 78)

    # -- 3.1 Dữ liệu ------------------------------------------------------
    df = ga.load_dataset()
    df_sample, train_df, test_df = ga.split_dataset(
        df, seed=args.seed, data_ratio=ga.DATA_RATIO, test_size=ga.TEST_SIZE
    )
    train_X, train_y = ga.to_xy(train_df)
    test_X, test_y = ga.to_xy(test_df)

    print(
        f"Dữ liệu: {len(df)} câu / {df['intent'].nunique()} intent -> "
        f"mẫu {len(df_sample)} -> Train {len(train_df)} / Test {len(test_df)}"
    )
    print(f"Thư mục kết quả: {output_dir}")

    # -- 3.2 Cấu hình GA --------------------------------------------------
    config = ga.GAConfig(
        population_size=args.population_size,
        mutation_rate=args.mutation_rate,
        crossover_rate=args.crossover_rate,
        generations=args.generations,
        elite_size=args.elite_size,
        tournament_size=args.tournament_size,
        random_seed=args.seed,
        patience=args.patience if args.patience > 0 else None,
        verbose=verbose,
    )
    config.validate()

    registered_ok = (
        config.population_size == ga.GA_POPULATION_SIZE
        and abs(config.mutation_rate - ga.GA_MUTATION_RATE) < 1e-12
        and config.crossover_rate == ga.GA_CROSSOVER_RATE
        and config.elite_size == ga.GA_ELITE_SIZE
        and config.tournament_size == ga.GA_TOURNAMENT_SIZE
        and config.random_seed == ga.SEED
    )
    is_smoke = args.generations != ga.GA_GENERATIONS
    if not registered_ok:
        print("CẢNH BÁO: cấu hình khác tham số đã đăng ký – không dùng làm kết quả chính thức.")

    # -- 3.3 Baseline: CV trên Train bằng đúng các fold -------------------
    print("\n[1/4] Tái lập baseline (Logistic Regression mặc định, 7-fold CV trên Train)...")
    evaluator = ga.FitnessEvaluator(train_X, train_y, cv_folds=ga.CV_FOLDS, random_state=args.seed)
    baseline_record = evaluator.evaluate(ga.BASELINE_CHROMOSOME, generation=-1)
    if not baseline_record.ok:  # pragma: no cover - dữ liệu hiện tại luôn chạy được
        print("LỖI: baseline không đánh giá được:", baseline_record.error)
        return 1

    print(
        f"    Baseline CV Macro F1 = {baseline_record.mean_f1:.4f} "
        f"+/- {baseline_record.std_f1:.4f} | từng fold: "
        f"{[round(s, 4) for s in baseline_record.fold_scores]}"
    )

    # -- 3.4 Chạy GA ------------------------------------------------------
    print(
        f"\n[2/4] Chạy GA: {config.population_size} cá thể, {config.generations} thế hệ, "
        f"đột biến {config.mutation_rate}, lai ghép {config.crossover_rate}, "
        f"elite {config.elite_size}, tournament {config.tournament_size}"
    )
    algorithm = ga.GeneticAlgorithm(config, evaluator)
    result = algorithm.run()

    print(
        f"\n    Kết thúc: {result.generations_run} thế hệ, "
        f"{result.n_unique_evaluations} cấu hình duy nhất / "
        f"{result.n_total_evaluations} lượt đánh giá, "
        f"thời gian {result.total_time_s:.2f}s"
    )
    print(f"    Lý do dừng: {result.stopping_reason}")
    print(f"    Best chromosome: {ga.chromosome_key(result.best_chromosome)}")
    print(f"    Best params: {result.best_params}")
    print(
        f"    Best CV Macro F1 = {result.best_record.mean_f1:.4f} "
        f"+/- {result.best_record.std_f1:.4f}"
    )

    # -- 3.5 Huấn luyện lại và đánh giá trên Test -------------------------
    print("\n[3/4] Huấn luyện lại trên toàn bộ Train và đánh giá trên Test...")

    baseline_pipeline = ga.baseline_pipeline()
    baseline_pipeline.fit(train_X, train_y)
    baseline_metrics = evaluate_on_test(baseline_pipeline, test_X, test_y)

    ga_pipeline = ga.build_pipeline(result.best_params)
    ga_pipeline.fit(train_X, train_y)
    ga_metrics = evaluate_on_test(ga_pipeline, test_X, test_y)

    raw_comparison = [
        ("cv_macro_f1_mean", baseline_record.mean_f1, result.best_record.mean_f1),
        ("cv_macro_f1_std", baseline_record.std_f1, result.best_record.std_f1),
        ("test_accuracy", baseline_metrics["accuracy"], ga_metrics["accuracy"]),
        ("test_macro_f1", baseline_metrics["macro_f1"], ga_metrics["macro_f1"]),
        ("test_weighted_f1", baseline_metrics["weighted_f1"], ga_metrics["weighted_f1"]),
    ]
    comparison_rows = []
    for metric, baseline_value, ga_value in raw_comparison:
        delta = ga_value - baseline_value
        comparison_rows.append(
            {
                "metric": metric,
                "baseline": round(baseline_value, 4),
                "ga": round(ga_value, 4),
                # Chênh lệch tính từ giá trị gốc rồi mới làm tròn, tránh sai số
                # do trừ hai số đã làm tròn.
                "delta": round(delta, 4),
                "delta_percent": round(delta / baseline_value * 100, 2)
                if baseline_value
                else None,
            }
        )
    comparison = pd.DataFrame(comparison_rows)

    # -- 3.6 Lưu mô hình --------------------------------------------------
    model_dir = ga.resolve_model_dir()
    model_dir.mkdir(parents=True, exist_ok=True)
    # Smoke test không được ghi đè mô hình GA chính thức.
    ga_model_name = "ga_best_model_smoke.joblib" if is_smoke else ga.GA_MODEL_NAME
    ga_model_path = model_dir / ga_model_name
    joblib.dump(ga_pipeline, ga_model_path)

    # Kiểm chứng mô hình nạp lại được và dự đoán đúng 12 nhãn intent.
    reloaded = joblib.load(ga_model_path)
    reloaded_classes = [str(label) for label in reloaded.classes_]
    assert reloaded_classes == ga_metrics["labels"], "Nhãn intent không khớp sau khi lưu."
    assert list(reloaded.predict(test_X[:3])) == list(ga_metrics["predictions"][:3])
    baseline_model_path = model_dir / ga.BASELINE_MODEL_NAME
    print(f"    Đã lưu mô hình GA: {ga_model_path.relative_to(ga.BASE_DIR)}")
    print(
        "    Mô hình baseline vẫn giữ nguyên: "
        f"{baseline_model_path.relative_to(ga.BASE_DIR)} "
        f"({'tồn tại' if baseline_model_path.is_file() else 'KHÔNG tồn tại'})"
    )

    # -- 3.7 Ghi bằng chứng ------------------------------------------------
    print("\n[4/4] Ghi bằng chứng thí nghiệm...")
    finished_at = datetime.now().astimezone()
    total_script_time = time.perf_counter() - script_start
    fingerprint = ga.dataset_fingerprint(df)
    commit = ga.git_commit_id()

    run_config = {
        "experiment": "DT51 – GA tối ưu TF-IDF + Logistic Regression",
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "execution_timestamp": started_at.isoformat(),
        "is_smoke_run": is_smoke,
        "registered_parameters_preserved": registered_ok,
        "random_seed": args.seed,
        "registered_parameters": {
            "SEED": ga.SEED,
            "DATA_RATIO": ga.DATA_RATIO,
            "TEST_SIZE": ga.TEST_SIZE,
            "CV_FOLDS": ga.CV_FOLDS,
            "GA_POPULATION_SIZE": ga.GA_POPULATION_SIZE,
            "GA_MUTATION_RATE": ga.GA_MUTATION_RATE,
        },
        "ga_config": config.as_dict(),
        "gene_spaces": {
            name: [list(value) if isinstance(value, tuple) else value for value in space]
            for name, space in zip(ga.GENE_NAMES, ga.GENE_SPACES)
        },
        "search_space_size": ga.SEARCH_SPACE_SIZE,
        "chromosome_encoding": (
            "Tuple 5 chỉ số nguyên (G1..G5) trỏ vào GENE_SPACES; "
            "giải mã thành dict siêu tham số rồi dựng Pipeline."
        ),
        "baseline_chromosome": list(ga.BASELINE_CHROMOSOME),
        "dataset": {
            **fingerprint,
            "raw_rows": int(len(df)),
            "sampled_rows": int(len(df_sample)),
            "train_rows": int(len(train_df)),
            "test_rows": int(len(test_df)),
            "data_ratio": ga.DATA_RATIO,
            "test_size": ga.TEST_SIZE,
            "split_seed": args.seed,
            "source": "data/questions.csv",
        },
        "methodology": METHODOLOGY,
        "git_commit": commit,
        "git_commit_note": (
            "Không lấy được commit hash (không có Git hoặc lỗi gọi git)."
            if commit is None
            else "Commit của lần chạy thí nghiệm (trước khi commit kết quả)."
        ),
        "environment": ga.package_versions(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "results": {
            "generations_requested": config.generations,
            "generations_executed": result.generations_run,
            "stopping_reason": result.stopping_reason,
            "unique_fitness_evaluations": result.n_unique_evaluations,
            "total_fitness_evaluations": result.n_total_evaluations,
            "ga_runtime_seconds": round(result.total_time_s, 4),
            "script_runtime_seconds": round(total_script_time, 4),
            "failed_evaluations": int(evaluator.n_failures),
        },
        "artifacts": {
            "ga_model": str(ga_model_path.relative_to(ga.BASE_DIR)),
            "baseline_model": str(baseline_model_path.relative_to(ga.BASE_DIR)),
            "output_dir": _relative_or_name(output_dir),
        },
    }
    ga.write_json(output_dir / "ga_run_config.json", run_config)

    best_parameters = {
        "chromosome": list(result.best_chromosome),
        "chromosome_key": ga.chromosome_key(result.best_chromosome),
        "genes": {
            name: (list(value) if isinstance(value, tuple) else value)
            for name, value in result.best_params.items()
        },
        "decoded_pipeline": {
            "tfidf__ngram_range": list(result.best_params["ngram_range"]),
            "tfidf__min_df": result.best_params["min_df"],
            "tfidf__sublinear_tf": result.best_params["sublinear_tf"],
            "logreg__C": result.best_params["C"],
            "logreg__class_weight": result.best_params["class_weight"],
            "logreg__max_iter": 1000,
            "logreg__solver": "lbfgs",
        },
        "fitness": {
            "cv_macro_f1_mean": round(result.best_record.mean_f1, 6),
            "cv_macro_f1_std": round(result.best_record.std_f1, 6),
            "fold_scores": [round(score, 6) for score in result.best_record.fold_scores],
        },
        "selection_basis": "Macro F1 trung bình 7-fold CV trên tập Train (không dùng Test).",
        "discovered_at_generation": result.best_record.first_generation,
        "baseline_chromosome": list(ga.BASELINE_CHROMOSOME),
        "baseline_fitness": {
            "cv_macro_f1_mean": round(baseline_record.mean_f1, 6),
            "cv_macro_f1_std": round(baseline_record.std_f1, 6),
            "fold_scores": [round(score, 6) for score in baseline_record.fold_scores],
        },
    }
    ga.write_json(output_dir / "ga_best_parameters.json", best_parameters)

    history = result.history_frame()
    history.to_csv(output_dir / "ga_fitness_history.csv", index=False, encoding="utf-8-sig")
    result.trials.to_csv(output_dir / "ga_trials.csv", index=False, encoding="utf-8-sig")
    comparison.to_csv(output_dir / "ga_comparison.csv", index=False, encoding="utf-8-sig")

    intent_table = per_intent_frame(
        baseline_metrics["report"], ga_metrics["report"], baseline_metrics["labels"]
    )
    intent_table.to_csv(
        output_dir / "ga_classification_report.csv", index=False, encoding="utf-8-sig"
    )

    # So sánh cặp theo từng fold (cùng fold nên ghép cặp được) để tránh kết luận
    # "tốt hơn" chỉ dựa vào chênh lệch trung bình trên mẫu nhỏ.
    from scipy import stats

    baseline_folds = np.array(baseline_record.fold_scores)
    ga_folds = np.array(result.best_record.fold_scores)
    paired = stats.ttest_rel(ga_folds, baseline_folds)
    paired_test = {
        "test": "paired t-test trên 7 điểm Macro F1 theo fold (cùng cách chia fold)",
        "mean_difference": round(float(np.mean(ga_folds - baseline_folds)), 6),
        "t_statistic": round(float(paired.statistic), 6),
        "p_value": round(float(paired.pvalue), 6),
        "folds_ga_better": int(np.sum(ga_folds > baseline_folds)),
        "folds_equal": int(np.sum(np.isclose(ga_folds, baseline_folds))),
        "folds_baseline_better": int(np.sum(ga_folds < baseline_folds)),
        "alpha": 0.05,
        "significant_at_alpha": bool(paired.pvalue < 0.05),
    }

    metrics_payload = {
        "experiment": "GA optimization of TF-IDF + Logistic Regression (DT51)",        "timestamp": started_at.isoformat(),
        "is_smoke_run": is_smoke,
        "n_test_samples": int(len(test_y)),
        "baseline": {
            "hyperparameters": ga.decode_chromosome(ga.BASELINE_CHROMOSOME),
            "cv_macro_f1_mean": round(baseline_record.mean_f1, 6),
            "cv_macro_f1_std": round(baseline_record.std_f1, 6),
            "cv_fold_scores": [round(score, 6) for score in baseline_record.fold_scores],
            "test_accuracy": round(baseline_metrics["accuracy"], 6),
            "test_macro_f1": round(baseline_metrics["macro_f1"], 6),
            "test_weighted_f1": round(baseline_metrics["weighted_f1"], 6),
            "test_confusion_matrix": baseline_metrics["confusion_matrix"].tolist(),
            "labels": baseline_metrics["labels"],
            "n_test_errors": int(
                sum(
                    1
                    for true_label, predicted in zip(test_y, baseline_metrics["predictions"])
                    if true_label != predicted
                )
            ),
        },
        "ga_optimized": {
            "chromosome": list(result.best_chromosome),
            "hyperparameters": {
                name: (list(value) if isinstance(value, tuple) else value)
                for name, value in result.best_params.items()
            },
            "cv_macro_f1_mean": round(result.best_record.mean_f1, 6),
            "cv_macro_f1_std": round(result.best_record.std_f1, 6),
            "cv_fold_scores": [round(score, 6) for score in result.best_record.fold_scores],
            "test_accuracy": round(ga_metrics["accuracy"], 6),
            "test_macro_f1": round(ga_metrics["macro_f1"], 6),
            "test_weighted_f1": round(ga_metrics["weighted_f1"], 6),
            "test_confusion_matrix": ga_metrics["confusion_matrix"].tolist(),
            "labels": ga_metrics["labels"],
            "n_test_errors": int(
                sum(
                    1
                    for true_label, predicted in zip(test_y, ga_metrics["predictions"])
                    if true_label != predicted
                )
            ),
        },
        "difference": {
            row["metric"]: {"delta": row["delta"], "delta_percent": row["delta_percent"]}
            for row in comparison_rows
        },
        "paired_fold_test": paired_test,
        "ga_run": {
            "generations_executed": result.generations_run,
            "stopping_reason": result.stopping_reason,
            "unique_fitness_evaluations": result.n_unique_evaluations,
            "total_fitness_evaluations": result.n_total_evaluations,
            "ga_runtime_seconds": round(result.total_time_s, 4),
            "failed_evaluations": int(evaluator.n_failures),
        },
        "artifacts": [
            "ga_run_config.json",
            "ga_best_parameters.json",
            "ga_fitness_history.csv",
            "ga_trials.csv",
            "ga_comparison.csv",
            "ga_metrics.json",
            "ga_classification_report.csv",
            "ga_fitness_curve.png",
            "ga_confusion_matrix.png",
            "baseline_confusion_matrix.png",
            "environment.txt",
        ],
    }
    ga.write_json(output_dir / "ga_metrics.json", metrics_payload)

    ga.plot_fitness_curve(result, output_dir / "ga_fitness_curve.png")
    ga.plot_confusion_matrix(
        test_y,
        ga_metrics["predictions"],
        ga_metrics["labels"],
        output_dir / "ga_confusion_matrix.png",
        title="Confusion Matrix – Mô hình tối ưu bằng GA (tập Test)",
    )
    ga.plot_confusion_matrix(
        test_y,
        baseline_metrics["predictions"],
        baseline_metrics["labels"],
        output_dir / "baseline_confusion_matrix.png",
        title="Confusion Matrix – Baseline Logistic Regression (tập Test)",
    )
    ga.write_environment_file(output_dir / "environment.txt")

    # -- 3.8 Báo cáo trên terminal ----------------------------------------
    print("\n" + "=" * 78)
    print("BẢNG SO SÁNH BASELINE vs GA (cùng fold CV, cùng tập Test)")
    print("=" * 78)
    header = f"{'Chỉ số':<24}{'Baseline':>12}{'GA':>12}{'Δ':>12}"
    print(header)
    print("-" * len(header))
    labels_vi = {
        "cv_macro_f1_mean": "CV Macro F1 (mean)",
        "cv_macro_f1_std": "CV Macro F1 (std)",
        "test_accuracy": "Test Accuracy",
        "test_macro_f1": "Test Macro F1",
        "test_weighted_f1": "Test Weighted F1",
    }
    for row in comparison_rows:
        print(
            f"{labels_vi[row['metric']]:<24}{row['baseline']:>12.4f}"
            f"{row['ga']:>12.4f}{row['delta']:>+12.4f}"
        )

    print(
        f"\nSo sánh cặp theo fold: Δ TB = {paired_test['mean_difference']:+.4f}, "
        f"t = {paired_test['t_statistic']:.3f}, p = {paired_test['p_value']:.4f} -> "
        f"{'có ý nghĩa thống kê' if paired_test['significant_at_alpha'] else 'KHÔNG có ý nghĩa thống kê'}"
        f" ở alpha = 0.05 (GA tốt hơn {paired_test['folds_ga_better']}/7 fold, "
        f"kém hơn {paired_test['folds_baseline_better']}/7 fold)"
    )

    print("\nChỉ số theo từng intent (Test):")
    print(intent_table.to_string(index=False))

    print(f"\nSố câu Test dự đoán sai: baseline {metrics_payload['baseline']['n_test_errors']}"
          f" | GA {metrics_payload['ga_optimized']['n_test_errors']}")

    print("\nFile đã ghi:")
    for path in sorted(output_dir.iterdir()):
        if path.is_file():
            print(f"  - {_relative_or_name(path.parent)}/{path.name} ({path.stat().st_size} bytes)")

    print(f"\nTổng thời gian script: {total_script_time:.2f}s")
    if is_smoke:
        print(
            "\nLƯU Ý: ĐÂY LÀ SMOKE TEST (số thế hệ khác 12). "
            "Không dùng làm kết quả chính thức trong báo cáo."
        )
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
