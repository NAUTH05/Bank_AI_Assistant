"""Kiểm thử nhẹ cho giải thuật di truyền và tính tái lập của thí nghiệm DT51.

Chạy::

    python -m pytest tests -q

Bộ test cố tình giữ thời gian chạy ngắn (không huấn luyện GA đầy đủ) nhưng
vẫn kiểm tra các bất biến quan trọng: tính hợp lệ của nhiễm sắc thể, các toán
tử GA, kích thước quần thể, elitism, tính tất định, hàm thích nghi, việc không
truy cập Test trong quá trình tối ưu, lưu/nạp mô hình và hành vi chatbot gốc.
"""

from __future__ import annotations

import importlib
import inspect
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pytest
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

import genetic_algorithm as ga

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:  # pragma: no cover - conftest xử lý khi chạy pytest
    sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# Fixtures dùng chung
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def dataset():
    df = ga.load_dataset()
    return ga.split_dataset(df)


@pytest.fixture(scope="session")
def train_xy(dataset):
    _, train_df, _ = dataset
    return ga.to_xy(train_df)


@pytest.fixture(scope="session")
def test_xy(dataset):
    _, _, test_df = dataset
    return ga.to_xy(test_df)


@pytest.fixture(scope="session")
def evaluator(train_xy):
    train_X, train_y = train_xy
    return ga.FitnessEvaluator(train_X, train_y)


@pytest.fixture()
def tiny_config():
    """Cấu hình GA rất nhỏ, giữ nguyên population 40 và mutation 0.01."""
    return ga.GAConfig(
        population_size=ga.GA_POPULATION_SIZE,
        mutation_rate=ga.GA_MUTATION_RATE,
        generations=2,
        verbose=False,
    )


# ---------------------------------------------------------------------------
# 1. Biểu diễn nhiễm sắc thể
# ---------------------------------------------------------------------------


def test_registered_parameters_are_preserved():
    assert ga.SEED == 45116
    assert ga.DATA_RATIO == 0.95
    assert ga.TEST_SIZE == 0.30
    assert ga.CV_FOLDS == 7
    assert ga.GA_POPULATION_SIZE == 40
    assert ga.GA_MUTATION_RATE == 0.01
    assert ga.GA_CROSSOVER_RATE == 0.80
    assert ga.GA_ELITE_SIZE == 2
    assert ga.GA_TOURNAMENT_SIZE == 3


def test_search_space_matches_specification():
    assert ga.SEARCH_SPACE_SIZE == 9 * 3 * 2 * 2 * 2 == 216
    assert len(ga.GENE_SPACES) == 5
    assert len(ga.GENE_SPACES[0]) == 9
    assert ga.GENE_SPACES[1] == ((1, 1), (1, 2), (1, 3))
    assert ga.GENE_SPACES[2] == (1, 2)
    assert ga.GENE_SPACES[3] == (False, True)
    assert ga.GENE_SPACES[4] == (None, "balanced")


def test_random_chromosomes_are_valid():
    rng = np.random.default_rng(ga.SEED)
    for _ in range(500):
        chromosome = ga.random_chromosome(rng)
        assert ga.is_valid_chromosome(chromosome)
        assert len(chromosome) == 5
        assert all(isinstance(gene, int) for gene in chromosome)


@pytest.mark.parametrize(
    "bad_chromosome",
    [
        (),
        (0, 0, 0, 0),
        (0, 0, 0, 0, 0, 0),
        (9, 0, 0, 0, 0),  # C vượt miền
        (0, 3, 0, 0, 0),  # ngram_range vượt miền
        (0, 0, 2, 0, 0),  # min_df vượt miền
        (0, 0, 0, 2, 0),  # sublinear_tf vượt miền
        (0, 0, 0, 0, 2),  # class_weight vượt miền
        (0, 0, 0, 0, -1),  # chỉ số âm
        (0.0, 0, 0, 0, 0),  # không phải số nguyên
    ],
)
def test_invalid_chromosomes_are_rejected(bad_chromosome):
    assert not ga.is_valid_chromosome(bad_chromosome)
    with pytest.raises(ValueError):
        ga.decode_chromosome(bad_chromosome)


# ---------------------------------------------------------------------------
# 2. Giải mã nhiễm sắc thể
# ---------------------------------------------------------------------------


def test_baseline_chromosome_decodes_to_default_pipeline():
    params = ga.decode_chromosome(ga.BASELINE_CHROMOSOME)
    assert params == {
        "C": 1.0,
        "ngram_range": (1, 1),
        "min_df": 1,
        "sublinear_tf": False,
        "class_weight": None,
    }

    pipeline = ga.baseline_pipeline()
    tfidf = pipeline.named_steps["tfidf"]
    model = pipeline.named_steps["model"]
    assert tfidf.ngram_range == (1, 1)
    assert tfidf.min_df == 1
    assert tfidf.sublinear_tf is False
    assert model.C == 1.0
    assert model.class_weight is None
    assert model.max_iter == 1000
    assert model.solver == "lbfgs"


def test_every_chromosome_decodes_to_a_fittable_pipeline(train_xy):
    train_X, train_y = train_xy
    rng = np.random.default_rng(0)
    for _ in range(6):
        chromosome = ga.random_chromosome(rng)
        pipeline = ga.build_pipeline(ga.decode_chromosome(chromosome))
        pipeline.fit(train_X[:60], train_y[:60])
        assert len(pipeline.predict(train_X[:5])) == 5


# ---------------------------------------------------------------------------
# 3. Toán tử GA
# ---------------------------------------------------------------------------


def test_mutation_rate_is_configurable_and_defaults_to_registered_value():
    assert ga.GAConfig().mutation_rate == 0.01
    with pytest.raises(ValueError):
        ga.GAConfig(mutation_rate=1.5).validate()


def test_mutation_frequency_matches_configured_rate(evaluator):
    """Tần suất đột biến thực nghiệm phải xấp xỉ 0.01 trên mỗi gen."""
    algorithm = ga.GeneticAlgorithm(ga.GAConfig(generations=1, verbose=False), evaluator)
    base = ga.BASELINE_CHROMOSOME
    trials = 4000
    flips = 0
    for _ in range(trials):
        mutated = algorithm.mutate(base)
        assert ga.is_valid_chromosome(mutated)
        flips += sum(1 for a, b in zip(base, mutated) if a != b)

    expected = trials * len(ga.GENE_SPACES) * ga.GA_MUTATION_RATE
    # Khoảng tin cậy rộng: kiểm tra bậc độ lớn, không kiểm tra từng lần đột biến.
    assert expected * 0.5 <= flips <= expected * 1.6, (flips, expected)


def test_mutation_never_keeps_same_gene_value(evaluator):
    """Khi một gen đột biến, giá trị mới phải khác giá trị cũ."""
    algorithm = ga.GeneticAlgorithm(
        ga.GAConfig(mutation_rate=1.0, generations=1, verbose=False), evaluator
    )
    base = ga.BASELINE_CHROMOSOME
    for _ in range(200):
        mutated = algorithm.mutate(base)
        assert all(a != b for a, b in zip(base, mutated))
        assert ga.is_valid_chromosome(mutated)


def test_crossover_children_are_valid_and_inherit_parent_genes(evaluator):
    algorithm = ga.GeneticAlgorithm(ga.GAConfig(generations=1, verbose=False), evaluator)
    parent_a = (0, 0, 0, 0, 0)
    parent_b = (8, 2, 1, 1, 1)
    seen_from_a = False
    seen_from_b = False
    for _ in range(200):
        child = algorithm.crossover(parent_a, parent_b)
        assert ga.is_valid_chromosome(child)
        for gene, value_a, value_b in zip(child, parent_a, parent_b):
            assert gene in (value_a, value_b)
        seen_from_a = seen_from_a or any(g == a for g, a in zip(child, parent_a))
        seen_from_b = seen_from_b or any(g == b for g, b in zip(child, parent_b))
    assert seen_from_a and seen_from_b


def test_tournament_selection_prefers_better_individuals(evaluator):
    """Chọn lọc tournament phải ưu tiên cá thể có fitness cao hơn quần thể."""
    algorithm = ga.GeneticAlgorithm(ga.GAConfig(generations=1, verbose=False), evaluator)
    population = algorithm.initialize_population()
    for chromosome in population:
        evaluator.evaluate(chromosome)

    scores = np.array([algorithm._fitness_of(c) for c in population])
    population_mean = float(scores.mean())

    selected = np.array(
        [algorithm._fitness_of(algorithm.tournament_select(population)) for _ in range(400)]
    )
    random_draws = np.array(
        [
            algorithm._fitness_of(population[int(algorithm.rng.integers(0, len(population)))])
            for _ in range(400)
        ]
    )

    # Chọn lọc ngẫu nhiên thuần tuý chỉ tái tạo trung bình quần thể.
    assert selected.mean() > population_mean
    assert selected.mean() > random_draws.mean()
    # Tournament size 3 phải tạo áp lực chọn lọc rõ rệt (quan sát thực tế ~+0.05).
    assert selected.mean() > random_draws.mean() + 0.02


def test_elitism_keeps_best_and_does_not_inflate_population(evaluator):
    algorithm = ga.GeneticAlgorithm(ga.GAConfig(generations=1, verbose=False), evaluator)
    population = algorithm.initialize_population()
    for chromosome in population:
        evaluator.evaluate(chromosome)

    elites = algorithm.elites(population)
    assert len(elites) == ga.GA_ELITE_SIZE
    assert len(set(elites)) == len(elites)

    ordered_scores = sorted((algorithm._fitness_of(c) for c in population), reverse=True)
    assert [algorithm._fitness_of(c) for c in elites] == ordered_scores[: ga.GA_ELITE_SIZE]

    # Elitism + sinh sản không được làm quần thể vượt kích thước đăng ký.
    next_generation = list(elites)
    while len(next_generation) < ga.GA_POPULATION_SIZE:
        parent_a = algorithm.tournament_select(population)
        parent_b = algorithm.tournament_select(population)
        next_generation.append(algorithm.mutate(algorithm.crossover(parent_a, parent_b)))
    population_after = next_generation[: ga.GA_POPULATION_SIZE]
    assert len(population_after) == ga.GA_POPULATION_SIZE


def test_initial_population_contains_baseline_and_has_exact_size(evaluator):
    algorithm = ga.GeneticAlgorithm(ga.GAConfig(generations=1, verbose=False), evaluator)
    population = algorithm.initialize_population()
    assert len(population) == ga.GA_POPULATION_SIZE
    assert ga.BASELINE_CHROMOSOME in population
    assert all(ga.is_valid_chromosome(chromosome) for chromosome in population)


def test_deterministic_initialization_with_same_seed(evaluator):
    config = ga.GAConfig(generations=1, verbose=False)
    first = ga.GeneticAlgorithm(config, evaluator).initialize_population()
    second = ga.GeneticAlgorithm(config, evaluator).initialize_population()
    assert first == second


# ---------------------------------------------------------------------------
# 4. Hàm thích nghi
# ---------------------------------------------------------------------------


def test_fitness_equals_mean_macro_f1_over_identical_folds(train_xy, evaluator):
    """Fitness phải bằng Macro F1 trung bình của 7 fold, tính lại thủ công."""
    train_X, train_y = train_xy
    record = evaluator.evaluate(ga.BASELINE_CHROMOSOME)
    assert record.ok
    assert len(record.fold_scores) == ga.CV_FOLDS

    manual = []
    for train_index, valid_index in evaluator.splits:
        pipeline = ga.build_pipeline(record.params)
        pipeline.fit(train_X[train_index], train_y[train_index])
        manual.append(
            f1_score(
                train_y[valid_index],
                pipeline.predict(train_X[valid_index]),
                average="macro",
                zero_division=0,
            )
        )
    assert np.allclose(manual, record.fold_scores)
    assert record.mean_f1 == pytest.approx(float(np.mean(manual)), abs=1e-12)


def test_baseline_fitness_reproduces_recorded_reference(evaluator):
    """Giá trị tham chiếu 0.7782 trong notebook phải tái lập được."""
    record = evaluator.evaluate(ga.BASELINE_CHROMOSOME)
    assert record.mean_f1 == pytest.approx(0.7782, abs=1e-4)
    assert record.std_f1 == pytest.approx(0.0628, abs=1e-4)


def test_folds_are_identical_for_all_candidates(train_xy, evaluator):
    """Mọi ứng viên phải được đánh giá trên cùng một cách chia fold."""
    other = ga.FitnessEvaluator(*train_xy)
    assert len(other.splits) == len(evaluator.splits)
    for (train_a, valid_a), (train_b, valid_b) in zip(evaluator.splits, other.splits):
        assert np.array_equal(train_a, train_b)
        assert np.array_equal(valid_a, valid_b)

    different_seed = ga.FitnessEvaluator(*train_xy, random_state=1)
    assert any(
        not np.array_equal(valid_a, valid_b)
        for (_, valid_a), (_, valid_b) in zip(evaluator.splits, different_seed.splits)
    )


def test_fitness_is_cached_and_not_recomputed(evaluator):
    chromosome = ga.random_chromosome(np.random.default_rng(7))
    first = evaluator.evaluate(chromosome)
    cache_size = evaluator.n_unique_evaluations
    second = evaluator.evaluate(chromosome)
    assert first is second
    assert evaluator.n_unique_evaluations == cache_size


def test_invalid_candidate_is_recorded_as_failure(evaluator):
    before = evaluator.n_failures
    record = evaluator.evaluate((99, 0, 0, 0, 0))
    assert record.status == "failed"
    assert record.error == "invalid_chromosome"
    assert record.mean_f1 == float("-inf")
    assert evaluator.n_failures == before + 1
    row = record.as_row()
    assert row["mean_macro_f1"] == ""  # không bịa điểm


def test_split_reproduces_reference_sizes(dataset):
    df_sample, train_df, test_df = dataset
    assert len(df_sample) == 251
    assert len(train_df) == 175
    assert len(test_df) == 76
    assert train_df["intent"].nunique() == 12


def test_split_is_deterministic_and_stratified(dataset):
    df_sample, train_df, test_df = dataset
    df = ga.load_dataset()
    df_sample_2, train_df_2, test_df_2 = ga.split_dataset(df)
    assert list(train_df_2["question"]) == list(train_df["question"])
    assert list(test_df_2["question"]) == list(test_df["question"])
    assert test_df["intent"].value_counts().min() >= 6


# ---------------------------------------------------------------------------
# 5. Cô lập tập Test
# ---------------------------------------------------------------------------


def test_fitness_evaluator_has_no_test_input():
    """Về mặt cấu trúc, evaluator không thể nhận dữ liệu Test."""
    parameters = list(inspect.signature(ga.FitnessEvaluator.__init__).parameters)
    assert "test_questions" not in parameters
    assert "test_labels" not in parameters
    assert not any("test" in name for name in parameters)


def test_fitness_only_sees_train_partition(train_xy, evaluator, dataset):
    _, train_df, _ = dataset
    train_X, train_y = train_xy
    assert len(evaluator.X) == len(train_df)
    assert set(evaluator.X) == set(train_X)
    assert set(evaluator.y) == set(train_y)


def test_ga_run_does_not_require_test_data(evaluator):
    """GA chạy được khi hoàn toàn không có tập Test trong bộ nhớ."""
    result = ga.GeneticAlgorithm(
        ga.GAConfig(population_size=6, generations=1, elite_size=1, verbose=False),
        evaluator,
    ).run()
    assert result.best_record.ok
    assert math_is_finite(result.best_record.mean_f1)


def math_is_finite(value: float) -> bool:
    return value == value and value not in (float("inf"), float("-inf"))


# ---------------------------------------------------------------------------
# 6. Vòng lặp tiến hoá và tính tái lập
# ---------------------------------------------------------------------------


def test_generation_history_records_required_fields(train_xy):
    train_X, train_y = train_xy
    config = ga.GAConfig(population_size=8, generations=3, elite_size=1, verbose=False)
    result = ga.GeneticAlgorithm(config, ga.FitnessEvaluator(train_X, train_y)).run()

    required = {
        "generation",
        "best_fitness",
        "global_best_fitness",
        "mean_fitness",
        "worst_fitness",
        "std_fitness",
        "best_chromosome",
        "n_unique_configs",
        "elapsed_seconds",
    }
    assert required.issubset(set(result.history_frame().columns))
    assert result.generations_run == 3
    # Mỗi thế hệ đánh giá đúng population_size cá thể.
    assert result.n_total_evaluations == 3 * 8
    assert result.n_unique_evaluations <= 3 * 8
    assert result.best_record.ok
    assert ga.is_valid_chromosome(result.best_chromosome)


def test_full_run_is_reproducible_with_same_seed(train_xy):
    train_X, train_y = train_xy
    config = ga.GAConfig(population_size=8, generations=2, elite_size=1, verbose=False)

    first = ga.GeneticAlgorithm(config, ga.FitnessEvaluator(train_X, train_y)).run()
    second = ga.GeneticAlgorithm(config, ga.FitnessEvaluator(train_X, train_y)).run()

    assert first.best_chromosome == second.best_chromosome
    assert first.best_record.mean_f1 == pytest.approx(second.best_record.mean_f1, abs=1e-12)
    assert [row["best_fitness"] for row in first.history] == [
        row["best_fitness"] for row in second.history
    ]


def test_early_stopping_triggers_after_patience(train_xy):
    train_X, train_y = train_xy
    config = ga.GAConfig(
        population_size=40,
        generations=12,
        patience=1,
        verbose=False,
    )
    result = ga.GeneticAlgorithm(config, ga.FitnessEvaluator(train_X, train_y)).run()
    assert result.generations_run < 12
    assert "early_stopping" in result.stopping_reason


# ---------------------------------------------------------------------------
# 7. Lưu và nạp mô hình
# ---------------------------------------------------------------------------


def test_model_serialization_roundtrip(train_xy, test_xy, tmp_path):
    train_X, train_y = train_xy
    test_X, _ = test_xy

    pipeline = ga.build_pipeline(ga.decode_chromosome(ga.BASELINE_CHROMOSOME))
    pipeline.fit(train_X, train_y)

    path = tmp_path / "model.joblib"
    joblib.dump(pipeline, path)
    assert path.is_file()

    reloaded = joblib.load(path)
    assert list(reloaded.classes_) == list(pipeline.classes_)
    assert len(reloaded.classes_) == 12
    assert list(reloaded.predict(test_X)) == list(pipeline.predict(test_X))


def test_ga_model_does_not_overwrite_baseline_name():
    assert ga.GA_MODEL_NAME != ga.BASELINE_MODEL_NAME
    assert ga.GA_MODEL_NAME == "ga_best_model.joblib"


# ---------------------------------------------------------------------------
# 8. Bảo toàn hành vi chatbot gốc
# ---------------------------------------------------------------------------


@pytest.fixture()
def chatbot_module(monkeypatch):
    monkeypatch.delenv("BANKAI_MODEL_PATH", raising=False)
    import chatbot

    module = importlib.reload(chatbot)
    yield module
    monkeypatch.undo()
    importlib.reload(chatbot)


def test_chatbot_defaults_to_baseline_model(chatbot_module):
    assert chatbot_module.model_path.name == ga.BASELINE_MODEL_NAME
    assert "best_model.joblib" in str(chatbot_module.model_path)


def test_chatbot_rule_layer_and_fallback_unchanged(chatbot_module):
    answer, intent, method = chatbot_module.chatbot("Tôi bị mất thẻ ATM")
    assert method == "IF-THEN"
    assert intent == "lost_card"
    assert isinstance(answer, str) and answer

    answer, intent, method = chatbot_module.chatbot("Thời tiết hôm nay thế nào?")
    assert intent == "fallback"
    assert answer == chatbot_module.responses.get("fallback", chatbot_module.DEFAULT_FALLBACK)

    answer, intent, method = chatbot_module.chatbot("")
    assert intent == "fallback"
    assert method == "Không có"


def test_chatbot_model_can_be_switched_via_environment(monkeypatch):
    monkeypatch.setenv("BANKAI_MODEL_PATH", "Models/best_model.joblib")
    import chatbot

    module = importlib.reload(chatbot)
    try:
        assert module.model_path.is_file()
        answer, intent, method = module.chatbot("Tôi muốn mở tài khoản ngân hàng")
        assert isinstance(answer, str) and answer
        assert isinstance(intent, str)
    finally:
        monkeypatch.undo()
        importlib.reload(chatbot)


def test_chatbot_rejects_missing_override_path(monkeypatch):
    monkeypatch.setenv("BANKAI_MODEL_PATH", "Models/does_not_exist.joblib")
    import chatbot

    with pytest.raises(FileNotFoundError):
        importlib.reload(chatbot)

    monkeypatch.undo()
    importlib.reload(chatbot)


def test_intent_labels_and_rules_are_intact():
    import json

    responses = json.loads((ROOT / "data" / "responses.json").read_text(encoding="utf-8-sig"))
    rules = json.loads((ROOT / "data" / "rules.json").read_text(encoding="utf-8-sig"))

    df = ga.load_dataset()
    labels = set(df["intent"].unique())
    assert len(labels) == 12
    assert labels.issubset(set(responses))
    assert "fallback" in responses
    assert {rule["intent"] for rule in rules}.issubset(labels)
    assert os.path.exists(ROOT / "data" / "questions.csv")
