"""Giải thuật di truyền (GA) tối ưu pipeline TF-IDF + Logistic Regression.

Module này là nguồn duy nhất (single source of truth) cho:
    * Nạp dữ liệu và chia Train/Test giống hệt quy trình gốc trong ``train.py``.
    * Biểu diễn nhiễm sắc thể, giải mã thành Pipeline scikit-learn hợp lệ.
    * Hàm thích nghi (fitness) = Macro F1 trung bình qua 7-fold Stratified CV
      trên **phân vùng Train**, fit lại toàn bộ Pipeline trong từng fold.
    * Các toán tử GA: khởi tạo quần thể, chọn lọc tournament, lai ghép đều,
      đột biến theo từng gen, ưu tú (elitism) và vòng lặp tiến hoá.
    * Trực quan hoá lịch sử hội tụ.

Notebook ``DT51.ipynb`` và script ``train_ga.py`` đều import từ module này để
tránh tồn tại hai cài đặt GA song song.

Tham số đăng ký (bắt buộc, không được thay đổi khi chạy thí nghiệm chính thức):
    SEED = 45116, DATA_RATIO = 0.95, TEST_SIZE = 0.30, CV_FOLDS = 7,
    GA_POPULATION_SIZE = 40, GA_MUTATION_RATE = 0.01.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

# ---------------------------------------------------------------------------
# 1. Tham số thí nghiệm đã đăng ký
# ---------------------------------------------------------------------------

SEED = 45116
DATA_RATIO = 0.95
TEST_SIZE = 0.30
CV_FOLDS = 7

GA_POPULATION_SIZE = 40
GA_MUTATION_RATE = 0.01

# Giá trị mặc định của các siêu tham số GA còn lại
GA_CROSSOVER_RATE = 0.80
GA_GENERATIONS = 12
GA_ELITE_SIZE = 2
GA_TOURNAMENT_SIZE = 3
# Dừng sớm là tuỳ chọn: khi bật, dừng sau GA_PATIENCE thế hệ liên tiếp không
# cải thiện quá GA_IMPROVEMENT_EPS. Mặc định tắt để thí nghiệm chính thức chạy
# đúng số thế hệ đã đăng ký (12); xem train_ga.py --patience.
GA_PATIENCE = 5
GA_IMPROVEMENT_EPS = 1e-6

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "data" / "questions.csv"
RESULTS_DIR = BASE_DIR / "ga_results"

# Tên file mô hình: giữ nguyên quy ước thư mục hiện có của dự án.
MODEL_DIR_CANDIDATES = ("Models", "models")
BASELINE_MODEL_NAME = "best_model.joblib"
GA_MODEL_NAME = "ga_best_model.joblib"


# ---------------------------------------------------------------------------
# 2. Không gian gen và biểu diễn nhiễm sắc thể
# ---------------------------------------------------------------------------
# Mỗi gen là một *chỉ số* trỏ vào không gian giá trị tương ứng. Nhờ vậy nhiễm
# sắc thể là một tuple số nguyên: băm được (dùng làm khoá cache), so sánh được
# và luôn hợp lệ về mặt cấu trúc.
#
#   G1  C                  : 0.01, 0.1, 0.5, 1, 2, 5, 10, 20, 50
#   G2  ngram_range        : (1,1), (1,2), (1,3)
#   G3  min_df             : 1, 2
#   G4  sublinear_tf       : False, True
#   G5  class_weight       : None, "balanced"

GENE_SPACES: Tuple[Tuple[Any, ...], ...] = (
    (0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0),
    ((1, 1), (1, 2), (1, 3)),
    (1, 2),
    (False, True),
    (None, "balanced"),
)

GENE_NAMES: Tuple[str, ...] = (
    "C",
    "ngram_range",
    "min_df",
    "sublinear_tf",
    "class_weight",
)

# Nhiễm sắc thể tương ứng cấu hình gốc: TfidfVectorizer() + LogisticRegression(max_iter=1000)
BASELINE_CHROMOSOME: Tuple[int, ...] = (3, 0, 0, 0, 0)

SEARCH_SPACE_SIZE = int(np.prod([len(space) for space in GENE_SPACES]))  # 216


def random_chromosome(rng: np.random.Generator) -> Tuple[int, ...]:
    """Sinh ngẫu nhiên một nhiễm sắc thể hợp lệ."""
    return tuple(int(rng.integers(0, len(space))) for space in GENE_SPACES)


def is_valid_chromosome(chromosome: Sequence[int]) -> bool:
    """Kiểm tra nhiễm sắc thể có đúng số gen và chỉ số nằm trong miền giá trị."""
    if len(chromosome) != len(GENE_SPACES):
        return False
    for index, space in zip(chromosome, GENE_SPACES):
        if not isinstance(index, (int, np.integer)):
            return False
        if not 0 <= int(index) < len(space):
            return False
    return True


def decode_chromosome(chromosome: Sequence[int]) -> Dict[str, Any]:
    """Giải mã nhiễm sắc thể thành dict siêu tham số có ý nghĩa."""
    if not is_valid_chromosome(chromosome):
        raise ValueError(f"Nhiễm sắc thể không hợp lệ: {chromosome!r}")
    return {
        name: space[int(index)]
        for name, space, index in zip(GENE_NAMES, GENE_SPACES, chromosome)
    }


def chromosome_key(chromosome: Sequence[int]) -> str:
    """Khoá văn bản ổn định cho nhiễm sắc thể (dùng trong log/CSV)."""
    return "-".join(str(int(gene)) for gene in chromosome)


def chromosome_id(chromosome: Sequence[int]) -> str:
    """Mã định danh ngắn, ổn định cho nhiễm sắc thể."""
    return "chr_" + chromosome_key(chromosome).replace("-", "")


def params_to_json(params: Dict[str, Any]) -> str:
    """Chuyển dict siêu tham số sang JSON gọn (giữ nguyên kiểu dữ liệu)."""

    def _encode(value: Any) -> Any:
        if isinstance(value, tuple):
            return list(value)
        return value

    return json.dumps({key: _encode(value) for key, value in params.items()}, ensure_ascii=False)


def build_pipeline(params: Dict[str, Any]) -> Pipeline:
    """Dựng Pipeline scikit-learn hợp lệ từ dict siêu tham số.

    Lưu ý: solver ``lbfgs`` là tất định nên không cần ``random_state`` cho
    Logistic Regression; ``random_state`` của CV và của GA mới là yếu tố
    quyết định tính tái lập.
    """
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=tuple(params["ngram_range"]),
                    min_df=params["min_df"],
                    sublinear_tf=params["sublinear_tf"],
                ),
            ),
            (
                "model",
                LogisticRegression(
                    C=float(params["C"]),
                    class_weight=params["class_weight"],
                    max_iter=1000,
                    solver="lbfgs",
                ),
            ),
        ]
    )


def baseline_pipeline() -> Pipeline:
    """Pipeline gốc dùng cho đối chứng (đúng như ``train.py``)."""
    return build_pipeline(decode_chromosome(BASELINE_CHROMOSOME))


# ---------------------------------------------------------------------------
# 3. Dữ liệu: nạp, làm sạch, chia Train/Test (giống hệt train.py)
# ---------------------------------------------------------------------------


def load_dataset(csv_path: Optional[Path] = None) -> pd.DataFrame:
    """Đọc ``data/questions.csv`` và làm sạch theo đúng quy trình gốc."""
    path = Path(csv_path) if csv_path is not None else DATA_FILE
    df = pd.read_csv(path, encoding="utf-8-sig")

    df = df.dropna(subset=["question", "intent"]).copy()
    df["question"] = df["question"].str.lower().str.strip()
    df["question"] = df["question"].str.replace(r"\s+", " ", regex=True)
    df["intent"] = df["intent"].str.strip()
    df = df[(df["question"] != "") & (df["intent"] != "")].copy()

    conflict = df.groupby("question")["intent"].nunique()
    if (conflict > 1).any():
        raise ValueError("Có câu hỏi trùng nhưng khác intent. Hãy kiểm tra questions.csv.")

    df = df.drop_duplicates(subset=["question"]).reset_index(drop=True)
    return df


def split_dataset(
    df: pd.DataFrame,
    seed: int = SEED,
    data_ratio: float = DATA_RATIO,
    test_size: float = TEST_SIZE,
    cv_folds: int = CV_FOLDS,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Lấy mẫu ``data_ratio`` rồi chia Train/Test phân tầng.

    Trả về ``(df_sample, train_df, test_df)``. Không thay đổi thứ tự hay nội
    dung câu hỏi so với quy trình gốc.
    """
    df_sample = df.sample(frac=data_ratio, random_state=seed)
    train_df, test_df = train_test_split(
        df_sample,
        test_size=test_size,
        random_state=seed,
        stratify=df_sample["intent"],
    )
    if train_df["intent"].value_counts().min() < cv_folds:
        raise ValueError(
            f"Có nhóm intent có ít hơn {cv_folds} câu trong Train. Hãy bổ sung dữ liệu."
        )
    return df_sample, train_df, test_df


def to_xy(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Chuyển DataFrame sang hai mảng numpy (văn bản, nhãn)."""
    return (
        df["question"].to_numpy(dtype=object),
        df["intent"].to_numpy(dtype=object),
    )


def dataset_fingerprint(df: pd.DataFrame) -> Dict[str, Any]:
    """Thông tin nhận dạng dữ liệu dùng cho metadata của thí nghiệm."""
    import hashlib

    payload = "\n".join(f"{q}\t{i}" for q, i in zip(df["question"], df["intent"]))
    return {
        "n_rows": int(len(df)),
        "n_intents": int(df["intent"].nunique()),
        "sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
    }


# ---------------------------------------------------------------------------
# 4. Hàm thích nghi (fitness)
# ---------------------------------------------------------------------------


@dataclass
class FitnessRecord:
    """Kết quả đánh giá một nhiễm sắc thể."""

    chromosome: Tuple[int, ...]
    params: Dict[str, Any]
    mean_f1: float
    std_f1: float
    fold_scores: Tuple[float, ...]
    status: str = "ok"
    error: str = ""
    duration_s: float = 0.0
    first_generation: int = -1

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    def as_row(self) -> Dict[str, Any]:
        """Dòng phẳng để ghi CSV."""
        row: Dict[str, Any] = {
            "chromosome_id": chromosome_id(self.chromosome),
            "chromosome": chromosome_key(self.chromosome),
        }
        for name, gene_index in zip(GENE_NAMES, self.chromosome):
            row[f"gene_{name}"] = int(gene_index)
        for name in GENE_NAMES:
            value = self.params.get(name)
            row[name] = json.dumps(list(value)) if isinstance(value, tuple) else value
        row.update(
            {
                "mean_macro_f1": round(self.mean_f1, 6) if self.ok else "",
                "std_macro_f1": round(self.std_f1, 6) if self.ok else "",
                "status": self.status,
                "error": self.error,
                "first_seen_generation": self.first_generation,
                "duration_s": round(self.duration_s, 6),
            }
        )
        for fold_index, score in enumerate(self.fold_scores, start=1):
            row[f"fold_{fold_index}_f1"] = round(score, 6) if self.ok else ""
        return row


class FitnessEvaluator:
    """Đánh giá nhiễm sắc thể bằng 7-fold Stratified CV trên phân vùng Train.

    Nguyên tắc bắt buộc:
        * Chỉ nhận dữ liệu Train; **không** có tham số nào cho tập Test, nên
          về mặt cấu trúc không thể rò rỉ Test vào quá trình tối ưu.
        * Các fold được tính **một lần** và dùng lại cho mọi ứng viên, nên mọi
          cá thể được so sánh trên đúng cùng một cách chia fold.
        * TF-IDF được fit lại bên trong Pipeline ở từng fold (không fit toàn cục).
        * Kết quả được cache theo nhiễm sắc thể; cá thể lỗi được ghi nhận
          minh bạch thay vì gán điểm giả.
    """

    def __init__(
        self,
        train_questions: Sequence[str],
        train_labels: Sequence[str],
        cv_folds: int = CV_FOLDS,
        random_state: int = SEED,
    ) -> None:
        self.X = np.asarray(train_questions, dtype=object)
        self.y = np.asarray(train_labels, dtype=object)
        if len(self.X) != len(self.y):
            raise ValueError("Số câu hỏi và số nhãn không khớp.")

        self.cv_folds = int(cv_folds)
        self.random_state = int(random_state)
        splitter = StratifiedKFold(
            n_splits=self.cv_folds,
            shuffle=True,
            random_state=self.random_state,
        )
        # Tính fold một lần duy nhất -> mọi ứng viên dùng chung fold.
        self.splits: List[Tuple[np.ndarray, np.ndarray]] = list(
            splitter.split(self.X, self.y)
        )
        self.cache: Dict[Tuple[int, ...], FitnessRecord] = {}
        self.n_failures = 0

    # -- API ---------------------------------------------------------------

    def evaluate(self, chromosome: Sequence[int], generation: int = -1) -> FitnessRecord:
        key = tuple(int(gene) for gene in chromosome)
        if key in self.cache:
            return self.cache[key]

        if not is_valid_chromosome(key):
            record = FitnessRecord(
                chromosome=key,
                params={},
                mean_f1=float("-inf"),
                std_f1=float("nan"),
                fold_scores=(),
                status="failed",
                error="invalid_chromosome",
                first_generation=generation,
            )
            self.cache[key] = record
            self.n_failures += 1
            return record

        params = decode_chromosome(key)
        start = time.perf_counter()
        fold_scores: List[float] = []
        try:
            for train_index, valid_index in self.splits:
                pipeline = build_pipeline(params)
                pipeline.fit(self.X[train_index], self.y[train_index])
                predictions = pipeline.predict(self.X[valid_index])
                fold_scores.append(
                    float(
                        f1_score(
                            self.y[valid_index],
                            predictions,
                            average="macro",
                            zero_division=0,
                        )
                    )
                )
        except Exception as exc:  # pragma: no cover - phụ thuộc dữ liệu
            self.n_failures += 1
            record = FitnessRecord(
                chromosome=key,
                params=params,
                mean_f1=float("-inf"),
                std_f1=float("nan"),
                fold_scores=tuple(fold_scores),
                status="failed",
                error=f"{type(exc).__name__}: {exc}",
                duration_s=time.perf_counter() - start,
                first_generation=generation,
            )
            self.cache[key] = record
            return record

        record = FitnessRecord(
            chromosome=key,
            params=params,
            mean_f1=float(np.mean(fold_scores)),
            std_f1=float(np.std(fold_scores)),
            fold_scores=tuple(fold_scores),
            status="ok",
            error="",
            duration_s=time.perf_counter() - start,
            first_generation=generation,
        )
        self.cache[key] = record
        return record

    # -- tiện ích ----------------------------------------------------------

    @property
    def n_unique_evaluations(self) -> int:
        return len(self.cache)

    def trials_frame(self) -> pd.DataFrame:
        """Bảng mọi cấu hình đã đánh giá, sắp theo lần xuất hiện đầu tiên."""
        records = sorted(
            self.cache.values(),
            key=lambda rec: (rec.first_generation, rec.chromosome),
        )
        if not records:
            return pd.DataFrame()
        return pd.DataFrame([record.as_row() for record in records])


# ---------------------------------------------------------------------------
# 5. Cấu hình GA và kết quả
# ---------------------------------------------------------------------------


@dataclass
class GAConfig:
    """Cấu hình một lần chạy GA."""

    population_size: int = GA_POPULATION_SIZE
    mutation_rate: float = GA_MUTATION_RATE
    crossover_rate: float = GA_CROSSOVER_RATE
    generations: int = GA_GENERATIONS
    elite_size: int = GA_ELITE_SIZE
    tournament_size: int = GA_TOURNAMENT_SIZE
    random_seed: int = SEED
    patience: Optional[int] = None
    improvement_eps: float = GA_IMPROVEMENT_EPS
    verbose: bool = True

    def validate(self) -> None:
        if self.population_size < 2:
            raise ValueError("population_size phải >= 2.")
        if not 0.0 <= self.mutation_rate <= 1.0:
            raise ValueError("mutation_rate phải nằm trong [0, 1].")
        if not 0.0 <= self.crossover_rate <= 1.0:
            raise ValueError("crossover_rate phải nằm trong [0, 1].")
        if self.generations < 1:
            raise ValueError("generations phải >= 1.")
        if not 0 <= self.elite_size < self.population_size:
            raise ValueError("elite_size phải nằm trong [0, population_size).")
        if self.tournament_size < 2:
            raise ValueError("tournament_size phải >= 2.")

    def as_dict(self) -> Dict[str, Any]:
        return {
            "population_size": self.population_size,
            "mutation_rate": self.mutation_rate,
            "crossover_rate": self.crossover_rate,
            "generations": self.generations,
            "elite_size": self.elite_size,
            "tournament_size": self.tournament_size,
            "random_seed": self.random_seed,
            "patience": self.patience,
            "improvement_eps": self.improvement_eps,
        }


@dataclass
class GAResult:
    """Toàn bộ kết quả một lần chạy GA."""

    config: GAConfig
    history: List[Dict[str, Any]]
    trials: pd.DataFrame
    best_chromosome: Tuple[int, ...]
    best_params: Dict[str, Any]
    best_record: Optional[FitnessRecord]
    total_time_s: float
    n_unique_evaluations: int
    n_total_evaluations: int
    generations_run: int
    stopping_reason: str
    baseline_record: Optional[FitnessRecord] = None

    def history_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.history)


# ---------------------------------------------------------------------------
# 6. Giải thuật di truyền
# ---------------------------------------------------------------------------


class GeneticAlgorithm:
    """GA tối đa hoá Macro F1 trung bình qua 7-fold CV trên tập Train."""

    def __init__(self, config: GAConfig, evaluator: FitnessEvaluator) -> None:
        config.validate()
        self.config = config
        self.evaluator = evaluator
        self.rng = np.random.default_rng(config.random_seed)
        self.history: List[Dict[str, Any]] = []
        self._n_evaluations = 0
        self._generation_index = 0

    # -- 6.1 Khởi tạo quần thể ---------------------------------------------

    def initialize_population(self) -> List[Tuple[int, ...]]:
        """Tạo quần thể ban đầu đa dạng, có nhiễm sắc thể gốc nếu có thể."""
        population: List[Tuple[int, ...]] = []
        seen = set()

        if is_valid_chromosome(BASELINE_CHROMOSOME):
            population.append(BASELINE_CHROMOSOME)
            seen.add(BASELINE_CHROMOSOME)

        max_attempts = 200 * self.config.population_size
        attempts = 0
        while len(population) < self.config.population_size and attempts < max_attempts:
            attempts += 1
            candidate = random_chromosome(self.rng)
            if candidate in seen:
                continue
            seen.add(candidate)
            population.append(candidate)

        # Không gian tìm kiếm nhỏ hơn quần thể: chấp nhận lặp để giữ đúng kích thước.
        while len(population) < self.config.population_size:
            population.append(random_chromosome(self.rng))

        return population[: self.config.population_size]

    # -- 6.2 Chọn lọc tournament -------------------------------------------

    def tournament_select(self, population: Sequence[Tuple[int, ...]]) -> Tuple[int, ...]:
        """Chọn cá thể tốt nhất trong ``tournament_size`` cá thể ngẫu nhiên."""
        indices = self.rng.integers(0, len(population), size=self.config.tournament_size)
        contestants = [population[int(i)] for i in indices]
        return max(contestants, key=self._fitness_of)

    # -- 6.3 Lai ghép đều ---------------------------------------------------

    def crossover(
        self, parent_a: Tuple[int, ...], parent_b: Tuple[int, ...]
    ) -> Tuple[int, ...]:
        """Lai ghép đều: mỗi gen lấy ngẫu nhiên từ bố hoặc mẹ."""
        child = tuple(
            int(parent_a[i]) if self.rng.random() < 0.5 else int(parent_b[i])
            for i in range(len(GENE_SPACES))
        )
        if not is_valid_chromosome(child):  # pragma: no cover - phòng vệ
            raise ValueError(f"Lai ghép tạo nhiễm sắc thể không hợp lệ: {child!r}")
        return child

    # -- 6.4 Đột biến -------------------------------------------------------

    def mutate(self, chromosome: Tuple[int, ...]) -> Tuple[int, ...]:
        """Đột biến độc lập theo từng gen với xác suất ``mutation_rate``.

        Khi một gen đột biến, giá trị mới được chọn trong không gian giá trị
        của gen đó và **khác** giá trị cũ.
        """
        genes = list(chromosome)
        for position, space in enumerate(GENE_SPACES):
            if self.rng.random() >= self.config.mutation_rate:
                continue
            if len(space) < 2:  # pragma: no cover - không xảy ra với cấu hình này
                continue
            alternatives = [i for i in range(len(space)) if i != genes[position]]
            genes[position] = int(self.rng.choice(alternatives))
        mutated = tuple(genes)
        if not is_valid_chromosome(mutated):  # pragma: no cover - phòng vệ
            raise ValueError(f"Đột biến tạo nhiễm sắc thể không hợp lệ: {mutated!r}")
        return mutated

    # -- 6.5 Ưu tú ----------------------------------------------------------

    def elites(self, population: Sequence[Tuple[int, ...]]) -> List[Tuple[int, ...]]:
        """Lấy ``elite_size`` cá thể tốt nhất, không tạo bản sao thừa."""
        ordered = sorted(population, key=self._fitness_of, reverse=True)
        chosen: List[Tuple[int, ...]] = []
        for chromosome in ordered:
            if chromosome in chosen:
                continue
            chosen.append(chromosome)
            if len(chosen) == self.config.elite_size:
                break
        return chosen

    # -- 6.6 Vòng lặp tiến hoá ---------------------------------------------

    def run(self) -> GAResult:
        start_time = time.perf_counter()
        self._log(
            f"Khởi tạo quần thể: {self.config.population_size} cá thể "
            f"(đột biến {self.config.mutation_rate}, lai ghép {self.config.crossover_rate}, "
            f"elite {self.config.elite_size}, seed {self.config.random_seed})"
        )

        population = self.initialize_population()
        self._evaluate_population(population, generation=0)
        self.history.append(self._snapshot(0, population, start_time))

        best_chromosome, best_fitness = self._best_of(population)
        stall = 0
        stopping_reason = "max_generations"
        generations_run = 1

        for generation in range(1, self.config.generations):
            self._generation_index = generation
            next_population: List[Tuple[int, ...]] = list(self.elites(population))

            while len(next_population) < self.config.population_size:
                parent_a = self.tournament_select(population)
                parent_b = self.tournament_select(population)
                if self.rng.random() < self.config.crossover_rate:
                    child = self.crossover(parent_a, parent_b)
                else:
                    child = parent_a
                child = self.mutate(child)
                next_population.append(child)

            # Cắt đúng kích thước: elitism không được làm quần thể phình ra.
            population = next_population[: self.config.population_size]
            self._evaluate_population(population, generation)
            self.history.append(self._snapshot(generation, population, start_time))
            generations_run = generation + 1

            current_best_chromosome, current_best_fitness = self._best_of(population)
            if current_best_fitness > best_fitness + self.config.improvement_eps:
                best_chromosome, best_fitness = current_best_chromosome, current_best_fitness
                stall = 0
            else:
                stall += 1
                if current_best_fitness > best_fitness:
                    best_chromosome, best_fitness = (
                        current_best_chromosome,
                        current_best_fitness,
                    )

            if self.config.patience is not None and stall >= self.config.patience:
                stopping_reason = (
                    f"early_stopping: {self.config.patience} thế hệ liên tiếp "
                    "không cải thiện đáng kể"
                )
                self._log(f"Dừng sớm tại thế hệ {generation}: {stopping_reason}")
                break

        total_time = time.perf_counter() - start_time
        best_record = self.evaluator.cache.get(best_chromosome)

        self._log(
            f"Hoàn tất sau {generations_run} thế hệ, "
            f"{self.evaluator.n_unique_evaluations} cấu hình duy nhất, "
            f"{total_time:.2f} giây. Best CV Macro F1 = {best_fitness:.4f}"
        )

        return GAResult(
            config=self.config,
            history=list(self.history),
            trials=self.evaluator.trials_frame(),
            best_chromosome=best_chromosome,
            best_params=decode_chromosome(best_chromosome),
            best_record=best_record,
            total_time_s=total_time,
            n_unique_evaluations=self.evaluator.n_unique_evaluations,
            n_total_evaluations=self._n_evaluations,
            generations_run=generations_run,
            stopping_reason=stopping_reason,
            baseline_record=self.evaluator.cache.get(BASELINE_CHROMOSOME),
        )

    # -- nội bộ -------------------------------------------------------------

    def _fitness_of(self, chromosome: Tuple[int, ...]) -> float:
        record = self.evaluator.cache.get(chromosome)
        return record.mean_f1 if record is not None else float("-inf")

    def _evaluate_population(
        self, population: Sequence[Tuple[int, ...]], generation: int
    ) -> None:
        for chromosome in population:
            self.evaluator.evaluate(chromosome, generation=generation)
            self._n_evaluations += 1

    def _best_of(
        self, population: Sequence[Tuple[int, ...]]
    ) -> Tuple[Tuple[int, ...], float]:
        finite = [
            (chromosome, self._fitness_of(chromosome))
            for chromosome in population
            if math.isfinite(self._fitness_of(chromosome))
        ]
        if not finite:  # pragma: no cover - mọi ứng viên đều lỗi
            raise RuntimeError("Không có ứng viên nào đánh giá thành công.")
        return max(finite, key=lambda item: item[1])

    def _snapshot(
        self, generation: int, population: Sequence[Tuple[int, ...]], start_time: float
    ) -> Dict[str, Any]:
        scores = [
            self._fitness_of(chromosome)
            for chromosome in population
            if math.isfinite(self._fitness_of(chromosome))
        ]
        best_chromosome, best_fitness = self._best_of(population)
        global_best = max(row["global_best_fitness"] for row in self.history) if self.history else best_fitness
        global_best = max(global_best, best_fitness)

        row = {
            "generation": generation,
            "best_fitness": best_fitness,
            "global_best_fitness": global_best,
            "mean_fitness": float(np.mean(scores)),
            "worst_fitness": float(np.min(scores)),
            "std_fitness": float(np.std(scores)),
            "n_unique_configs": self.evaluator.n_unique_evaluations,
            "n_failed": int(self.evaluator.n_failures),
            "elapsed_seconds": time.perf_counter() - start_time,
            "best_chromosome": chromosome_key(best_chromosome),
            "best_params": params_to_json(decode_chromosome(best_chromosome)),
        }

        self._log(
            f"[Gen {generation:02d}] best={row['best_fitness']:.4f} "
            f"global={row['global_best_fitness']:.4f} "
            f"mean={row['mean_fitness']:.4f} std={row['std_fitness']:.4f} "
            f"unique={row['n_unique_configs']} "
            f"t={row['elapsed_seconds']:.2f}s "
            f"| {row['best_chromosome']}"
        )
        return row

    def _log(self, message: str) -> None:
        if self.config.verbose:
            print(message, flush=True)


# ---------------------------------------------------------------------------
# 7. Trực quan hoá
# ---------------------------------------------------------------------------


def plot_fitness_curve(result: GAResult, output_path: Path) -> Path:
    """Vẽ đường hội tụ: best / global best / mean ± std theo thế hệ."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    history = result.history_frame()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 5.2))
    generations = history["generation"].to_numpy()

    ax.plot(
        generations,
        history["global_best_fitness"],
        linestyle="--",
        linewidth=2.2,
        color="#1f77b4",
        label="Best toàn cục",
        zorder=2,
    )
    ax.plot(
        generations,
        history["best_fitness"],
        linestyle="none",
        marker="o",
        markersize=6.5,
        markerfacecolor="white",
        markeredgewidth=1.6,
        color="#d62728",
        label="Best trong từng thế hệ",
        zorder=3,
    )
    ax.plot(
        generations,
        history["mean_fitness"],
        marker="^",
        linewidth=1.6,
        color="#2ca02c",
        label="Trung bình quần thể",
    )
    ax.fill_between(
        generations,
        history["mean_fitness"] - history["std_fitness"],
        history["mean_fitness"] + history["std_fitness"],
        alpha=0.18,
        color="#2ca02c",
        label="± 1 độ lệch chuẩn",
    )

    baseline = result.baseline_record.mean_f1 if result.baseline_record else None
    if baseline is not None and math.isfinite(baseline):
        ax.axhline(
            baseline,
            color="gray",
            linestyle=":",
            linewidth=1.4,
            label=f"Baseline CV Macro F1 = {baseline:.4f}",
        )

    ax.set_xlabel("Thế hệ")
    ax.set_ylabel("Macro F1 (7-fold CV trên tập Train)")
    ax.set_title("Đường hội tụ của giải thuật di truyền")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def plot_confusion_matrix(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    labels: Sequence[str],
    output_path: Path,
    title: str = "Confusion Matrix - Mô hình tối ưu bằng GA",
) -> Path:
    """Vẽ và lưu ma trận nhầm lẫn trên tập Test."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    matrix = confusion_matrix(y_true, y_pred, labels=list(labels))
    fig, ax = plt.subplots(figsize=(11, 9))
    display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=list(labels))
    display.plot(ax=ax, xticks_rotation=90, values_format="d", colorbar=False, cmap="Blues")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


# ---------------------------------------------------------------------------
# 8. Tiện ích môi trường / metadata
# ---------------------------------------------------------------------------


def git_commit_id(repo_dir: Optional[Path] = None) -> Optional[str]:
    """Lấy commit hash hiện tại; trả ``None`` nếu không có Git."""
    import subprocess

    directory = str(repo_dir or BASE_DIR)
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=directory,
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    commit = completed.stdout.strip()
    return commit or None


def package_versions() -> Dict[str, str]:
    """Phiên bản các gói chính dùng trong thí nghiệm."""
    import platform
    import sklearn

    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit-learn": sklearn.__version__,
        "joblib": _safe_version("joblib"),
        "matplotlib": _safe_version("matplotlib"),
    }


def _safe_version(module_name: str) -> str:
    try:
        module = __import__(module_name)
    except ImportError:  # pragma: no cover
        return "not-installed"
    return getattr(module, "__version__", "unknown")


def write_environment_file(output_path: Path) -> Path:
    """Ghi ``environment.txt``: hệ điều hành, phiên bản Python, gói chính."""
    import platform
    import sys

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Môi trường thực thi thí nghiệm GA (DT51)",
        f"platform: {platform.platform()}",
        f"machine: {platform.machine()}",
        f"processor: {platform.processor()}",
        f"python_executable: {sys.executable}",
        "",
        "# Phiên bản gói chính",
    ]
    lines += [f"{name}=={version}" for name, version in package_versions().items()]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def write_json(path: Path, payload: Any) -> Path:
    """Ghi JSON UTF-8 (không dùng ASCII escape để giữ tiếng Việt)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def _default(value: Any) -> Any:
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return float(value)
        if isinstance(value, (np.bool_,)):
            return bool(value)
        if isinstance(value, tuple):
            return list(value)
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
        raise TypeError(f"Không serialize được: {type(value)!r}")

    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_default) + "\n",
        encoding="utf-8",
    )
    return path


def resolve_model_dir(base_dir: Optional[Path] = None) -> Path:
    """Chọn thư mục mô hình đang tồn tại (``Models`` hoặc ``models``).

    Dự án hiện lưu mô hình trong ``Models/`` trong khi ``train.py`` ghi vào
    ``models/``; trên Windows hai đường dẫn trùng nhau, trên Linux/macOS thì
    không. Hàm này ưu tiên thư mục đã tồn tại để tránh tạo thư mục thứ hai.
    """
    root = Path(base_dir) if base_dir is not None else BASE_DIR
    for name in MODEL_DIR_CANDIDATES:
        candidate = root / name
        if candidate.is_dir():
            return candidate
    return root / MODEL_DIR_CANDIDATES[0]


# ---------------------------------------------------------------------------
# 9. Chạy trực tiếp: chỉ để kiểm tra nhanh module
# ---------------------------------------------------------------------------

if __name__ == "__main__":  # pragma: no cover
    print("Không gian tìm kiếm:", SEARCH_SPACE_SIZE, "cấu hình")
    print("Nhiễm sắc thể gốc:", BASELINE_CHROMOSOME)
    print("Siêu tham số gốc:", decode_chromosome(BASELINE_CHROMOSOME))
