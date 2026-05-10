from __future__ import annotations

from typing import Any, Dict

import numpy as np
from scipy.linalg import eigh
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
##修改开始
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, BatchNormalization, Activation, Dropout, MaxPool2D, Layer
from keras.models import Model
from keras.layers import Input, Conv2D, BatchNormalization, Activation, Dropout, AvgPool2D, Flatten, Dense
import numpy as np
##结束
def _rbf_kernel_mean_chunked(
    X_left: np.ndarray,
    X_right: np.ndarray,
    gamma: float,
    chunk_size: int = 128,
) -> float:
    """分块计算 RBF 核均值，避免一次性构造过大的距离矩阵。"""
    X_left = np.asarray(X_left, dtype=np.float32)
    X_right = np.asarray(X_right, dtype=np.float32)
    total = 0.0
    right_sq = np.sum(X_right * X_right, axis=1)

    for start in range(0, len(X_left), chunk_size):
        batch = X_left[start:start + chunk_size]
        batch_sq = np.sum(batch * batch, axis=1, keepdims=True)
        dist = np.maximum(batch_sq + right_sq[None, :] - 2.0 * batch @ X_right.T, 0.0)
        total += np.exp(-gamma * dist).sum(dtype=np.float64)

    return float(total / max(len(X_left) * len(X_right), 1))


def estimate_gamma(
    X_np: np.ndarray,
    sample: int = 160,
    random_state: int = 42,
) -> float:
    """用中位数距离启发式估计 RBF 核的 gamma。"""
    X_np = np.asarray(X_np, dtype=np.float32)
    rng = np.random.default_rng(random_state)

    if len(X_np) > sample:
        X_np = X_np[rng.choice(len(X_np), size=sample, replace=False)]

    sq = np.sum(X_np * X_np, axis=1, keepdims=True)
    dist = np.maximum(sq + sq.T - 2.0 * X_np @ X_np.T, 0.0)
    median_distance = float(np.median(np.sqrt(dist + 1e-8)))
    return 1.0 / (2.0 * median_distance * median_distance + 1e-6)


def compute_mmd(
    X_source: np.ndarray,
    X_target: np.ndarray,
    gamma: float,
    chunk_size: int = 128,
) -> float:
    """根据 RBF 核均值估计源域与目标域之间的 MMD 距离。"""
    k_ss = _rbf_kernel_mean_chunked(X_source, X_source, gamma=gamma, chunk_size=chunk_size)
    k_tt = _rbf_kernel_mean_chunked(X_target, X_target, gamma=gamma, chunk_size=chunk_size)
    k_st = _rbf_kernel_mean_chunked(X_source, X_target, gamma=gamma, chunk_size=chunk_size)
    return float(k_ss + k_tt - 2.0 * k_st)


def _proxy_a_distance(
    X_source: np.ndarray,
    X_target: np.ndarray,
) -> float:
    """使用线性域分类器估计源域与目标域的 Proxy-A-Distance。"""
    X_source = np.asarray(X_source, dtype=np.float32)
    X_target = np.asarray(X_target, dtype=np.float32)
    if len(X_source) == 0 or len(X_target) == 0:
        return 0.0

    X_train = np.vstack([X_source, X_target]).astype(np.float32)
    y_train = np.hstack(
        [
            np.zeros(len(X_source), dtype=np.int32),
            np.ones(len(X_target), dtype=np.int32),
        ]
    )
    if np.unique(y_train).size < 2:
        return 0.0

    try:
        clf = LinearSVC(random_state=0, dual="auto", max_iter=5000)
        clf.fit(X_train, y_train)
        error = float(np.mean(clf.predict(X_train) != y_train))
    except Exception:
        return 0.0
    return float(2.0 * (1.0 - 2.0 * error))


def _estimate_bda_mu(
    Z_source: np.ndarray,
    y_source: np.ndarray,
    Z_target: np.ndarray,
    y_target_pseudo: np.ndarray,
) -> Dict[str, float]:
    """按照传统 BDA 的思路，用边缘/条件域差距自适应估计 mu。"""
    Z_source = np.asarray(Z_source, dtype=np.float32)
    Z_target = np.asarray(Z_target, dtype=np.float32)
    y_source = np.asarray(y_source, dtype=np.int32).reshape(-1)
    y_target_pseudo = np.asarray(y_target_pseudo, dtype=np.int32).reshape(-1)

    pad_marginal = _proxy_a_distance(Z_source, Z_target)
    conditional_distances: list[float] = []
    for class_id in np.unique(y_source):
        source_mask = y_source == class_id
        target_mask = y_target_pseudo == class_id
        if not np.any(source_mask) or not np.any(target_mask):
            continue
        conditional_distances.append(_proxy_a_distance(Z_source[source_mask], Z_target[target_mask]))

    if conditional_distances:
        pad_conditional = float(np.mean(conditional_distances))
        mu = pad_conditional / max(pad_conditional + pad_marginal, 1e-6)
    else:
        pad_conditional = 0.0
        mu = 0.0

    mu = float(np.clip(mu, 0.0, 1.0))
    if mu < 1e-3:
        mu = 0.0
    return {
        "mu": mu,
        "pad_marginal": float(pad_marginal),
        "pad_conditional": float(pad_conditional),
    }


def _prepare_linear_adapter_inputs(
    X_source: np.ndarray,
    y_source: np.ndarray,
    X_target: np.ndarray,
    y_target_debug: np.ndarray | None,
    n_components: int | None,
    random_state: int,
) -> Dict[str, Any]:
    """为线性核迁移方法准备联合输入矩阵与维度信息。"""
    X_source = np.asarray(X_source, dtype=np.float32)
    X_target = np.asarray(X_target, dtype=np.float32)
    y_source = np.asarray(y_source, dtype=np.int32).reshape(-1)
    y_target_debug = None if y_target_debug is None else np.asarray(y_target_debug, dtype=np.int32).reshape(-1)

    if X_source.ndim != 2 or X_target.ndim != 2:
        raise ValueError("迁移适配器输入必须是二维特征矩阵。")
    if len(X_source) != len(y_source):
        raise ValueError("源域特征和标签数量不一致，无法执行迁移学习。")

    X_all = np.vstack([X_source, X_target]).astype(np.float32)
    raw_dim = int(X_all.shape[1])
    work_dim = raw_dim
    pca = None
    if n_components is not None:
        work_dim = min(int(n_components), raw_dim, max(2, X_all.shape[0] - 1))
    if work_dim < raw_dim:
        pca = PCA(n_components=work_dim, svd_solver="randomized", random_state=random_state)
        X_all = pca.fit_transform(X_all).astype(np.float32)

    scaler = StandardScaler()
    X_all = scaler.fit_transform(X_all).astype(np.float32)

    ns = int(X_source.shape[0])
    nt = int(X_target.shape[0])
    X_joint = np.hstack([X_all[:ns].T, X_all[ns:].T]).astype(np.float64)
    X_joint /= np.clip(np.linalg.norm(X_joint, axis=0, keepdims=True), 1e-8, None)

    return {
        "X_joint": X_joint,
        "X_source": X_source,
        "X_target": X_target,
        "y_source": y_source,
        "y_target_debug": y_target_debug,
        "ns": ns,
        "nt": nt,
        "raw_dim": raw_dim,
        "work_dim": int(work_dim),
        "classes": np.unique(y_source),
        "pca": pca,
        "scaler": scaler,
    }


def _build_marginal_mmd_matrix(ns: int, nt: int, class_scale: float = 1.0) -> np.ndarray:
    """构造只包含边缘分布对齐项的 MMD 矩阵。"""
    e = np.vstack(
        (
            np.full((ns, 1), 1.0 / max(ns, 1), dtype=np.float64),
            np.full((nt, 1), -1.0 / max(nt, 1), dtype=np.float64),
        )
    )
    return (e @ e.T) * float(class_scale)


def _build_conditional_mmd_matrix(
    y_source: np.ndarray,
    y_target_pseudo: np.ndarray | None,
    ns: int,
    nt: int,
    classes: np.ndarray,
) -> np.ndarray:
    """根据伪标签构造条件分布对齐项。"""
    matrix_size = ns + nt
    conditional_matrix = np.zeros((matrix_size, matrix_size), dtype=np.float64)
    if y_target_pseudo is None:
        return conditional_matrix

    for class_id in classes:
        e = np.zeros((matrix_size, 1), dtype=np.float64)
        idx_source = np.flatnonzero(y_source == class_id)
        idx_target = np.flatnonzero(y_target_pseudo == class_id)
        if len(idx_source) > 0:
            e[idx_source] = 1.0 / len(idx_source)
        if len(idx_target) > 0:
            e[idx_target + ns] = -1.0 / len(idx_target)
        if len(idx_source) > 0 or len(idx_target) > 0:
            conditional_matrix += e @ e.T
    return conditional_matrix


def _split_latent_domains(latent_joint: np.ndarray, ns: int) -> tuple[np.ndarray, np.ndarray]:
    """把联合潜空间特征拆回源域和目标域。"""
    return (
        np.asarray(latent_joint[:, :ns].T, dtype=np.float32),
        np.asarray(latent_joint[:, ns:].T, dtype=np.float32),
    )


def _project_with_linear_da(
    X_joint: np.ndarray,
    mmd_matrix: np.ndarray,
    latent_dim: int,
    lamb: float,
    reg: float,
    ns: int,
) -> tuple[np.ndarray, np.ndarray]:
    """求解广义特征值问题，并返回投影后的源域与目标域表示。"""
    feature_dim, sample_count = X_joint.shape
    eye = np.eye(feature_dim, dtype=np.float64)
    centering = np.eye(sample_count, dtype=np.float64) - np.ones((sample_count, sample_count), dtype=np.float64) / float(sample_count)
    left = X_joint @ mmd_matrix @ X_joint.T + float(lamb) * eye
    right = X_joint @ centering @ X_joint.T + float(reg) * eye
    eigvals, eigvecs = eigh(left, right)
    order = np.argsort(np.real(eigvals))
    projector = np.real(eigvecs[:, order[:latent_dim]])
    latent_joint = projector.T @ X_joint
    latent_joint /= np.clip(np.linalg.norm(latent_joint, axis=0, keepdims=True), 1e-8, None)
    return _split_latent_domains(latent_joint, ns)


def _solve_linear_da_projection(
    X_joint: np.ndarray,
    mmd_matrix: np.ndarray,
    latent_dim: int,
    lamb: float,
    reg: float,
) -> tuple[np.ndarray, np.ndarray]:
    """求解线性迁移适配器的投影矩阵，并返回标准化后的潜空间联合表示。"""
    feature_dim, sample_count = X_joint.shape
    eye = np.eye(feature_dim, dtype=np.float64)
    centering = np.eye(sample_count, dtype=np.float64) - np.ones((sample_count, sample_count), dtype=np.float64) / float(sample_count)
    left = X_joint @ mmd_matrix @ X_joint.T + float(lamb) * eye
    right = X_joint @ centering @ X_joint.T + float(reg) * eye
    eigvals, eigvecs = eigh(left, right)
    order = np.argsort(np.real(eigvals))
    projector = np.real(eigvecs[:, order[:latent_dim]])
    latent_joint = projector.T @ X_joint
    latent_joint /= np.clip(np.linalg.norm(latent_joint, axis=0, keepdims=True), 1e-8, None)
    return projector, latent_joint


def _fit_pseudo_target_labels_1nn(
    Z_source: np.ndarray,
    y_source: np.ndarray,
    Z_target: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """使用传统 JDA 常见的 1NN 在潜空间中生成目标域伪标签。"""
    pseudo_clf = KNeighborsClassifier(n_neighbors=1)
    pseudo_clf.fit(Z_source, y_source)
    pseudo_labels = pseudo_clf.predict(Z_target).astype(np.int32)
    distances, _ = pseudo_clf.kneighbors(Z_target, n_neighbors=1, return_distance=True)
    confidence = np.clip(1.0 - distances.reshape(-1) / 2.0, 0.0, 1.0).astype(np.float32)
    return pseudo_labels, confidence


def _decision_confidence_from_margin(margin: np.ndarray) -> np.ndarray:
    """把分类间隔映射到 0~1 置信度，便于和 KNN/LR 的概率口径统一。"""
    margin = np.asarray(margin, dtype=np.float64)
    return np.asarray(1.0 / (1.0 + np.exp(-np.clip(np.abs(margin), 0.0, 12.0))), dtype=np.float32)


def _predict_with_confidence(model: Any, X_target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """用统一接口返回伪标签和置信度，兼容 KNN、LR 和 LinearSVC。"""
    X_target = np.asarray(X_target, dtype=np.float32)
    if hasattr(model, "predict_proba"):
        proba = np.asarray(model.predict_proba(X_target), dtype=np.float32)
        class_ids = np.asarray(getattr(model, "classes_", np.arange(proba.shape[1])), dtype=np.int32)
        chosen = np.argmax(proba, axis=1)
        pseudo_labels = class_ids[chosen].astype(np.int32)
        confidence = np.max(proba, axis=1).astype(np.float32)
        return pseudo_labels, confidence

    if hasattr(model, "decision_function"):
        decision = np.asarray(model.decision_function(X_target))
        class_ids = np.asarray(getattr(model, "classes_", np.array([0, 1])), dtype=np.int32)
        if decision.ndim == 1:
            positive_class = int(class_ids[-1])
            negative_class = int(class_ids[0])
            pseudo_labels = np.where(decision >= 0.0, positive_class, negative_class).astype(np.int32)
            confidence = _decision_confidence_from_margin(decision)
            return pseudo_labels, confidence
        top_order = np.argsort(decision, axis=1)
        top1 = top_order[:, -1]
        top2 = top_order[:, -2]
        pseudo_labels = class_ids[top1].astype(np.int32)
        confidence = _decision_confidence_from_margin(
            decision[np.arange(len(decision)), top1] - decision[np.arange(len(decision)), top2]
        )
        return pseudo_labels, confidence

    pseudo_labels = np.asarray(model.predict(X_target), dtype=np.int32)
    confidence = np.ones(len(pseudo_labels), dtype=np.float32)
    return pseudo_labels, confidence


def _fit_pseudo_target_labels(
    Z_source: np.ndarray,
    y_source: np.ndarray,
    Z_target: np.ndarray,
    method: str = "1nn",
    neighbors: int = 3,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """在 JDA 潜空间中生成目标域伪标签，并返回逐样本置信度。"""
    # 中文作用: 允许在 1NN、距离加权 KNN、LR、LinearSVC 之间切换伪标签器，比较哪种伪标签更稳。
    method = str(method).lower().strip()
    neighbors = max(1, int(neighbors))
    if method == "1nn":
        pseudo_clf = KNeighborsClassifier(n_neighbors=1)
        pseudo_clf.fit(Z_source, y_source)
        pseudo_labels = pseudo_clf.predict(Z_target).astype(np.int32)
        distances, _ = pseudo_clf.kneighbors(Z_target, n_neighbors=1, return_distance=True)
        confidence = np.asarray(1.0 / (1.0 + distances.reshape(-1)), dtype=np.float32)
        return pseudo_labels, confidence
    if method == "knn":
        pseudo_clf = KNeighborsClassifier(n_neighbors=neighbors, weights="distance")
    elif method == "lr":
        pseudo_clf = LogisticRegression(
            random_state=int(random_state),
            solver="liblinear",
            class_weight="balanced",
            max_iter=2000,
        )
    elif method == "svm":
        pseudo_clf = LinearSVC(
            random_state=int(random_state),
            dual="auto",
            class_weight="balanced",
            max_iter=5000,
        )
    else:
        raise ValueError(f"不支持的 JDA 伪标签分类器: {method!r}")

    pseudo_clf.fit(Z_source, y_source)
    return _predict_with_confidence(pseudo_clf, Z_target)


def _coral_covariance_gap(
    X_source: np.ndarray,
    X_target: np.ndarray,
) -> float:
    """计算源域与目标域协方差矩阵的 Frobenius 差值，便于观察 CORAL 是否真的拉近二阶统计量。"""
    X_source = np.asarray(X_source, dtype=np.float32)
    X_target = np.asarray(X_target, dtype=np.float32)
    if len(X_source) <= 1 or len(X_target) <= 1:
        return 0.0

    cov_source = np.cov(X_source.T) + np.eye(X_source.shape[1], dtype=np.float64)
    cov_target = np.cov(X_target.T) + np.eye(X_target.shape[1], dtype=np.float64)
    return float(np.linalg.norm(cov_source - cov_target, ord="fro"))


def _fit_coral_source_transform(
    X_source: np.ndarray,
    X_target: np.ndarray,
) -> np.ndarray:
    """按照传统 CORAL 的公式，把源域特征映射到更接近目标域协方差的空间。"""
    X_source = np.asarray(X_source, dtype=np.float32)
    X_target = np.asarray(X_target, dtype=np.float32)
    cov_source = np.cov(X_source.T) + np.eye(X_source.shape[1], dtype=np.float64)
    cov_target = np.cov(X_target.T) + np.eye(X_target.shape[1], dtype=np.float64)
    from scipy.linalg import fractional_matrix_power
    # CORAL 的核心是先对白化源域，再着色到目标域协方差。
    coral_transform = np.real(
        fractional_matrix_power(cov_source, -0.5) @ fractional_matrix_power(cov_target, 0.5)
    )

    # CORAL 的核心就是先对白化源域，再着色到目标域协方差。
    coral_transform = np.real(
        fractional_matrix_power(cov_source, -0.5) @ fractional_matrix_power(cov_target, 0.5)
    )
    return np.asarray(X_source @ coral_transform, dtype=np.float32)


def _debug_latent_gap(
    tag: str,
    Z_source: np.ndarray,
    Z_target: np.ndarray,
    random_state: int,
    iteration: int,
    y_target_pseudo: np.ndarray | None = None,
    confidence: np.ndarray | None = None,
    y_target_debug: np.ndarray | None = None,
) -> Dict[str, float]:
    """输出潜空间里的 MMD 与伪标签稳定性，帮助定位负迁移来源。"""
    rng = np.random.default_rng(random_state + iteration * 11 + 1)
    debug_source = Z_source
    debug_target = Z_target
    if len(debug_source) > 192:
        debug_source = debug_source[rng.choice(len(debug_source), size=192, replace=False)]
    if len(debug_target) > 192:
        debug_target = debug_target[rng.choice(len(debug_target), size=192, replace=False)]

    debug_gamma = estimate_gamma(
        np.vstack([debug_source, debug_target]),
        sample=min(160, len(debug_source) + len(debug_target)),
        random_state=random_state + iteration * 11 + 1,
    )
    latent_mmd = compute_mmd(
        debug_source,
        debug_target,
        gamma=debug_gamma,
        chunk_size=min(128, max(32, len(debug_source))),
    )

    history_row: Dict[str, float] = {
        "iteration": float(iteration + 1),
        "mmd": float(latent_mmd),
    }
    if y_target_pseudo is not None and confidence is not None:
        history_row["confidence_mean"] = float(np.mean(confidence))
        history_row["confidence_min"] = float(np.min(confidence))
        history_row["pseudo_class_ratio"] = float(np.mean(y_target_pseudo))
        if y_target_debug is not None and len(y_target_debug) == len(y_target_pseudo):
            history_row["pseudo_acc"] = float(accuracy_score(y_target_debug, y_target_pseudo))

    if y_target_pseudo is None or confidence is None:
        print(f"[{tag}] iter={iteration + 1} latent_mmd={latent_mmd:.6f}")
    else:
        pseudo_acc_text = ""
        if y_target_debug is not None and len(y_target_debug) == len(y_target_pseudo):
            pseudo_acc_text = f" pseudo_acc={accuracy_score(y_target_debug, y_target_pseudo):.4f}"
        print(
            f"[{tag}] iter={iteration + 1} latent_mmd={latent_mmd:.6f} "
            f"conf_mean={np.mean(confidence):.4f} conf_min={np.min(confidence):.4f} "
            f"positive_ratio={np.mean(y_target_pseudo):.4f}{pseudo_acc_text}"
        )
    return history_row

# 梯度反转层（GRL）
class GradientReversalLayer(Layer):
    def __init__(self, lambda_=1.0, **kwargs):
        super().__init__(**kwargs)
        self.lambda_ = lambda_

    def call(self, x):
        @tf.custom_gradient
        def reverse(x):
            def grad(dy):
                return -self.lambda_ * dy
            return x, grad
        return reverse(x)

    def get_config(self):
        config = super().get_config()
        config.update({"lambda_": self.lambda_})
        return config

class DANNAdapter:
    """DANN 域自适应适配器，替代 JDA 使用。"""
    def __init__(self, input_dim, encoding_dim=64, lambda_=1.0, epochs=20, batch_size=32, verbose=True):
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        self.lambda_ = lambda_
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.feature_extractor = None

    def _build_feature_extractor(self, inputs):
        """构建 2D-CNN 特征提取器（轻量版）"""
        x = Conv2D(16, (1, 3), padding='same')(inputs)
        x = BatchNormalization()(x)
        x = Activation('elu')(x)
        x = MaxPool2D((1, 2))(x)
        x = Conv2D(32, (1, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('elu')(x)
        x = MaxPool2D((1, 2))(x)
        x = Dropout(0.5)(x)
        x = Flatten()(x)
        features = Dense(self.encoding_dim, activation='elu', name='feature_layer')(x)
        return features

    def _build_dann(self):
        inputs = Input(shape=(1, self.input_dim, 1))
        features = self._build_feature_extractor(inputs)

        # 标签分类器（源域监督）
        label_out = Dense(1, activation='sigmoid', name='label_clf')(features)

        # 域判别器（带 GRL）
        grl = GradientReversalLayer(lambda_=self.lambda_)(features)
        domain_out = Dense(64, activation='elu')(grl)
        domain_out = Dense(1, activation='sigmoid', name='domain_clf')(domain_out)

        model = Model(inputs, [label_out, domain_out])
        model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-4),
            loss={'label_clf': 'binary_crossentropy', 'domain_clf': 'binary_crossentropy'},
            loss_weights={'label_clf': 1.0, 'domain_clf': 1.0}
        )
        return model

    def fit_transform(self, X_source, y_source, X_target, Yt_target=None):
        # 重塑为 4D
        X_source_4d = X_source.reshape(-1, 1, X_source.shape[1], 1).astype(np.float32)
        X_target_4d = X_target.reshape(-1, 1, X_target.shape[1], 1).astype(np.float32)

        # 域标签：源域=0，目标域=1
        y_domain_source = np.zeros(len(X_source_4d), dtype=np.float32)
        y_domain_target = np.ones(len(X_target_4d), dtype=np.float32)

        # 合并数据
        X_combined = np.vstack([X_source_4d, X_target_4d])
        y_label_combined = np.hstack([y_source, np.zeros(len(X_target_4d))])  # 目标域标签占位
        y_domain_combined = np.hstack([y_domain_source, y_domain_target])

        # 样本权重：目标域部分不参与标签分类损失，域判别器损失全参与
        sample_weight_label = np.hstack([np.ones(len(y_source)), np.zeros(len(X_target_4d))])
        sample_weight = {
            'label_clf': sample_weight_label,
            'domain_clf': np.ones(len(X_combined))  # 可选，不加也可
        }

        dann = self._build_dann()
        dann.fit(
            X_combined,
            {'label_clf': y_label_combined, 'domain_clf': y_domain_combined},
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=self.verbose
        )

        # 提取特征提取器
        self.feature_extractor = Model(dann.input, dann.get_layer('feature_layer').output)

        Z_source = self.feature_extractor.predict(X_source_4d)
        Z_target = self.feature_extractor.predict(X_target_4d)
        return Z_source, Z_target

    def transform(self, X):
        X_4d = X.reshape(-1, 1, X.shape[1], 1).astype(np.float32)
        return self.feature_extractor.predict(X_4d)

class JDA_Linear:
    # 中文作用: 在共享潜空间里同时缩小源域/目标域的边缘分布与条件分布差异。
    """更贴近传统 JDA 的线性核实现，伪标签阶段固定使用 1NN。"""

    def __init__(
        self,
        dim: int = 96,
        T: int = 10,
        gamma: float | None = None,
        n_components: int = 160,
        kernel_type: str = "linear",
        lamb: float = 1.0,
        reg: float = 1e-6,
        pseudo_labeler: str = "1nn",
        pseudo_neighbors: int = 3,
        pseudo_change_tol: float = 1e-3,
        mmd_delta_tol: float = 1e-3,
        confidence_delta_tol: float = 5e-3,
        early_stop_patience: int = 2,
        min_iterations: int = 2,
        random_state: int = 42,
        verbose: bool = True,
    ) -> None:
        # 中文作用: 设置 JDA 的维度、迭代轮数以及收敛相关超参数。
        """初始化投影维度、迭代次数和线性核配置。"""
        self.dim = dim
        self.T = T
        self.gamma = gamma
        self.n_components = n_components
        self.kernel_type = kernel_type
        self.lamb = lamb
        self.reg = reg
        self.pseudo_labeler = str(pseudo_labeler).lower().strip()
        self.pseudo_neighbors = max(1, int(pseudo_neighbors))
        self.random_state = random_state
        self.verbose = verbose
        self.iteration_history: list[Dict[str, float]] = []
        self.projector_: np.ndarray | None = None
        self.scaler_: StandardScaler | None = None
        self.pca_: PCA | None = None
        self.work_dim_: int | None = None
        self.latent_dim_: int | None = None
        self.selected_target_pseudo_labels_: np.ndarray | None = None
        self.selected_target_confidence_: np.ndarray | None = None
        self.pseudo_change_tol: float = float(max(0.0, pseudo_change_tol))
        self.mmd_delta_tol: float = float(max(0.0, mmd_delta_tol))
        self.confidence_delta_tol: float = float(max(0.0, confidence_delta_tol))
        self.early_stop_patience: int = int(max(1, early_stop_patience))
        self.min_iterations: int = int(max(1, min_iterations))
        self.selected_iteration_: int = 0

    def fit_transform(
        self,
        X_source: np.ndarray,
        y_source: np.ndarray,
        X_target: np.ndarray,
        Yt_target: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        # 中文作用: 通过“伪标签迭代 + MMD 最小化”把源域和目标域映射到同一潜空间。
        """执行联合分布对齐，并返回对齐后的源域和目标域特征。"""
        if self.kernel_type not in {"linear", "primal"}:
            raise ValueError(f"当前仅支持线性核 JDA，收到 kernel_type={self.kernel_type!r}")

        prepared = _prepare_linear_adapter_inputs(
            X_source=X_source,
            y_source=y_source,
            X_target=X_target,
            y_target_debug=Yt_target,
            n_components=self.n_components,
            random_state=self.random_state,
        )
        X_joint = prepared["X_joint"]
        y_source = prepared["y_source"]
        ns = prepared["ns"]
        nt = prepared["nt"]
        classes = prepared["classes"]
        latent_dim = min(int(self.dim), int(X_joint.shape[0]))
        marginal_matrix = _build_marginal_mmd_matrix(ns, nt, class_scale=float(len(classes)))
        self.scaler_ = prepared["scaler"]
        self.pca_ = prepared["pca"]
        self.work_dim_ = int(prepared["work_dim"])
        self.latent_dim_ = int(latent_dim)
        self.projector_ = None

        y_target_pseudo = None
        prev_pseudo = None
        prev_mmd = None
        prev_confidence_mean = None
        stable_rounds = 0
        best_state_key = None
        best_state: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None
        best_iteration_index = 0
        self.iteration_history = []
        self.selected_iteration_ = 0
        self.selected_target_pseudo_labels_ = None
        self.selected_target_confidence_ = None
        if self.verbose:
            print(
                f"[JDA] kernel={self.kernel_type} iterations={self.T} "
                f"pseudo={self.pseudo_labeler}(k={self.pseudo_neighbors}) "
                f"source_samples={ns} target_samples={nt} input_dim={prepared['raw_dim']} "
                f"work_dim={prepared['work_dim']} latent_dim={latent_dim}"
            )

        for iteration in range(self.T):
            mmd_matrix = marginal_matrix.copy()
            if y_target_pseudo is not None:
                mmd_matrix += _build_conditional_mmd_matrix(y_source, y_target_pseudo, ns, nt, classes)
            fro_norm = np.linalg.norm(mmd_matrix, ord="fro")
            if fro_norm > 0:
                mmd_matrix /= fro_norm

            projector, latent_joint = _solve_linear_da_projection(
                X_joint=X_joint,
                mmd_matrix=mmd_matrix,
                latent_dim=latent_dim,
                lamb=self.lamb,
                reg=self.reg,
            )
            Z_source, Z_target = _split_latent_domains(latent_joint, ns)
            self.projector_ = projector
            y_target_pseudo, confidence = _fit_pseudo_target_labels(
                Z_source,
                y_source,
                Z_target,
                method=self.pseudo_labeler,
                neighbors=self.pseudo_neighbors,
                random_state=self.random_state + iteration * 17 + 5,
            )
            change_ratio = 1.0 if prev_pseudo is None else float(np.mean(prev_pseudo != y_target_pseudo))
            history_row = _debug_latent_gap(
                tag="JDA",
                Z_source=Z_source,
                Z_target=Z_target,
                random_state=self.random_state,
                iteration=iteration,
                y_target_pseudo=y_target_pseudo,
                confidence=confidence,
                y_target_debug=prepared["y_target_debug"],
            )
            history_row["pseudo_change_ratio"] = float(change_ratio)
            current_mmd = float(history_row.get("mmd", 0.0))
            mmd_delta = 0.0 if prev_mmd is None else float(current_mmd - prev_mmd)
            history_row["mmd_delta_vs_prev"] = float(mmd_delta)
            confidence_mean = float(history_row.get("confidence_mean", 0.0))
            confidence_delta = 0.0 if prev_confidence_mean is None else float(confidence_mean - prev_confidence_mean)
            history_row["confidence_delta_vs_prev"] = float(confidence_delta)
            # 中文作用: 论文里写“识别准确率收敛时结束”，这里用“伪标签变化趋稳 + MMD 变化趋稳”近似实现可执行的收敛停止条件。
            pseudo_stable = prev_pseudo is not None and float(change_ratio) <= float(self.pseudo_change_tol)
            mmd_stable = prev_mmd is not None and abs(float(mmd_delta)) <= float(self.mmd_delta_tol)
            confidence_stable = prev_confidence_mean is not None and abs(float(confidence_delta)) <= float(self.confidence_delta_tol)
            if iteration + 1 >= int(self.min_iterations) and pseudo_stable and mmd_stable and confidence_stable:
                stable_rounds += 1
            else:
                stable_rounds = 0
            history_row["stable_rounds"] = float(stable_rounds)
            history_row["selected"] = 0.0
            self.iteration_history.append(history_row)
            candidate_key = (
                -float(current_mmd),
                float(confidence_mean),
                -float(change_ratio),
                -abs(float(history_row.get("pseudo_class_ratio", 0.0)) - float(np.mean(y_source))),
            )
            if best_state_key is None or candidate_key > best_state_key:
                best_state_key = candidate_key
                best_state = (
                    np.asarray(Z_source, dtype=np.float32),
                    np.asarray(Z_target, dtype=np.float32),
                    np.asarray(projector, dtype=np.float64),
                    np.asarray(y_target_pseudo, dtype=np.int32),
                    np.asarray(confidence, dtype=np.float32),
                )
                best_iteration_index = len(self.iteration_history) - 1

            if self.verbose:
                print(
                    f"[JDA] iter={iteration + 1}/{self.T} pseudo_change={change_ratio:.4f} "
                    f"mmd_delta={mmd_delta:+.4f} conf_delta={confidence_delta:+.4f} "
                    f"stable_rounds={stable_rounds}"
                )
            prev_pseudo = y_target_pseudo.copy()
            prev_mmd = float(current_mmd)
            prev_confidence_mean = float(confidence_mean)
            if stable_rounds >= int(self.early_stop_patience):
                if self.verbose:
                    print(
                        f"[JDA] early_stop iter={iteration + 1} "
                        f"reason=pseudo_mmd_conf_converged patience={self.early_stop_patience}"
                    )
                break

        if best_state is not None:
            self.iteration_history[best_iteration_index]["selected"] = 1.0
            self.selected_iteration_ = int(self.iteration_history[best_iteration_index]["iteration"])
            best_Z_source, best_Z_target, best_projector, best_pseudo_labels, best_confidence = best_state
            self.projector_ = best_projector
            self.selected_target_pseudo_labels_ = best_pseudo_labels
            self.selected_target_confidence_ = best_confidence
            if self.verbose:
                selected_row = self.iteration_history[best_iteration_index]
                print(
                    f"[JDA] select_best iter={self.selected_iteration_} "
                    f"mmd={float(selected_row.get('mmd', 0.0)):.4f} "
                    f"conf_mean={float(selected_row.get('confidence_mean', 0.0)):.4f} "
                    f"pseudo_change={float(selected_row.get('pseudo_change_ratio', 0.0)):.4f}"
                )
            return best_Z_source, best_Z_target

        return Z_source, Z_target

    def transform(self, X: np.ndarray) -> np.ndarray:
        # 中文作用: 把验证集或测试集映射到已经学好的 JDA 潜空间，保证评估口径一致。
        """把新的域样本映射到当前 JDA 已学习好的潜空间。"""
        if self.projector_ is None or self.scaler_ is None:
            raise ValueError("JDA_Linear 还没有完成 fit_transform，无法直接 transform 新样本。")

        X = np.asarray(X, dtype=np.float32)
        X_work = X
        if self.pca_ is not None:
            X_work = np.asarray(self.pca_.transform(X_work), dtype=np.float32)
        X_work = np.asarray(self.scaler_.transform(X_work), dtype=np.float32)
        X_joint = X_work.T.astype(np.float64)
        X_joint /= np.clip(np.linalg.norm(X_joint, axis=0, keepdims=True), 1e-8, None)
        latent_joint = self.projector_.T @ X_joint
        latent_joint /= np.clip(np.linalg.norm(latent_joint, axis=0, keepdims=True), 1e-8, None)
        return np.asarray(latent_joint.T, dtype=np.float32)


class TCA_Linear:
    """使用线性核 TCA 的轻量迁移适配器。"""

    def __init__(
        self,
        dim: int = 96,
        T: int = 1,
        gamma: float | None = None,
        n_components: int = 160,
        kernel_type: str = "linear",
        lamb: float = 1.0,
        reg: float = 1e-6,
        random_state: int = 42,
        verbose: bool = True,
    ) -> None:
        """初始化 TCA 的投影维度、近似维度和随机种子。"""
        self.dim = dim
        self.T = T
        self.gamma = gamma
        self.n_components = n_components
        self.kernel_type = kernel_type
        self.lamb = lamb
        self.reg = reg
        self.random_state = random_state
        self.verbose = verbose
        self.iteration_history: list[Dict[str, float]] = []

    def fit_transform(
        self,
        X_source: np.ndarray,
        y_source: np.ndarray,
        X_target: np.ndarray,
        Yt_target: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """执行只包含边缘分布对齐的 TCA。"""
        if self.kernel_type not in {"linear", "primal"}:
            raise ValueError(f"TCA 当前仅支持线性核，收到 kernel_type={self.kernel_type!r}")

        prepared = _prepare_linear_adapter_inputs(
            X_source=X_source,
            y_source=y_source,
            X_target=X_target,
            y_target_debug=Yt_target,
            n_components=self.n_components,
            random_state=self.random_state,
        )
        X_joint = prepared["X_joint"]
        ns = prepared["ns"]
        nt = prepared["nt"]
        latent_dim = min(int(self.dim), int(X_joint.shape[0]))
        mmd_matrix = _build_marginal_mmd_matrix(ns, nt, class_scale=1.0)
        fro_norm = np.linalg.norm(mmd_matrix, ord="fro")
        if fro_norm > 0:
            mmd_matrix /= fro_norm

        if self.verbose:
            print(
                f"[TCA] kernel={self.kernel_type} source_samples={ns} "
                f"target_samples={nt} input_dim={prepared['raw_dim']} "
                f"work_dim={prepared['work_dim']} latent_dim={latent_dim}"
            )

        Z_source, Z_target = _project_with_linear_da(
            X_joint=X_joint,
            mmd_matrix=mmd_matrix,
            latent_dim=latent_dim,
            lamb=self.lamb,
            reg=self.reg,
            ns=ns,
        )
        self.iteration_history = [
            _debug_latent_gap(
                tag="TCA",
                Z_source=Z_source,
                Z_target=Z_target,
                random_state=self.random_state,
                iteration=0,
                y_target_debug=prepared["y_target_debug"],
            )
        ]
        return Z_source, Z_target


class CORAL_Adapter:
    """使用 CORAL 做二阶统计量对齐，只变换源域特征，目标域保持原表示。"""

    def __init__(
        self,
        dim: int = 96,
        T: int = 1,
        gamma: float | None = None,
        n_components: int = 160,
        kernel_type: str = "linear",
        lamb: float = 1.0,
        reg: float = 1e-6,
        random_state: int = 42,
        verbose: bool = True,
    ) -> None:
        """保留统一构造参数签名，便于和其他迁移适配器共用入口。"""
        self.dim = dim
        self.T = T
        self.gamma = gamma
        self.n_components = n_components
        self.kernel_type = kernel_type
        self.lamb = lamb
        self.reg = reg
        self.random_state = random_state
        self.verbose = verbose
        self.iteration_history: list[Dict[str, float]] = []

    def fit_transform(
        self,
        X_source: np.ndarray,
        y_source: np.ndarray,
        X_target: np.ndarray,
        Yt_target: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """执行 CORAL 协方差对齐，并打印对齐前后的协方差差值与 MMD 诊断。"""
        del y_source
        X_source = np.asarray(X_source, dtype=np.float32)
        X_target = np.asarray(X_target, dtype=np.float32)
        cov_gap_before = _coral_covariance_gap(X_source, X_target)
        before_history_row = _debug_latent_gap(
            tag="CORAL-Before",
            Z_source=X_source,
            Z_target=X_target,
            random_state=self.random_state,
            iteration=0,
            y_target_debug=Yt_target,
        )
        Z_source = _fit_coral_source_transform(X_source, X_target)
        Z_target = X_target.astype(np.float32)
        cov_gap_after = _coral_covariance_gap(Z_source, Z_target)
        history_row = _debug_latent_gap(
            tag="CORAL",
            Z_source=Z_source,
            Z_target=Z_target,
            random_state=self.random_state,
            iteration=0,
            y_target_debug=Yt_target,
        )
        history_row["mmd_before"] = float(before_history_row.get("mmd", 0.0))
        history_row["cov_gap_before"] = float(cov_gap_before)
        history_row["cov_gap_after"] = float(cov_gap_after)
        self.iteration_history = [history_row]
        if self.verbose:
            print(
                f"[CORAL] source_samples={len(X_source)} target_samples={len(X_target)} "
                f"input_dim={X_source.shape[1]} mmd_before={history_row['mmd_before']:.6f} "
                f"mmd_after={history_row['mmd']:.6f} cov_gap_before={cov_gap_before:.6f} "
                f"cov_gap_after={cov_gap_after:.6f}"
            )
        return Z_source, Z_target


class BDA_Linear:
    """使用线性核 BDA 的轻量迁移适配器，伪标签阶段同样固定使用 1NN。"""

    def __init__(
        self,
        dim: int = 96,
        T: int = 10,
        gamma: float | None = None,
        n_components: int = 160,
        kernel_type: str = "linear",
        lamb: float = 1.0,
        mu: float = 0.5,
        estimate_mu: bool = True,
        reg: float = 1e-6,
        random_state: int = 42,
        verbose: bool = True,
    ) -> None:
        """初始化 BDA 的投影维度、平衡系数和迭代次数。"""
        self.dim = dim
        self.T = T
        self.gamma = gamma
        self.n_components = n_components
        self.kernel_type = kernel_type
        self.lamb = lamb
        self.mu = mu
        self.estimate_mu = estimate_mu
        self.reg = reg
        self.random_state = random_state
        self.verbose = verbose
        self.iteration_history: list[Dict[str, float]] = []

    def fit_transform(
        self,
        X_source: np.ndarray,
        y_source: np.ndarray,
        X_target: np.ndarray,
        Yt_target: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """交替更新伪标签与分布矩阵，执行 BDA 对齐。"""
        if self.kernel_type not in {"linear", "primal"}:
            raise ValueError(f"BDA 当前仅支持线性核，收到 kernel_type={self.kernel_type!r}")

        prepared = _prepare_linear_adapter_inputs(
            X_source=X_source,
            y_source=y_source,
            X_target=X_target,
            y_target_debug=Yt_target,
            n_components=self.n_components,
            random_state=self.random_state,
        )
        X_joint = prepared["X_joint"]
        y_source = prepared["y_source"]
        ns = prepared["ns"]
        nt = prepared["nt"]
        classes = prepared["classes"]
        latent_dim = min(int(self.dim), int(X_joint.shape[0]))
        marginal_matrix = _build_marginal_mmd_matrix(ns, nt, class_scale=1.0)

        y_target_pseudo = None
        prev_pseudo = None
        prev_Z_source = None
        prev_Z_target = None
        self.iteration_history = []
        if self.verbose:
            print(
                f"[BDA] kernel={self.kernel_type} iterations={self.T} "
                f"mu={'auto' if self.estimate_mu else f'{self.mu:.2f}'} pseudo=1NN "
                f"source_samples={ns} target_samples={nt} input_dim={prepared['raw_dim']} "
                f"work_dim={prepared['work_dim']} latent_dim={latent_dim}"
            )

        for iteration in range(self.T):
            conditional_matrix = _build_conditional_mmd_matrix(y_source, y_target_pseudo, ns, nt, classes)
            mu_info = {
                "mu": float(self.mu),
                "pad_marginal": 0.0,
                "pad_conditional": 0.0,
            }
            if self.estimate_mu:
                if prev_Z_source is not None and prev_Z_target is not None and y_target_pseudo is not None:
                    mu_info = _estimate_bda_mu(prev_Z_source, y_source, prev_Z_target, y_target_pseudo)
                else:
                    mu_info["mu"] = 0.0
            current_mu = float(mu_info["mu"])
            mmd_matrix = (1.0 - current_mu) * marginal_matrix + current_mu * conditional_matrix
            fro_norm = np.linalg.norm(mmd_matrix, ord="fro")
            if fro_norm > 0:
                mmd_matrix /= fro_norm

            Z_source, Z_target = _project_with_linear_da(
                X_joint=X_joint,
                mmd_matrix=mmd_matrix,
                latent_dim=latent_dim,
                lamb=self.lamb,
                reg=self.reg,
                ns=ns,
            )
            y_target_pseudo, confidence = _fit_pseudo_target_labels_1nn(Z_source, y_source, Z_target)
            change_ratio = 1.0 if prev_pseudo is None else float(np.mean(prev_pseudo != y_target_pseudo))
            history_row = _debug_latent_gap(
                tag="BDA",
                Z_source=Z_source,
                Z_target=Z_target,
                random_state=self.random_state,
                iteration=iteration,
                y_target_pseudo=y_target_pseudo,
                confidence=confidence,
                y_target_debug=prepared["y_target_debug"],
            )
            history_row["mu"] = float(current_mu)
            history_row["pad_marginal"] = float(mu_info["pad_marginal"])
            history_row["pad_conditional"] = float(mu_info["pad_conditional"])
            history_row["pseudo_change_ratio"] = float(change_ratio)
            self.iteration_history.append(history_row)
            if self.verbose:
                print(
                    f"[BDA] iter={iteration + 1}/{self.T} mu={current_mu:.4f} "
                    f"pad_m={mu_info['pad_marginal']:.4f} pad_c={mu_info['pad_conditional']:.4f} "
                    f"pseudo_change={change_ratio:.4f}"
                )
            prev_pseudo = y_target_pseudo.copy()
            prev_Z_source = np.asarray(Z_source, dtype=np.float32)
            prev_Z_target = np.asarray(Z_target, dtype=np.float32)

        return Z_source, Z_target

##修改（加入2DCNN分支）
def build_domain_adapter(
        method: str,
        dim: int = 96,
        T: int = 10,
        gamma: float | None = None,
        n_components: int = 160,
        kernel_type: str = "linear",
        lamb: float = 1.0,
        bda_mu: float = 0.5,
        bda_estimate_mu: bool = True,
        reg: float = 1e-6,
        pseudo_labeler: str = "1nn",
        pseudo_neighbors: int = 3,
        pseudo_change_tol: float = 1e-3,
        mmd_delta_tol: float = 1e-3,
        confidence_delta_tol: float = 5e-3,
        early_stop_patience: int = 2,
        min_iterations: int = 2,
        random_state: int = 42,
        verbose: bool = True,
        **kwargs,
) -> Any:
    """按名称构造迁移适配器，统一 TCA/JDA/BDA/CORAL/TwoDCNN/DANN 的调用入口。"""
    method_name = str(method).lower()

    common_kwargs = {
        "dim": dim,
        "T": T,
        "gamma": gamma,
        "n_components": n_components,
        "kernel_type": kernel_type,
        "lamb": lamb,
        "reg": reg,
        "random_state": random_state,
        "verbose": verbose,
    }
    jda_kwargs = {
        "pseudo_labeler": pseudo_labeler,
        "pseudo_neighbors": pseudo_neighbors,
        "pseudo_change_tol": pseudo_change_tol,
        "mmd_delta_tol": mmd_delta_tol,
        "confidence_delta_tol": confidence_delta_tol,
        "early_stop_patience": early_stop_patience,
        "min_iterations": min_iterations,
    }

    if method_name == "jda":
        return JDA_Linear(**common_kwargs, **jda_kwargs)
    if method_name == "tca":
        return TCA_Linear(**common_kwargs)
    if method_name == "coral":
        return CORAL_Adapter(**common_kwargs)
    if method_name == "bda":
        return BDA_Linear(mu=bda_mu, estimate_mu=bda_estimate_mu, **common_kwargs)

    # TwoD_CNN 分支
    if method_name == "twod_cnn":
        input_dim = kwargs.get('input_dim')
        if input_dim is None:
            raise ValueError("使用 'twod_cnn' 必须提供 input_dim 参数。")
        return TwoDCNNAdapter(
            input_dim=input_dim,
            encoding_dim=kwargs.get('encoding_dim', 100),
            epochs=kwargs.get('epochs', 20),
            batch_size=kwargs.get('batch_size', 32),
            verbose=verbose,
        )

    # DANN 分支
    if method_name == "dann":
        input_dim = kwargs.get('input_dim')
        if input_dim is None:
            raise ValueError("使用 'dann' 必须提供 input_dim 参数。")
        return DANNAdapter(
            input_dim=input_dim,
            encoding_dim=kwargs.get('encoding_dim', 64),
            lambda_=kwargs.get('lambda_', 1.0),
            epochs=kwargs.get('epochs', 20),
            batch_size=kwargs.get('batch_size', 32),
            verbose=verbose,
        )

    # 如果所有分支都不匹配，抛出错误
    raise ValueError(
        f"未知迁移方法: {method!r}，目前仅支持 "
        f"'jda'、'tca'、'bda'、'coral'、'twod_cnn'、'dann'。"
    )



##修改##



class TwoDCNNAdapter:
    """使用 TwoD_CNN 作为特征提取器的域适配器（源域监督训练）。"""

    def __init__(self, input_dim, encoding_dim=100, epochs=20, batch_size=32, verbose=True):
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.feature_extractor = None

    def _build_extractor(self, inputs=None):
        """构建特征提取器，返回瓶颈层输出（命名）"""
        if inputs is None:
            inputs = Input(shape=(1, self.input_dim, 1))
        x = Conv2D(64, (1, 3), padding='same')(inputs)
        x = BatchNormalization()(x)
        x = Activation('elu')(x)
        x = Conv2D(64, (1, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('elu')(x)
        x = Conv2D(64, (1, 3), padding='same')(x)
        x = Dropout(0.5)(x)
        x = AvgPool2D(pool_size=(2, 1), padding='same')(x)
        x = Flatten()(x)
        x = Dense(self.encoding_dim, activation='elu', name='feature_layer')(x)  # 显式命名
        return x

    def fit_transform(self, X_source, y_source, X_target, Yt_target=None):
        X_source_4d = X_source.reshape(-1, 1, X_source.shape[1], 1).astype(np.float32)
        X_target_4d = X_target.reshape(-1, 1, X_target.shape[1], 1).astype(np.float32)

        # 构建完整模型用于训练
        inputs = Input(shape=(1, self.input_dim, 1))
        features = self._build_extractor(inputs)  # 直接得到带命名的输出张量
        outputs = Dense(1, activation='sigmoid', name='label_clf')(features)
        classifier = Model(inputs, outputs)
        classifier.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        classifier.fit(X_source_4d, y_source, epochs=self.epochs, batch_size=self.batch_size, verbose=self.verbose)

        # 提取特征提取器模型（输入到瓶颈层）
        self.feature_extractor = Model(inputs, features)  # 直接使用features张量，无需get_layer
        Z_source = self.feature_extractor.predict(X_source_4d)
        Z_target = self.feature_extractor.predict(X_target_4d)
        return Z_source, Z_target

    def transform(self, X):
        X_4d = X.reshape(-1, 1, X.shape[1], 1).astype(np.float32)
        return self.feature_extractor.predict(X_4d)
if __name__ == "__main__":
    print(build_domain_adapter("dann", input_dim=100))