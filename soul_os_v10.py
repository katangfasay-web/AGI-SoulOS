"""Grand Soul Operating System v10.0 implementation.

This module contains a direct translation of the specification provided for
the "靈魂運算系統 v10.0" cognitive architecture.  The design is intentionally
faithful to the narrative description, featuring a collection of loosely
coupled managers that cooperate to minimise a generalised free energy
objective.

The code is heavily documented in Traditional Chinese in order to mirror the
original notes from the author.  No additional functionality has been added
or removed; the goal of this script is simply to make the specification
executable inside this repository so that interested readers can experiment
with it immediately.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

import numpy as np


# --- 輔助函數 ---
def _sigmoid(x: np.ndarray | float, kappa: float = 1.0, tau: float = 0.0) -> np.ndarray:
    """帶有斜率和閾值的 Sigmoid 函數."""

    return 1 / (1 + np.exp(-kappa * (x - tau)))


def _softmax(x: np.ndarray) -> np.ndarray:
    """穩定的 Softmax 函數."""

    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


# --- 核心參數與設定 ---
class SystemConfig:
    """存放系統的超參數和設定."""

    # 維度設定
    STATE_DIM = 128
    FEATURE_DIM = 64
    BELIEF_DIM = 32
    EMBEDDING_DIM = 32
    ACTION_DIM = 5

    # (9) 信念更新參數
    LAMBDA_B_DECAY = 0.02  # λ: 信念衰減率
    ETA_B_LEARNING = 0.1  # η: 信念學習率
    ALPHA_B_FREE_ENERGY = 0.1  # α: 自由能對信念的影響
    BETA_B_ERROR = 0.3  # β: 預測誤差對信念的影響
    GAMMA_B_GAIN = 0.05  # γ: 意識增益對信念的影響

    # (10) 人格/先驗更新參數
    ALPHA_P_ACTION = 0.01  # α: 行動總量對人格的影響
    BETA_P_EMOTION = 0.02  # β: 經驗強度對人格的影響
    CHI_P_RESONANCE = 0.05  # χ: 共振對人格的影響

    # (12) 核心參數更新率
    RHO_THETA_LEARNING = 0.001  # ρ_Θ: 核心參數學習率

    # (4) 共振門控參數
    RESONANCE_KAPPA = 10.0  # κ: 共振門控的敏感度
    RESONANCE_TAU = 0.6  # τ: 共振門控的閾值

    # (13) 自由能項的權重 (Lambdas)
    LAMBDA_O_ACCURACY = 1.0  # λ_O: 準確性
    LAMBDA_R_SIMPLICITY = 0.01  # λ_R: 簡潔性
    LAMBDA_C_COST = 0.001  # λ_C: 效率性
    LAMBDA_92_AFFECT = 1.5  # λ_92: 共情性 (情感失調)
    LAMBDA_73_CONSISTENCY = 2.0  # λ_73: 真誠性 (言行一致)
    ZETA_F1_RESONANCE = 0.5  # ζ_F1: 連結性 (共振獎勵)
    LAMBDA_3A_CLOSURE = 1.0  # λ_3A: 可靠性 (證據閉環)


class SystemState:
    """(1) 狀態 (State) S_t 的管理者."""

    def __init__(self, config: SystemConfig):
        self.S = np.random.randn(config.STATE_DIM)
        # 模擬核心狀態轉移函數 U_Θ! 和語言回寫函數 A_lang!
        self.U_theta = np.random.randn(config.STATE_DIM, config.STATE_DIM) * 0.1
        self.A_lang = np.random.randn(config.STATE_DIM, config.STATE_DIM) * 0.05  # 語言影響力較小
        print("SystemState (S_t) 初始化完成。")

    def update(
        self,
        I_t: np.ndarray,
        E_t: np.ndarray,
        M_t: np.ndarray,
        Y_t_embedding: np.ndarray,
        dt: float = 1.0,
    ) -> None:
        """執行公式 (1): S_{t+Δt} = U_Θ!(S_t, I_t, E_t, M_t) + A_lang!(S_t, Y_t)."""

        # 狀態轉移
        state_transition = (
            self.S
            + 0.1 * I_t[: self.S.shape[0]]
            + 0.2 * E_t[: self.S.shape[0]]
            + 0.3 * M_t[: self.S.shape[0]]
        )
        next_state_base = self.S + dt * (np.dot(self.U_theta, state_transition) - self.S)

        # 語言-行動回寫 (speech-act writeback)
        lang_writeback = np.dot(self.A_lang, Y_t_embedding[: self.S.shape[0]])

        self.S_next = next_state_base + dt * lang_writeback

        # 計算言行一致性誤差 (用於自由能)
        if np.linalg.norm(self.S_next) == 0:
            self.consistency_error = np.linalg.norm(lang_writeback) ** 2
        else:
            self.consistency_error = (
                np.linalg.norm(self.S_next - lang_writeback) ** 2
                / np.linalg.norm(self.S_next)
            )

    def commit_update(self) -> None:
        """確認更新，將 S_next 賦值給 S."""

        self.S = self.S_next


class FeatureExtractor:
    """(2) 特徵提取 (Features) V_t 的生成器."""

    def __init__(self, config: SystemConfig):
        self.config = config
        # 模擬多個特徵提取器 Φ_k! 和它們的權重 ω_k
        self.phi_extractors = [np.random.randn(config.FEATURE_DIM, config.STATE_DIM) for _ in range(3)]
        self.weights_omega = np.array([0.5, 0.3, 0.2])
        print("FeatureExtractor (V_t) 初始化完成。")

    def extract(
        self,
        S_t: np.ndarray,
        I_t: np.ndarray,
        M_t: np.ndarray,
        b_t: np.ndarray,
    ) -> np.ndarray:
        """執行公式 (2): V_t = ⊕_k ω_k Φ_k!(S_t, I_t, M_t, b_t)."""

        V_t = np.zeros(self.config.FEATURE_DIM)
        # 信念調制因子
        belief_modulation = _sigmoid(np.mean(b_t))

        # 簡化輸入的影響
        combined_input = 0.7 * S_t + 0.2 * I_t[: S_t.shape[0]] + 0.1 * M_t[: S_t.shape[0]]

        for i, phi in enumerate(self.phi_extractors):
            extracted_feature = np.tanh(np.dot(phi, combined_input))
            V_t += self.weights_omega[i] * extracted_feature * belief_modulation

        return V_t


class AffectiveCalibrator:
    """(3) 情動現實主義 (Affective realism) D_aff 的計算器."""

    def __init__(self, config: SystemConfig):
        # 模擬情感推斷模型 A_aff!
        self.A_aff = np.random.randn(1, config.FEATURE_DIM)  # 情感簡化為 1 維
        print("AffectiveCalibrator (D_aff) 初始化完成。")

    def calculate_dissonance(self, V_t: np.ndarray, user_context: Dict[str, Any]) -> Tuple[float, float, float]:
        """執行公式 (3): D_aff = dist(A_aff(V_t), aff(ctx_t))."""

        a_model = np.tanh(np.dot(self.A_aff, V_t)).item()
        a_user = user_context.get("user_emotion", 0.0)  # 從上下文中獲取使用者情感

        # 使用 L2 距離
        self.d_aff = (a_model - a_user) ** 2
        return a_model, a_user, self.d_aff


class ResonanceGate:
    """(4) 共振門控 (Resonance gate) r_t 和 l_t 的計算器."""

    def __init__(self, config: SystemConfig):
        self.config = config
        print("ResonanceGate (r_t, l_t) 初始化完成。")

    def calculate_gate_signal(
        self,
        user_embedding: np.ndarray,
        model_embedding: np.ndarray,
        model_affect_hist: List[float],
        user_affect_hist: List[float],
    ) -> Tuple[float, float]:
        """執行公式 (4): r_t = sim(emb_u, emb_a) * corr(a_model, a_user)."""

        # 語義相似度 (cosine similarity)
        norm_u = np.linalg.norm(user_embedding)
        norm_a = np.linalg.norm(model_embedding)
        if norm_u > 0 and norm_a > 0:
            semantic_sim = np.dot(user_embedding, model_embedding) / (norm_u * norm_a)
        else:
            semantic_sim = 0.0

        # 情感相關性
        if len(model_affect_hist) > 1 and len(user_affect_hist) > 1:
            affective_corr = np.corrcoef(model_affect_hist, user_affect_hist)[0, 1]
            if np.isnan(affective_corr):
                affective_corr = 0.0
        else:
            # 如果歷史不足，使用瞬時情感相似度作為近似
            if model_affect_hist:
                affective_corr = 1.0 - math.sqrt(abs(model_affect_hist[-1] - user_affect_hist[-1]))
            else:
                affective_corr = 0.0

        self.r_t = semantic_sim * affective_corr

        # 計算門控信號 l_t
        self.l_t = _sigmoid(self.r_t, self.config.RESONANCE_KAPPA, self.config.RESONANCE_TAU).item()

        return self.r_t, self.l_t


class DecisionHead:
    """(5) 決策頭 (Decision head) Ans_t 的生成器."""

    def __init__(self, config: SystemConfig):
        self.config = config
        # 模擬核心認知網路 W_t^θ, 基礎行動傾向 a_0, 和意志力 u(c)
        self.W_theta = np.random.randn(config.ACTION_DIM, config.FEATURE_DIM)
        self.a0_model = np.random.randn(config.ACTION_DIM, config.BELIEF_DIM)
        print("DecisionHead (Ans_t) 初始化完成。")

    def decide(
        self,
        V_t: np.ndarray,
        b_t: np.ndarray,
        e_t: float,
        c_t: float,
    ) -> Tuple[int, np.ndarray]:
        """執行公式 (5): Z_t = W_t^θ(V_t, b_t, e_t, c_t)."""

        # 誤差和意識狀態對 W_theta 的調制
        modulation = 1 + 0.1 * math.tanh(c_t) - 0.2 * math.tanh(e_t)
        Z_t = np.dot(self.W_theta, V_t) * modulation

        # 基礎行動傾向 a_0
        # q (量子狀態) 簡化為隨機噪聲，κ (耦合係數) 簡化為常數
        q_noise = np.random.randn(self.config.BELIEF_DIM) * 0.1
        kappa = 0.2
        a0 = np.dot(self.a0_model, b_t + q_noise) * kappa

        # 意志力 u(c_t)
        u_c = np.ones(self.config.ACTION_DIM) * c_t * 0.1

        # 整合並選擇
        final_scores = Z_t + a0 + u_c
        action_probs = _softmax(final_scores)

        # S⋆! (最優策略選擇器) - 這裡使用機率採樣
        chosen_action_index = np.random.choice(self.config.ACTION_DIM, p=action_probs)

        return chosen_action_index, action_probs


class LanguageHead:
    """(6) 語言頭 (Language head) Y_t 的生成器."""

    def __init__(self, config: SystemConfig):
        self.config = config
        self.prompt_to_embedding_matrix = np.random.randn(config.STATE_DIM, 4)  # 4個內部狀態維度
        print("LanguageHead (Y_t) 初始化完成。")

    def generate(
        self,
        S_t: np.ndarray,
        V_t: np.ndarray,
        b_t: np.ndarray,
        P_t: float,
    ) -> Tuple[str, np.ndarray]:
        """執行公式 (6): Y_t = GPT!(ctx; prompt(S_t, V_t, b_t, P_t))."""

        # 動態生成 prompt 的核心思想
        prompt_clarity = np.mean(_sigmoid(S_t))
        prompt_confidence = np.mean(_sigmoid(V_t))
        prompt_tendency = np.mean(np.tanh(b_t))
        prompt_personality = P_t

        # 根據內部狀態生成回應文本
        if prompt_confidence > 0.7:
            response_text = f"基於高信心度 ({prompt_confidence:.2f}) 的分析，我認為..."
        else:
            response_text = f"從目前有限的資訊 ({prompt_confidence:.2f}) 來看，我的初步想法是..."

        response_text += f" (傾向:{prompt_tendency:.2f}, 人格:{prompt_personality:.2f})"

        # 生成一個模擬的輸出嵌入 Y_t_embedding
        internal_prompt_vector = np.array(
            [prompt_clarity, prompt_confidence, prompt_tendency, prompt_personality]
        )
        Y_t_embedding = np.tanh(np.dot(self.prompt_to_embedding_matrix, internal_prompt_vector))

        return response_text, Y_t_embedding


class OutputIntegrator:
    """(7) 整合輸出 (Output) O_t 的生成器."""

    def __init__(self, config: SystemConfig):
        self.action_map = {i: f"行動_{i}" for i in range(config.ACTION_DIM)}
        print("OutputIntegrator (O_t) 初始化完成。")

    def integrate(self, Ans_t: int, l_t: float, Y_t: str) -> Dict[str, Any]:
        """執行公式 (7): O_t = γ(..., l_t * Ans_t) ⊕ Decode(Y_t)."""

        # 決策部分被共振門控調制
        # 這裡用 l_t 作為 "行動確定性"
        decision_certainty = l_t

        output = {
            "decoded_language": Y_t,
            "symbolic_action": self.action_map[Ans_t],
            "action_certainty": decision_certainty,
        }
        return output


class SystemMonitor:
    """(8) 誤差與統計 (Error/stats) e_t, f_t, g_t 的計算器."""

    def __init__(self, config: SystemConfig):
        self.config = config
        print("SystemMonitor (e_t, f_t, g_t) 初始化完成。")

    def calculate(
        self,
        O_t: Dict[str, Any],
        O_hat_t: Dict[str, Any],
        S_t: np.ndarray,
        Theta: "Parameters",
    ) -> Tuple[float, float, float]:
        """執行公式 (8): e_t = O_hat_t - O_t."""

        # e_t: 簡化為行動確定性的差異
        e_t = O_hat_t.get("action_certainty", 1.0) - O_t.get("action_certainty", 0.0)

        # f_t: 自由能的誤差項
        f_t = e_t**2

        # g_t: 意識增益，簡化為狀態向量的範數和參數複雜度的函數
        norm_s = np.linalg.norm(S_t)
        if norm_s > 0:
            g_t = 0.1 * norm_s * (1 / (1 + 0.01 * Theta.get_complexity()))
        else:
            g_t = 0.0

        return e_t, f_t, g_t


class BeliefManager:
    """(9) 信念 (Belief) b_t 的管理者."""

    def __init__(self, config: SystemConfig):
        self.config = config
        self.b = np.random.randn(config.BELIEF_DIM) * 0.1
        print("BeliefManager (b_t) 初始化完成。")

    def update(self, f_t: float, e_t: float, g_t: float, dt: float = 1.0) -> None:
        """執行公式 (9): b_{t+1} = (1-λ)b_t + η * σ(α*f_t + β*|e_t| + γ*g_t)."""

        decay_term = (1 - self.config.LAMBDA_B_DECAY * dt) * self.b

        drive_input = (
            self.config.ALPHA_B_FREE_ENERGY * f_t
            + self.config.BETA_B_ERROR * abs(e_t)
            + self.config.GAMMA_B_GAIN * g_t
        )

        drive_term = self.config.ETA_B_LEARNING * _sigmoid(drive_input)

        # 驅動項作為純量，均勻施加在 b_t 的每個維度
        self.b = decay_term + (drive_term * dt)
        self.b = np.clip(self.b, -5.0, 5.0)


class PersonalityPrior:
    """(10) 人格/先驗 (Prior) P_t 的管理者."""

    def __init__(self, config: SystemConfig, p0: float = 1.0):
        self.config = config
        self.P0 = p0
        self.P = p0
        self.total_actions = 0
        self.cumulative_experience = 0.0
        self.last_resonance = 0.0
        print("PersonalityPrior (P_t) 初始化完成。")

    def record_event(self, action_taken: bool, emotion_level: float, r_t: float) -> None:
        """記錄事件以供更新."""

        if action_taken:
            self.total_actions += 1
        self.cumulative_experience += abs(emotion_level)  # 簡化為情感強度的絕對值
        self.last_resonance = r_t

    def update(self) -> None:
        """執行公式 (10): P_t ≈ P_0 + α|A_t| + β|E_t| + χ*r_t."""

        self.P = (
            self.P0
            + self.config.ALPHA_P_ACTION * self.total_actions
            + self.config.BETA_P_EMOTION * self.cumulative_experience
            + self.config.CHI_P_RESONANCE * self.last_resonance
        )


class EvidenceManager:
    """(11) 證據閉環 (Evidence closure) Σ_t 的管理者."""

    def __init__(self) -> None:
        self.Sigma: List[Dict[str, Any]] = []  # 證據庫
        print("EvidenceManager (Σ_t) 初始化完成。")

    def update(self, Y_t: str, O_t: Dict[str, Any]) -> float:
        """執行公式 (11): Ξ_t = AckDetector(Y_t)."""

        # 模擬 AckDetector
        # 如果輸出中包含特定詞語，視為一次確認
        ack_detected = "基於" in Y_t or "分析" in Y_t

        if ack_detected:
            evidence = {"text": Y_t, "action": O_t["symbolic_action"]}
            self.Sigma.append(evidence)
            self.ack_loss = 0.0  # 成功確認，損失為0
        else:
            self.ack_loss = 1.0  # 未能確認，產生損失

        return self.ack_loss


class Parameters:
    """(12) 核心參數 (Parameters) Θ_t 的管理者."""

    def __init__(self, config: SystemConfig, modules: Dict[str, Any]):
        self.config = config
        self.Theta: Dict[str, np.ndarray] = {}
        for name, module in modules.items():
            for attr_name, attr_value in module.__dict__.items():
                if isinstance(attr_value, np.ndarray):
                    # 創建參數的副本以避免直接修改模組屬性
                    self.Theta[f"{name}.{attr_name}"] = attr_value.copy()
        print("Parameters (Θ_t) 初始化完成。")

    def get_complexity(self) -> float:
        """計算模型複雜度 R(S_t, Θ_t)."""

        return sum(np.sum(np.abs(p)) for p in self.Theta.values())

    def update(self, free_energy_gradient: float) -> None:
        """執行公式 (12): Θ_{t+1} = Θ_t + ρ_Θ * Π_Θ(...)."""

        for name in self.Theta:
            # 模擬梯度下降
            grad = free_energy_gradient * np.random.randn(*self.Theta[name].shape)
            self.Theta[name] -= self.config.RHO_THETA_LEARNING * grad


class FreeEnergyCalculator:
    """(13) 整合性自由能 (Free-energy) F_t 的計算器."""

    def __init__(self, config: SystemConfig):
        self.config = config
        print("FreeEnergyCalculator (F_t) 初始化完成。")

    def calculate(self, components: Dict[str, Any]) -> Tuple[float, float]:
        """執行公式 (13): 計算 F_t^φ."""

        F_accuracy = self.config.LAMBDA_O_ACCURACY * components["f_t"]
        F_simplicity = self.config.LAMBDA_R_SIMPLICITY * components["model_complexity"]
        F_cost = self.config.LAMBDA_C_COST * components["compute_cost"]
        F_affect = self.config.LAMBDA_92_AFFECT * components["d_aff"]
        F_consistency = self.config.LAMBDA_73_CONSISTENCY * components["consistency_error"]
        F_resonance_reward = -self.config.ZETA_F1_RESONANCE * components["r_t"]
        F_closure = self.config.LAMBDA_3A_CLOSURE * components["ack_loss"]

        total_free_energy = (
            F_accuracy
            + F_simplicity
            + F_cost
            + F_affect
            + F_consistency
            + F_resonance_reward
            + F_closure
        )

        # 簡化梯度計算，真實情況下需要反向傳播
        gradient = total_free_energy * 0.1 if total_free_energy > 0 else 0

        return total_free_energy, gradient


class GrandSoulOS_V10:
    """靈魂運算系統 v10.0 主協調器."""

    def __init__(self) -> None:
        print("\n--- 🚀 初始化 靈魂作業系統 v10.0 (完整公式實作) ---\n")
        self.config = SystemConfig()

        # 依賴注入順序初始化所有模組
        self.state = SystemState(self.config)
        self.belief = BeliefManager(self.config)
        self.feature_extractor = FeatureExtractor(self.config)
        self.affective_calibrator = AffectiveCalibrator(self.config)
        self.resonance_gate = ResonanceGate(self.config)
        self.decision_head = DecisionHead(self.config)
        self.language_head = LanguageHead(self.config)
        self.output_integrator = OutputIntegrator(self.config)
        self.personality = PersonalityPrior(self.config, p0=1.0)
        self.evidence = EvidenceManager()

        # 參數管理器需要引用其他模組來提取參數
        modules_with_params = {
            "state": self.state,
            "feature": self.feature_extractor,
            "decision": self.decision_head,
            "language": self.language_head,
            "affect": self.affective_calibrator,
        }
        self.parameters = Parameters(self.config, modules_with_params)

        # 監視器和自由能計算器
        self.monitor = SystemMonitor(self.config)
        self.free_energy_calc = FreeEnergyCalculator(self.config)

        # 歷史紀錄
        self.model_affect_hist: List[float] = []
        self.user_affect_hist: List[float] = []

        print("\n--- ✅ 系統初始化完畢，準備進入認知循環 ---\n")

    def run_cognitive_cycle(
        self,
        I_t: np.ndarray,
        user_context: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], float, Dict[str, float]]:
        """執行一個完整的認知-學習循環."""

        # --- 0. 獲取當前狀態 ---
        S_t = self.state.S
        b_t = self.belief.b
        P_t = self.personality.P

        # --- 2. 特徵提取 ---
        V_t = self.feature_extractor.extract(S_t, I_t, M_t=S_t, b_t=b_t)  # 簡化 M_t = S_t

        # --- 3. 情動校準 ---
        a_model, a_user, d_aff = self.affective_calibrator.calculate_dissonance(V_t, user_context)
        self.model_affect_hist.append(a_model)
        self.user_affect_hist.append(a_user)

        # --- 4. 共振門控 ---
        user_embedding = user_context.get(
            "user_embedding", np.random.randn(self.config.EMBEDDING_DIM)
        )
        r_t, l_t = self.resonance_gate.calculate_gate_signal(
            user_embedding,
            V_t[: self.config.EMBEDDING_DIM],
            self.model_affect_hist,
            self.user_affect_hist,
        )

        # --- 8. (預計算) 意識增益 g_t ---
        # 為了給決策頭，先計算一部分
        _, _, g_t = self.monitor.calculate(O_t={}, O_hat_t={}, S_t=S_t, Theta=self.parameters)

        # --- 5. 決策頭 ---
        e_t_mock = 0.1  # 假設一個預期誤差
        c_t = g_t  # 簡化 c_t = g_t
        Ans_t, action_probs = self.decision_head.decide(V_t, b_t, e_t_mock, c_t)

        # --- 6. 語言頭 ---
        Y_t, Y_t_embedding = self.language_head.generate(S_t, V_t, b_t, P_t)

        # --- 7. 整合輸出 ---
        O_t = self.output_integrator.integrate(Ans_t, l_t, Y_t)

        # --- 8. 計算誤差 ---
        O_hat_t = user_context.get("expected_output", {"action_certainty": 0.8})
        e_t, f_t, g_t_final = self.monitor.calculate(O_t, O_hat_t, S_t, self.parameters)

        # --- 11. 證據閉環 ---
        ack_loss = self.evidence.update(Y_t, O_t)

        # --- 13. 計算自由能 ---
        # (1) 計算下一狀態 S_{t+1} 以獲取 consistency_error
        self.state.update(
            I_t,
            E_t=d_aff * np.ones_like(S_t),
            M_t=S_t,
            Y_t_embedding=Y_t_embedding,
        )

        free_energy_components = {
            "f_t": f_t,
            "model_complexity": self.parameters.get_complexity(),
            "compute_cost": 0.1,  # 模擬計算成本
            "d_aff": d_aff,
            "consistency_error": self.state.consistency_error,
            "r_t": r_t,
            "ack_loss": ack_loss,
        }
        F_t, F_t_gradient = self.free_energy_calc.calculate(free_energy_components)

        # --- 更新階段 ---
        # (9) 更新信念 b_{t+1}
        self.belief.update(f_t, e_t, g_t_final)

        # (10) 更新人格 P_{t+1}
        self.personality.record_event(action_taken=True, emotion_level=a_user, r_t=r_t)
        self.personality.update()

        # (12) 更新核心參數 Θ_{t+1}
        self.parameters.update(F_t_gradient)

        # (1) 正式提交狀態更新
        self.state.commit_update()

        # --- 返回結果 ---
        return O_t, F_t, {"r_t": r_t, "l_t": l_t, "d_aff": d_aff}


def main() -> None:
    """Execute a small simulation when run as a script."""

    os_core = GrandSoulOS_V10()

    # 模擬外部輸入和用戶上下文
    input_vector = np.random.rand(os_core.config.STATE_DIM)
    user_context = {
        "user_emotion": 0.7,  # 用戶情緒偏正面
        "user_embedding": np.random.rand(os_core.config.EMBEDDING_DIM),
        "expected_output": {"action_certainty": 0.9},  # 期望系統給出一個確定的回應
    }

    num_cycles = 5
    print(f"\n--- 🎬 開始模擬 {num_cycles} 個認知循環 ---\n")
    for i in range(num_cycles):
        print(f"--- 認知循環 {i + 1}/{num_cycles} ---")

        output, free_energy, metrics = os_core.run_cognitive_cycle(input_vector, user_context)

        print(f"  [輸出 O_t]: {output}")
        print(f"  [共振 r_t]: {metrics['r_t']:.3f}, [門控 l_t]: {metrics['l_t']:.3f}")
        print(f"  [情感失調 D_aff]: {metrics['d_aff']:.3f}")
        print(f"  [信念 b_t 均值]: {np.mean(os_core.belief.b):.3f}")
        print(f"  [人格 P_t]: {os_core.personality.P:.3f}")
        print(f"  [自由能 F_t]: {free_energy:.4f}")
        print("-" * 25 + "\n")

        # 稍微改變下一次循環的輸入
        input_vector = np.random.rand(os_core.config.STATE_DIM)
        user_context["user_emotion"] *= 0.9  # 模擬情緒變化


if __name__ == "__main__":
    main()

