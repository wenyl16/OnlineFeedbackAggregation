import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io
# 引入 pyomo
import pyomo.environ as pyo
from pyomo.opt import SolverStatus, TerminationCondition

plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题


def gen_ess_boundaries(ess_data, current_soc, T, deltaT):
    p_chg = ess_data['P_chg']
    b = ess_data['B']
    eta_chg = ess_data['eta_chg']
    e0 = current_soc  # 使用当前SOC而不是时间序列

    # 为浮点数比较设置一个很小的容忍误差
    tolerance = 1e-6

    # 根据当前SOC动态调整最大充放电功率
    max_charge_power = p_chg
    max_discharge_power = p_chg

    # 如果电池几乎满电，则禁止继续充电
    if e0 >= b - tolerance:
        max_charge_power = 0

    # 如果电池几乎没电，则禁止继续放电
    if e0 <= tolerance:
        max_discharge_power = 0

    pmin_results = np.zeros(T)
    pmax_results = np.zeros(T)
    emin_results = np.zeros(T)
    emax_results = np.zeros(T)

    for k in range(T):
        pmin_results[k] = -max_charge_power
        pmax_results[k] = max_discharge_power

        # 能量边界计算修正（统一为电网端能量基准，与EV保持一致）
        c1_emin = -p_chg * deltaT * (k + 1)
        # c2_emin = -e0                          # 旧：电池端放电容量
        c2_emin = -(e0 * eta_chg)               # 新：电网端可放出能量 = 电池存量 * eta
        c3_emin = -p_chg * deltaT * (T - k)
        emin_results[k] = np.maximum(np.maximum(c1_emin, c2_emin), c3_emin)

        c1_emax = p_chg * deltaT * (k + 1)
        # c2_emax = b - e0                        # 旧：电池端剩余容量
        c2_emax = (b - e0) / eta_chg            # 新：电网端需充入能量 = 电池空间 / eta
        c3_emax = p_chg * deltaT * (T - k)
        emax_results[k] = np.minimum(np.minimum(c1_emax, c2_emax), c3_emax)

    return pmin_results, pmax_results, emin_results, emax_results


def gen_pv_boundaries(pv_data, start_time, T, deltaT):
    # pmin_series = pv_data['Pmin']
    pmin_series = 0.8*pv_data['Pmax']
    pmax_series = pv_data['Pmax']
    end_index = start_time + T
    if end_index > len(pmax_series):
        raise IndexError(f"时间窗口超出光伏 Pmax 数据范围。")
    pmin_window = pmin_series[start_time:end_index]
    pmax_window = pmax_series[start_time:end_index]
    lp = -pmax_window
    up = -pmin_window
    le = np.cumsum(lp * deltaT)
    ue = np.cumsum(up * deltaT)
    return lp, up, le, ue


def gen_ev_window_boundaries(ev_params, current_time, T, deltaT, current_E_real):
    """
    计算 EV 在当前滚动窗口内的绝对能量边界，并生成"匀速充电"的基线轨迹
    注意：输入参数新增了 current_E_real，代表该车辆当前实际已充入的电网端能量
    """
    ta = int(round(ev_params['ta'] / deltaT))
    td = int(round(ev_params['td'] / deltaT))
    SOCa, SOCd, SOCmax = ev_params['SOCa'], ev_params['SOCd'], ev_params['SOCmax']
    P_chg, eta_chg, B = ev_params['P_chg'], ev_params['eta_chg'], ev_params['B']

    # 统一转换到【电网端能量】
    req_grid_energy = max(0, (SOCd - SOCa) * B / eta_chg)
    max_grid_energy = max(0, (SOCmax - SOCa) * B / eta_chg)

    full_day_len = max(td + 1, current_time + T + 1)
    UP_full = np.zeros(full_day_len)
    LP_full = np.zeros(full_day_len)
    LE_full = np.zeros(full_day_len)
    UE_full = np.zeros(full_day_len)

    # 初始化当前窗口的基线功率
    p_base_window = np.zeros(T)

    if ta < full_day_len:
        valid_end = min(td, full_day_len)
        if valid_end > ta:
            # --- 功率边界 ---
            UP_full[ta:valid_end] = P_chg
            LP_full[ta:valid_end] = 0

            # --- 能量边界 ---
            for t in range(ta, valid_end):
                k = t - ta  # 已经在网的步数 (0-indexed)

                # UE: ASAP逻辑 (尽早充，上限看当前最多能吃多少电)
                UE_full[t] = min((k + 1) * P_chg * deltaT, max_grid_energy)

                # LE: ALAP逻辑 (最晚充，下限看为了达标当前必须攒多少电)
                rem_steps_bound = valid_end - 1 - t  # 离开前还剩几步
                rem_capacity = rem_steps_bound * P_chg * deltaT
                LE_full[t] = max(0, req_grid_energy - rem_capacity)

            # EV 基线：动态匀速充电策略 ---
            # 1. 计算剩余需要充入的电量
            rem_energy = max(0, req_grid_energy - current_E_real)

            # 2. 计算剩余可用充电步数
            # 如果车还没来 (current_time < ta)，则从 ta 开始算；如果车已经在网，则从当前时刻开始算
            start_calc_time = max(ta, current_time)
            rem_steps_base = valid_end - start_calc_time

            # 3. 如果还需要充电且还有时间，则分配匀速功率
            if rem_steps_base > 0 and rem_energy > 0:
                # 匀速功率 = (剩余能量 / 剩余步数) / 时间步长
                constant_power = (rem_energy / rem_steps_base) / deltaT

                # 安全限制：确保匀速计算出的功率不越过物理上限 P_chg
                constant_power = min(constant_power, P_chg)

                # 将恒定功率映射到当前求解窗口 [current_time, current_time + T)
                for t in range(start_calc_time, valid_end):
                    if current_time <= t < current_time + T:
                        idx = t - current_time
                        p_base_window[idx] = constant_power
        # --- 离网后的处理 ---
        if valid_end < full_day_len:
            LE_full[valid_end:] = req_grid_energy
            UE_full[valid_end:] = max_grid_energy

        # --- 修正：保证 LE <= UE，防止边界交叉导致求解器不可行 ---
        for t in range(ta, full_day_len):
            if LE_full[t] > UE_full[t]:
                LE_full[t] = UE_full[t]

    # 切片并转换为以 current_time 为起点的窗口内增量，与 ESS/TCL 语义一致
    le_window = LE_full[current_time: current_time + T] - current_E_real
    ue_window = UE_full[current_time: current_time + T] - current_E_real
    le_window = np.maximum(le_window, 0)
    ue_window = np.maximum(ue_window, 0)

    return (LP_full[current_time: current_time + T],
            UP_full[current_time: current_time + T],
            le_window,
            ue_window,
            p_base_window)


def gen_tcl_boundaries(tcl_params, current_temp, current_time, T, deltaT, ambient_temp_series):
    """
    计算 TCL (热泵/供暖模式) 在当前窗口内的功率和能量边界
    """
    # 1. 解析参数
    R = tcl_params['R']  # 热阻 (degC/kW)
    C = tcl_params['C']  # 热容 (kWh/degC)
    P_rated = tcl_params['P_rated']  # 额定功率 (kW)
    eta = tcl_params.get('eta', 2.5)  # 能效比 (COP)
    T_set = tcl_params['T_set']  # 设定温度 (degC)
    T_deadband = tcl_params['delta']  # 温控死区 (degC)

    # 温度舒适区间 [T_min, T_max]
    # 供暖模式
    T_max = T_set + T_deadband
    T_min = T_set - T_deadband

    # 2. 计算物理系数
    # 论文公式 (4.23): theta_t = alpha * theta_{t-1} + (1-alpha)*(theta_amb + eta*R*P)
    alpha = np.exp(-deltaT / (R * C))
    factor_amb = 1 - alpha
    factor_p = factor_amb * eta * R  # 功率项系数

    # 3. 获取当前时间窗口的环境温度
    end_index = current_time + T
    if end_index > len(ambient_temp_series):
        pad_len = end_index - len(ambient_temp_series)
        temp_window = np.concatenate([ambient_temp_series[current_time:],
                                      np.full(pad_len, ambient_temp_series[-1])])
    else:
        temp_window = ambient_temp_series[current_time:end_index]

    # 4. 初始化边界数组
    lp_window = np.zeros(T)
    up_window = np.full(T, P_rated)
    le_window = np.zeros(T)
    ue_window = np.zeros(T)
    p_base_window = np.zeros(T)

    # --- 计算 LE (最小能量边界) ---
    # 策略：尽可能少加热，允许温度掉到 T_min
    T_curr = current_temp
    e_cum = 0.0

    for k in range(T):
        Tamb = temp_window[k]
        # 自然演变 (假设 P=0)
        T_natural = alpha * T_curr + factor_amb * Tamb

        # 如果自然掉温低于 T_min，必须加热
        if T_natural < T_min:
            # 反解需要的功率: T_min = T_natural + factor_p * P
            p_req = (T_min - T_natural) / factor_p
            p_req = np.clip(p_req, 0, P_rated)
        else:
            p_req = 0.0

        e_cum += p_req * deltaT
        le_window[k] = e_cum
        # 更新状态
        T_curr = alpha * T_curr + factor_amb * Tamb + factor_p * p_req

    # --- 计算 UE (最大能量边界) ---
    # 策略：尽可能多加热，允许温度升到 T_max
    T_curr = current_temp
    e_cum = 0.0

    for k in range(T):
        Tamb = temp_window[k]
        # 满功率加热后的温度
        T_heated = alpha * T_curr + factor_amb * Tamb + factor_p * P_rated

        # 如果温度超过 T_max，限制功率
        if T_heated > T_max:
            # 反解最大允许功率: T_max = T_natural + factor_p * P
            T_natural = alpha * T_curr + factor_amb * Tamb
            p_max_allow = (T_max - T_natural) / factor_p
            p_use = np.clip(p_max_allow, 0, P_rated)
        else:
            p_use = P_rated

        e_cum += p_use * deltaT
        ue_window[k] = e_cum
        # 更新状态
        T_curr = alpha * T_curr + factor_amb * Tamb + factor_p * p_use

    T_curr_base = current_temp
    for k in range(T):
        Tamb = temp_window[k]
        # 预测自然演变 (假设 P=0)
        T_natural = alpha * T_curr_base + factor_amb * Tamb

        # 对齐 MATLAB: if T_in_baseline(t+1) < TCL.T_set
        if T_natural < T_set:
            # 反解需要的功率: T_set = T_natural + factor_p * P
            p_req = (T_set - T_natural) / factor_p
            p_use = np.clip(p_req, 0, P_rated)  # 限制在额定功率内
        else:
            p_use = 0.0

        p_base_window[k] = p_use
        # 更新基线温度状态
        T_curr_base = alpha * T_curr_base + factor_amb * Tamb + factor_p * p_use

    return lp_window, up_window, le_window, ue_window , p_base_window


def calculate_total_agg_boundaries(ess_df, pv_df, ev_list,tcl_list,current_time, deltaT,
                                   T_horizon, alpha, beta, ess_soc_list,tcl_current_temps, ambient_temp_data,ev_current_energies):

    T = int(T_horizon / deltaT)
    P_base_true = np.zeros(T)

    # 初始化累计数组
    P_bar_raw = np.zeros(T)
    P_under_raw = np.zeros(T)
    E_bar_raw = np.zeros(T)
    E_under_raw = np.zeros(T)

    # 聚合 ESS
    for i, (_, row) in enumerate(ess_df.iterrows()):
        lp, up, le, ue = gen_ess_boundaries(row.to_dict(), ess_soc_list[i], T, deltaT)
        P_under_raw += lp;
        P_bar_raw += up;
        E_under_raw += le;
        E_bar_raw += ue
        P_base_true += np.zeros(T)
    # 聚合 PV
    for _, row in pv_df.iterrows():
        lp, up, le, ue = gen_pv_boundaries(row.to_dict(), current_time, T, deltaT)
        P_under_raw += lp;
        P_bar_raw += up;
        E_under_raw += le;
        E_bar_raw += ue
        P_base_true += (lp + up) / 2.0
     # 聚合 EV
    for i, ev_params in enumerate(ev_list):
        current_E_real = ev_current_energies[i]
        # 注意这里接收了第 5 个返回值 p_base_ev
        lp, up, le, ue, p_base_ev = gen_ev_window_boundaries(ev_params, current_time, T, deltaT, current_E_real)

        P_under_raw += lp
        P_bar_raw += up
        E_under_raw += le   # le/ue 已在 gen_ev_window_boundaries 内转为窗口增量
        E_bar_raw += ue
        P_base_true += p_base_ev

    #聚合tcl
    for i, tcl_params in enumerate(tcl_list):
        current_temp = tcl_current_temps[i]
        lp, up, le, ue, p_base_tcl = gen_tcl_boundaries(tcl_params, current_temp, current_time, T, deltaT,
                                                        ambient_temp_data)
        P_under_raw += lp
        P_bar_raw += up
        # 叠加能量增量 (le/ue 已经是窗口内的累计量)
        E_under_raw += le
        E_bar_raw += ue
        P_base_true += p_base_tcl

    P_base = P_base_true

    # 构建功率约束
    # A * DeltaP <= alpha * b

    # --- 构建矩阵 A ---
    I_matrix = np.eye(T)  # T x T 单位矩阵
    A_matrix = np.vstack([I_matrix,  # 上半部分 I
                          -I_matrix])  # 下半部分 -I

    # --- 构建向量 b ---
    # 笔记: b = [ \bar{P} - P_base ]
    #          [ -\underline{P} + P_base ]
    b_upper = P_bar_raw - P_base  # 上半部分
    b_lower = -P_under_raw + P_base  # 下半部分

    b_vector = np.concatenate([b_upper,  # 拼接到一起
                               b_lower])

    # 2. 构建能量约束
    #  C * DeltaP <= beta * d
    # --- 构建下三角矩阵 L ---
    L_matrix = np.tril(np.ones((T, T))) * deltaT

    # --- 构建矩阵 C ---
    # 笔记: C = [ L ]
    #          [ -L ]
    C_matrix = np.vstack([L_matrix,  # 上半部分 L
                          -L_matrix])  # 下半部分 -L

    # --- 计算能量基线 L * P_base ---
    L_times_Pbase = np.dot(L_matrix, P_base)

    # --- 构建向量 d ---
    # 笔记: d = [ \bar{E} - L*P_base ]
    #          [ -\underline{E} + L*P_base ]
    d_upper = E_bar_raw - L_times_Pbase  # 上半部分
    d_lower = -E_under_raw + L_times_Pbase  # 下半部分

    d_vector = np.concatenate([d_upper,  # 拼接到一起
                               d_lower])

    # 最终功率上界 = 基线 + alpha * (b的上半部分)
    UP_final = P_base + alpha * b_upper

    # 最终功率下界 = 基线 - alpha * (b的下半部分)
    LP_final = P_base - alpha * b_lower

    # 最终能量上界
    UE_final = L_times_Pbase + beta * d_upper

    # 最终能量下界
    LE_final = L_times_Pbase - beta * d_lower

    return LP_final, UP_final, LE_final, UE_final, b_upper, b_lower, d_upper, d_lower,P_base

def sample_curves_from_boundaries(lp_agg, up_agg, le_agg, ue_agg, deltaT, alpha, beta, P_cmd_t):
    """
    针对大规模系统强化的轨迹采样函数
    """
    T = len(lp_agg)
    v = np.random.randn(T)
    model = pyo.ConcreteModel()
    model.T = pyo.RangeSet(0, T - 1)
    model.p = pyo.Var(model.T)
    # 1. 目标函数
    model.obj = pyo.Objective(rule=lambda m: sum(v[t] * m.p[t] for t in m.T), sense=pyo.minimize)

    model.power_limits = pyo.ConstraintList()
    model.energy_limits = pyo.ConstraintList()

    # 2. 对首步指令进行严格的物理可行性检查
    if P_cmd_t is not None:
        p_energy_min = le_agg[0] / deltaT
        p_energy_max = ue_agg[0] / deltaT

        # 放宽一点点容差，防止因极端浮点误差导致 safe_lp0 > safe_up0
        safe_lp0 = max(lp_agg[0], p_energy_min) - 1e-4
        safe_up0 = min(up_agg[0], p_energy_max) + 1e-4

        # 确保下界不大于上界
        if safe_lp0 > safe_up0:
            safe_lp0, safe_up0 = safe_up0, safe_lp0

        P_cmd_t_safe = max(safe_lp0, min(safe_up0, P_cmd_t))
        model.cmd_constraint = pyo.Constraint(expr=model.p[0] == P_cmd_t_safe)
    # 3. 建立常规约束
    tolerance = 1e-3
    for t in model.T:
        model.power_limits.add(model.p[t] >= lp_agg[t] - tolerance)
        model.power_limits.add(model.p[t] <= up_agg[t] + tolerance)

        energy_sum = sum(model.p[tau] * deltaT for tau in range(t + 1))
        model.energy_limits.add(energy_sum >= le_agg[t] - tolerance)
        model.energy_limits.add(energy_sum <= ue_agg[t] + tolerance)

    # 4. 求解与诊断
    solver = pyo.SolverFactory('gurobi')
    solver.options['DualReductions'] = 0
    # 为大系统采样设置时间限制，防止卡死
    solver.options['TimeLimit'] = 5

    results = solver.solve(model, tee=False)

    # 5. 严格的状态判定：非 Optimal 直接抛出异常
    if (results.solver.status == pyo.SolverStatus.ok) and \
            (results.solver.termination_condition == pyo.TerminationCondition.optimal):
        return np.array([pyo.value(model.p[t]) for t in model.T])
    else:
        # =================================================================
        # 失败直接报错，打印当前的 alpha 和 beta 以及求解器状态，方便 Debug
        # =================================================================
        error_msg = (
            f"\n[致命错误] 采样优化失败 (Infeasible)!\n"
            f"求解器状态: {results.solver.termination_condition}\n"
            f"当前缩放系数: Alpha = {alpha:.4f}, Beta = {beta:.4f}\n"
            f"请检查传入的聚合边界是否有交叉，或尝试调整学习率。"
        )
        raise RuntimeError(error_msg)

def solve_decomposition_strictly_step2(p_target_total, p_base, ess_df, pv_df, ev_list, tcl_list,
                                       ev_current_energies, tcl_current_temps, ambient_temp_data,
                                       current_time, deltaT, T_horizon, ess_soc_list):
    """
    目标: min (DeltaP_0 - DeltaP)^2
    约束: 各设备的物理功率与能量边界约束
    """
    T = int(T_horizon / deltaT)
    n_ess = len(ess_df)
    n_pv = len(pv_df)
    n_ev = len(ev_list)
    n_tcl = len(tcl_list)
    n_total_resources = n_ess + n_pv + n_ev + n_tcl

    delta_p0 = p_target_total - p_base

    # --- 1. 构建模型 ---
    model = pyo.ConcreteModel()
    model.I = pyo.RangeSet(0, n_total_resources - 1)
    model.T = pyo.RangeSet(0, T - 1)

    # 决策变量：各设备功率
    model.p = pyo.Var(model.I, model.T)

    # 实际总偏差计算
    def delta_p_actual_rule(m, t):
        return sum(m.p[i, t] for i in m.I) - p_base[t]

    model.delta_p_actual = pyo.Expression(model.T, rule=delta_p_actual_rule)

    # 目标函数
    def obj_rule(m):
        return(sum((delta_p0[t] - m.delta_p_actual[t]) ** 2 for t in m.T))

    model.obj = pyo.Objective(rule=obj_rule, sense=pyo.minimize)

    # --- 2. 添加约束 ---
    model.power_limits = pyo.ConstraintList()
    model.energy_limits = pyo.ConstraintList()
    resource_idx = 0

    # ==================== 1. ESS 约束 ====================
    for i, (_, row) in enumerate(ess_df.iterrows()):
        current_soc = ess_soc_list[i]
        lp, up, le, ue = gen_ess_boundaries(row.to_dict(), current_soc, T, deltaT)
        for t in model.T:
            model.power_limits.add(model.p[resource_idx, t] >= lp[t])
            model.power_limits.add(model.p[resource_idx, t] <= up[t])
            energy_sum = sum(model.p[resource_idx, tau] * deltaT for tau in range(t + 1))
            model.energy_limits.add(energy_sum >= le[t])
            model.energy_limits.add(energy_sum <= ue[t])
        resource_idx += 1

    # ==================== 2. PV 约束 ====================
    for _, row in pv_df.iterrows():
        lp, up, le, ue = gen_pv_boundaries(row.to_dict(), current_time, T, deltaT)
        for t in model.T:
            model.power_limits.add(model.p[resource_idx, t] >= lp[t])
            model.power_limits.add(model.p[resource_idx, t] <= up[t])
            energy_sum = sum(model.p[resource_idx, tau] * deltaT for tau in range(t + 1))
            model.energy_limits.add(energy_sum >= le[t])
            model.energy_limits.add(energy_sum <= ue[t])
        resource_idx += 1

    # ==================== 4. EV 约束 ====================
    for i, ev_params in enumerate(ev_list):
        current_E_real = ev_current_energies[i]
        lp, up, le, ue, _ = gen_ev_window_boundaries(ev_params, current_time, T, deltaT, current_E_real)

        ta_step = int(round(ev_params['ta'] / deltaT))
        td_step = int(round(ev_params['td'] / deltaT))

        for t in model.T:
            model.power_limits.add(model.p[resource_idx, t] >= lp[t])
            model.power_limits.add(model.p[resource_idx, t] <= up[t])

            abs_t = current_time + t  # 当前窗口第 t 步对应的绝对时间步
            energy_gain = sum(model.p[resource_idx, tau] * deltaT for tau in range(t + 1))

            # le/ue 已是窗口内增量，直接与 energy_gain 比较
            if ta_step <= abs_t < td_step:
                model.energy_limits.add(energy_gain >= le[t])
                model.energy_limits.add(energy_gain <= ue[t])
            elif abs_t >= td_step:
                # 离网后只施加下界约束（已完成最低充电目标）
                model.energy_limits.add(energy_gain >= le[t])
        resource_idx += 1

    # ==================== 5. TCL 约束 ====================
    for i, tcl_params in enumerate(tcl_list):
        P_rated = tcl_params['P_rated']
        lp, up, le, ue, _ = gen_tcl_boundaries(tcl_params, tcl_current_temps[i], current_time, T, deltaT,
                                                ambient_temp_data)
        for t in model.T:
            model.power_limits.add(model.p[resource_idx, t] >= lp[t])
            model.power_limits.add(model.p[resource_idx, t] <= up[t])
            energy_sum = sum(model.p[resource_idx, tau] * deltaT for tau in range(t + 1))
            model.energy_limits.add(energy_sum >= le[t])
            model.energy_limits.add(energy_sum <= ue[t])
        resource_idx += 1

    # --- 3. 求解 ---
    solver = pyo.SolverFactory('gurobi')
    solver.options['DualReductions'] = 0  # 方便排查 Infeasible 问题
    results = solver.solve(model, tee=False)

    # --- 4. 结果提取 ---
    if (results.solver.status == SolverStatus.ok) and (
            results.solver.termination_condition == TerminationCondition.optimal):

        ess_powers = np.zeros((n_ess, T))
        pv_powers = np.zeros((n_pv, T))
        ev_powers = np.zeros((n_ev, T))
        tcl_powers = np.zeros((n_tcl, T))

        for i in range(n_ess):
            for t in model.T: ess_powers[i, t] = pyo.value(model.p[i, t])
        for i in range(n_pv):
            idx = n_ess + i
            for t in model.T: pv_powers[i, t] = pyo.value(model.p[idx, t])
        for i in range(n_ev):
            idx = n_ess + n_pv + i
            for t in model.T: ev_powers[i, t] = pyo.value(model.p[idx, t])
        for i in range(n_tcl):
            idx = n_ess + n_pv + n_ev + i
            for t in model.T: tcl_powers[i, t] = pyo.value(model.p[idx, t])

        # 重新汇总全部设备的功率
        p_actual_total = (np.sum(ess_powers, axis=0) + np.sum(pv_powers, axis=0) +
                          np.sum(ev_powers, axis=0) + np.sum(tcl_powers, axis=0))

        delta_p_star = p_actual_total - p_base

        # 误差计算：只考核物理偏差，排除了惩罚项的影响
        sqrt_error = np.sum((delta_p0 - delta_p_star)**2)

        return sqrt_error, delta_p_star, p_actual_total, ess_powers, pv_powers, ev_powers, tcl_powers

    else:
        print(f"Critical Warning: Decomposition Optimization failed at step {current_time}.")
        raise RuntimeError(
            f"Decomposition Optimization Infeasible at time {current_time}. Solver condition: {results.solver.termination_condition}")


def identify_active_constraints_step3_strict(delta_p_star, delta_p0, T_horizon, deltaT):
    """
    判断哪些约束离 Delta P0 比离 Delta P* 更近
    即寻找约束下标集 J 和 K，使:
       A_j * Delta P* < A_j * Delta P0
       C_k * Delta P* < C_k * Delta P0
    """
    T = int(T_horizon / deltaT)

    # 1. 重构矩阵 A
    #   A = [ I ]
    #      [ -I ]
    I_matrix = np.eye(T)
    A_matrix = np.vstack([I_matrix, -I_matrix])  # 形状 (2T, T)

    # 2. 重构矩阵 C
    #     C = [ L ]
    #         [ -L ]
    L_matrix = np.tril(np.ones((T, T))) * deltaT
    C_matrix = np.vstack([L_matrix, -L_matrix])  # 形状 (2T, T)

    # A * Delta P*
    A_times_dP_star = np.dot(A_matrix, delta_p_star)

    # A * Delta P0
    A_times_dP0 = np.dot(A_matrix, delta_p0)

    # C * Delta P*
    C_times_dP_star = np.dot(C_matrix, delta_p_star)

    # C * Delta P0
    C_times_dP0 = np.dot(C_matrix, delta_p0)

    # 4. 寻找下标集 J
    #    条件: A_j * Delta P* < A_j * Delta P0
    # True 的位置代表该下标属于集合 J
    J_mask = A_times_dP_star < (A_times_dP0 - 1e-6)

    # 5. 寻找下标集 K
    #  条件: C_k * Delta P* < C_k * Delta P0

    K_mask = C_times_dP_star < (C_times_dP0 - 1e-6)

    return J_mask, K_mask, A_matrix, C_matrix


def calculate_gradients_step4_strict(J_mask, K_mask,
                                     A_matrix, C_matrix,
                                     delta_p_star,
                                     b_upper, d_upper,
                                     alpha, beta,
                                     P_base, P_bar_raw, P_under_raw,
                                     E_base, E_bar_raw, E_under_raw):
    """
    严格按照论文公式 (11), (12), (13) 修改：
    加入分母 ||A_j||^2 和 ||C_l||^2，并在求导中引入系数 2。
    """
    # 重构向量 b 和 d
    b_vec_upper = P_bar_raw - P_base
    b_vec_lower = -P_under_raw + P_base
    b_vector_full = np.concatenate([b_vec_upper, b_vec_lower])  # 形状 (2T,)

    d_vec_upper = E_bar_raw - E_base
    d_vec_lower = -E_under_raw + E_base
    d_vector_full = np.concatenate([d_vec_upper, d_vec_lower])  # 形状 (2T,)

    # 1. 提取集合 J 和 K 对应的子矩阵/子向量
    A_J = A_matrix[J_mask]  # 形状 (N_active_J, T)
    b_J = b_vector_full[J_mask]  # 形状 (N_active_J,)

    C_K = C_matrix[K_mask]  # 形状 (N_active_K, T)
    d_K = d_vector_full[K_mask]  # 形状 (N_active_K,)

    # ================= 核心修改部分 =================

    # 计算法向量的模长平方 (对应公式中的 ||A_j||^2 和 ||C_k||^2)
    # 加上极小值 1e-8 防止出现除以 0 的数值异常
    norm_A_sq = np.sum(A_J ** 2, axis=1) + 1e-8
    norm_C_sq = np.sum(C_K ** 2, axis=1) + 1e-8

    # --- 对 alpha 的计算 (公式 11 & 12) ---
    # 计算括号内的残差
    term_alpha_residual = alpha * b_J - np.dot(A_J, delta_p_star)

    # grad_alpha = 2 * sum( (b_j * 残差_j) / ||A_j||^2 )
    grad_alpha = 2 * np.sum((b_J * term_alpha_residual) / norm_A_sq)

    # Alpha 项误差: sum( 残差_j^2 / ||A_j||^2 )
    loss_alpha = np.sum((term_alpha_residual ** 2) / norm_A_sq)

    # --- 对 beta 的计算  ---
    # 计算括号内的残差
    term_beta_residual = beta * d_K - np.dot(C_K, delta_p_star)

    # :grad_beta = 2 * sum( (d_k * 残差_k) / ||C_k||^2 )
    grad_beta = 2 * np.sum((d_K * term_beta_residual) / norm_C_sq)

    # Beta 项误差: sum( 残差_k^2 / ||C_k||^2 )
    loss_beta = np.sum((term_beta_residual ** 2) / norm_C_sq)

    # 论文中定义的严谨总距离误差 S
    total_loss = loss_alpha + loss_beta

    return grad_alpha, grad_beta, total_loss


def update_parameters_step5_strict(alpha, beta, grad_alpha, grad_beta, learning_rate1,learning_rate2):
    """
    按照论文公式 (14) 执行带非负投影的梯度下降
    """
    step_alpha = learning_rate1 * grad_alpha
    step_beta = learning_rate2 * grad_beta
    # 核心更新逻辑
    alpha_new = alpha - step_alpha
    beta_new = beta - step_beta

    # 论文公式 (14) 中的投影：max{0, 变量}
    # 在实际代码工程中，为了防止可行域完全坍缩导致 Pyomo 求解无解，
    # 建议保留一个极小的正数下限（如 0.001），这里我将其改为 0.0 以绝对贴合论文，
    # 如果运行报错死锁，可以回调到 0.01 或 0.001。
    alpha_new = max(0.0, alpha_new)
    beta_new = max(0.0, beta_new)

    return alpha_new, beta_new

def rolling_power_decomposition():
    # ================= 参数初始化 =================
    deltaT = 1/60
    T_horizon = 2
    T = int(T_horizon / deltaT)
    alpha = 1.0  # 功率边界缩放系数
    beta = 1.0  # 能量边界缩放系数
    learning_rate1 = 1.5e-9
    learning_rate2 = 2.5e-12

    np.random.seed(0)
    np.set_printoptions(precision=3, suppress=True)

    try:

        # ================= 读取并处理温度数据 =================
        profiles = np.load('profiles_data.npz')
        temp_raw = profiles['temp_data']  # 获取 5 分钟间隔的温度数据

        # 构建原始时间轴 (0, 5, 10, 15... 分钟)
        original_time_temp = np.arange(len(temp_raw)) * 5

        # 构建目标时间轴 (1 分钟间隔，全天 1440 分钟)
        target_time = np.arange(1440)

        # 线性插值：将 5 分钟数据映射到 1 分钟数据上
        ambient_temp_1440 = np.interp(target_time, original_time_temp, temp_raw)

        # 拼接延展：防止滚动优化看向第二天的未来(T=120步)时数组越界
        # 将当天的前 T+100 步拼接到末尾，模拟第二天的循环气温
        ambient_temp_data = np.concatenate([ambient_temp_1440, ambient_temp_1440[:T + 100]])
        # ============================================================


        # 插值：将 15分钟分辨率 (96点) 映射到 1分钟分辨率 (1440点)
        # ================= 2. 读取并处理光伏数据 =================
        pv_mat_data = scipy.io.loadmat('PV_samples.mat')

        if pv_mat_data['PV_samples'].shape[0] == 96:
            pv_raw_96 = pv_mat_data['PV_samples'][:, 0]
            original_time_pv = np.linspace(0, 1440, 96, endpoint=False)
            pv_raw_1440 = np.interp(target_time, original_time_pv, pv_raw_96)
            pv_power_template = np.concatenate([pv_raw_1440, pv_raw_1440[:T + 100]])

    except FileNotFoundError as e:
        print(f"错误: 文件未找到 - {e.filename}。请确保数据文件已上传。")
        return None, None, None, None, None, None


    total_simulation_periods = 1440

    # ================= 批量生成 100 个设备参数  =================

    # 1. 设定数量 (EV:TCL:PV:ESS = 10:8:1:1 => 50:40:5:5)
    n_ev_target = 50
    n_tcl_target = 40

    ess_df = pd.DataFrame({'P_chg': [100,150,100,150,100], 'B': [600,800,600,800,600], 'eta_chg': [0.95, 0.95,0.95,0.95,0.95]})
    #ess_df = pd.DataFrame({'P_chg': [], 'B': [], 'eta_chg': []})

    pv_pmax_scalers = [50,70,50,70,50]
    pv_df = pd.DataFrame({
         'Pmin': [np.zeros(total_simulation_periods) for _ in range(5)],
         'Pmax': [scaler * pv_power_template for scaler in pv_pmax_scalers]})
    #pv_df = pd.DataFrame({'Pmin': [], 'Pmax': []})

    ess_soc_list = [300.0, 300.0,300,300,300]
    #ess_soc_list = []

    # 2. 批量生成 EV (电动汽车)
    ev_list = []

    # 物理常量预设
    P_chg = 7.0
    eta_chg = 0.95
    B_options = [50, 60, 80]

    for _ in range(n_ev_target):
        dice = np.random.uniform(0, 1)
        B = np.random.choice(B_options)

        if dice <= 0.8:
            # 80% A类: 傍晚下班回家充电 (跨天)
            # 使用正态分布更符合真实人类行为规律
            ta = min(np.random.normal(19.0, 2.0), 22.0)  # 均值19点，最晚22点
            td = max(24.0 + 7.0, np.random.normal(24.0 + 8.0, 1.5))  # 均值次日8点，最早次日7点
            SOCa = min(0.9, max(np.random.normal(0.5, 0.2), 0.2))  # 初始SOC均值50%
            SOCd = 0.9 + 0.09 * np.random.binomial(1, 0.5)  # 目标SOC 90%或99%

        else:
            # 20% B类: 白天工作时间充电
            ta = min(10.0, max(7.0, np.random.normal(8.5, 1.5)))  # 均值8点半
            td = min(18.0, ta + np.random.uniform(7.0, 9.0))  # 停放7-9小时
            SOCa = min(0.9, max(np.random.normal(0.6, 0.2), 0.2))  # 初始SOC均值60%
            SOCd = 0.9 + 0.09 * np.random.binomial(1, 0.5)

        # 计算电网侧视角下，期望抽取的总能量
        req_grid_energy = (SOCd - SOCa) * B / eta_chg
        # 计算电网侧视角下，这段停放时间内最多能抽取的极限能量
        max_grid_energy = P_chg * (td - ta)

        if req_grid_energy > max_grid_energy:
            # 如果时间根本不够充满，强制下调用户的目标 SOC，保证后续求解器有解
            # 实际充入电池的能量 = max_grid_energy * eta_chg
            SOCd = SOCa + (max_grid_energy * eta_chg / B)
            # 退让 0.01 防止浮点数精度在边界卡死求解器
            SOCd = max(SOCa, SOCd - 0.01)

        ev_param = {
            'ta': ta,
            'td': td,
            'SOCa': SOCa,
            'SOCd': SOCd,
            'SOCmax': 1.0,
            'P_chg': P_chg,
            'eta_chg': eta_chg,
            'B': B
        }
        ev_list.append(ev_param)

    # 6. 批量生成 TCL (温控负荷) - 从真实的建筑热工物理表读取
    tcl_list = []
    tcl_current_temps_list = []

    try:
        # 读取真实的建筑热工参数与热泵数据表
        tcl_df_raw = pd.read_csv('ZH_buildings.csv')

        # 检查数据行数是否满足我们设定的 n_tcl_target (比如你设定的 40 台)
        actual_tcl_count = min(n_tcl_target, len(tcl_df_raw))
        if actual_tcl_count < n_tcl_target:
            print(
                f"警告：数据表中的建筑数量 ({len(tcl_df_raw)}) 少于设定的目标数量 ({n_tcl_target})，将只生成 {actual_tcl_count} 台 TCL。")

        # 遍历数据表提取物理参数
        for i in range(actual_tcl_count):
            hbld = tcl_df_raw.iloc[i]['HBLD']  # 建筑整体等效热损耗系数 (kW/K)
            cbld = tcl_df_raw.iloc[i]['CBLD']  # 建筑整体等效热容 (kWh/K)
            prt = tcl_df_raw.iloc[i]['PRT']  # 热泵额定制热功率 (kW)

            # 统一设定能效比 COP (一般空气源热泵在 2.5~3.0 左右)
            eta_cop = 3.85

            tcl_param = {
                'R': 1.0 / hbld,  # 物理转换: 热阻 = 1 / 热损耗系数
                'C': cbld,  # 直接取热容
                'P_rated': 2*prt / eta_cop,  # 物理转换: 额定电功率 = 热功率 / COP
                'eta': eta_cop,
                'T_set': 20.3,  # 统一设定舒适温度
                'delta': 5 # 设定温控死区
                }
            tcl_list.append(tcl_param)
            tcl_current_temps_list.append(tcl_param['T_set'])

    except FileNotFoundError as e:
        print(f"严重错误: 找不到 TCL 数据文件 - {e.filename}。请确认文件名和路径！")
        raise  # 找不到文件直接报错停机，防止后续使用空列表计算出错

    tcl_current_temps = np.array(tcl_current_temps_list)  # 转为 numpy 数组

    # 动态更新实际生成的 TCL 数量 (覆盖掉之前可能设定的target 值)
    n_tcl_target = len(tcl_list)
    n_tcl = len(tcl_list)


    n_ess = len(ess_soc_list)
    n_pv = len(pv_df)
    n_ev = len(ev_list)
    ev_current_energies = np.zeros(len(ev_list))
    # 历史数据记录列表
    pv_energy_list = [0.0] * n_pv

    all_ess_powers, all_pv_powers= [], []
    all_ev_powers = []  # 记录历史
    all_tcl_powers = []
    all_tcl_temps = []
    all_soc_history, all_pv_energy_history = [], []
    # 记录 alpha 和 beta 的变化历史
    all_alpha_history = []
    all_beta_history = []
    all_loss_history = []
    # 记录指令功率和实际执行功率
    all_p_cmd = []
    all_p_actual = []
    print("开始实时滚动功率分解 (Strict Gradient Descent Mode)...")
    print(f"初始 Alpha: {alpha}, Beta: {beta}, LR1(功率): {learning_rate1}, LR2(能量): {learning_rate2}")

    current_time = 0

    while current_time + T <= total_simulation_periods:

        print(f"\n--- 时间步 {current_time} ---")


        #  计算带有当前 alpha/beta 的边界
        # 它返回缩放后的边界 (LP_agg, UP_agg...) 和 宽度向量 (b_vec, d_vec)
        (LP_agg, UP_agg, LE_agg, UE_agg,b_upper_vec, b_lower_vec, d_upper_vec, d_lower_vec, P_base) = calculate_total_agg_boundaries(
            ess_df, pv_df, ev_list, tcl_list, current_time, deltaT,
            T_horizon, alpha, beta, ess_soc_list, tcl_current_temps, ambient_temp_data,ev_current_energies
        )

        L_matrix = np.tril(np.ones((T, T))) * deltaT
        E_base = np.dot(L_matrix, P_base)

        # 记录历史
        all_alpha_history.append(alpha)
        all_beta_history.append(beta)

        # 采样 P0 (Target)
        p_cmd_current = np.random.uniform(LP_agg[0], UP_agg[0])
        p_target = sample_curves_from_boundaries(
            LP_agg, UP_agg, LE_agg, UE_agg, deltaT, alpha, beta, P_cmd_t=p_cmd_current)
        # 3. 求解分解问题
        (sqrt_error, delta_p_star, p_actual_agg,
         ess_powers_horizon, pv_powers_horizon, ev_powers_horizon,
         tcl_powers_horizon) = solve_decomposition_strictly_step2(
            p_target, P_base, ess_df, pv_df, ev_list, tcl_list,
            ev_current_energies, tcl_current_temps, ambient_temp_data,
            current_time, deltaT, T_horizon, ess_soc_list
        )

        # 记录指令功率和实际执行功率（仅记录当前时间步）
        all_p_cmd.append(p_target[0])
        all_p_actual.append(p_actual_agg[0])

        # 4. 寻找生效约束集合
        delta_p0 = p_target - P_base
        (J_mask, K_mask, A_matrix, C_matrix) = identify_active_constraints_step3_strict(
            delta_p_star, delta_p0, T_horizon, deltaT
        )

        # 5. 计算梯度
        # 【修改 5】: 构建合成后的 raw 边界，补齐 15 个参数，对齐严格版本的梯度公式
        P_bar_raw_synth = P_base + b_upper_vec
        P_under_raw_synth = P_base - b_lower_vec
        E_bar_raw_synth = E_base + d_upper_vec
        E_under_raw_synth = E_base - d_lower_vec

        (grad_alpha, grad_beta, current_loss) = calculate_gradients_step4_strict(
            J_mask, K_mask, A_matrix, C_matrix, delta_p_star,
            b_upper_vec, d_upper_vec, alpha, beta,
            P_base, P_bar_raw_synth, P_under_raw_synth,
            E_base, E_bar_raw_synth, E_under_raw_synth
        )

        all_loss_history.append(sqrt_error)
        # 6. 更新系数
        # 【修改 6】: 传入两个独立学习率 learning_rate1 和 learning_rate2
        if current_loss > 1e-4:
            alpha, beta = update_parameters_step5_strict(
                alpha, beta, grad_alpha, grad_beta, learning_rate1, learning_rate2
            )
            print(
                f"  [反馈] Loss={sqrt_error:.4f}, Grad_A={grad_alpha:.4f}, Grad_B={grad_beta:.4f} -> New Alpha={alpha:.3f}, Beta={beta:.3f}")

        # 资源状态更新
        first_step_ess_powers = ess_powers_horizon[:, 0]
        first_step_ev_powers = ev_powers_horizon[:, 0]
        ev_current_energies += first_step_ev_powers * deltaT

        if len(tcl_powers_horizon) > 0:
            first_step_tcl_powers = tcl_powers_horizon[:, 0]
            current_amb = ambient_temp_data[current_time]

            for i, tcl in enumerate(tcl_list):
                # 物理参数
                R, C, eta = tcl['R'], tcl['C'], tcl.get('eta', 2.5)
                power = first_step_tcl_powers[i]

                factor_a = np.exp(-deltaT / (R * C))
                factor_amb = 1 - factor_a
                factor_p = factor_amb * eta * R

                # 更新真实温度状态
                tcl_current_temps[i] = (factor_a * tcl_current_temps[i] +
                                        factor_amb * current_amb +
                                        factor_p * power)

            all_tcl_powers.append(first_step_tcl_powers.copy())
            all_tcl_temps.append(tcl_current_temps.copy())
        # 更新储能 SOC
        for i in range(n_ess):
            eta = ess_df.iloc[i]['eta_chg']
            power = first_step_ess_powers[i]
            # 根据你代码中的约定：power > 0 为放电，power < 0 为充电
            if power >= 0:
                ess_soc_list[i] -= power * deltaT / eta
            else:
                ess_soc_list[i] -= power * deltaT * eta
            ess_soc_list[i] = max(0, min(ess_soc_list[i], ess_df.iloc[i]['B']))

        # 记录历史
        all_ess_powers.append(first_step_ess_powers.copy())
        all_soc_history.append(ess_soc_list.copy())
        all_ev_powers.append(first_step_ev_powers.copy())
        current_time += 1

    print(f"\n滚动完成，共处理了 {len(all_ess_powers)} 个时间步")

    # 打包结果返回
    histories = {
        "ess_power": np.array(all_ess_powers),
        "ess_soc": np.array(all_soc_history),
        "alpha": np.array(all_alpha_history),
        "beta": np.array(all_beta_history),
        "ev_power": np.array(all_ev_powers),
        "tcl_power": np.array(all_tcl_powers),
        "tcl_temp": np.array(all_tcl_temps),
        "loss": np.array(all_loss_history),
        "p_cmd": np.array(all_p_cmd),  # 指令功率轨迹
        "p_actual": np.array(all_p_actual),  # 实际执行功率轨迹
        "ambient_temp": ambient_temp_data
    }

    return histories, ess_df, ev_list, tcl_list


if __name__ == '__main__':
    # 运行主函数
    # 注意：运行此代码需要 'PV_samples.mat' 和 'BL_samples.mat' 文件在同一目录下
    histories, ess_params, ev_params, tcl_params = rolling_power_decomposition()

    if histories is not None and len(histories["ess_power"]) > 0:
        # 1. 提取数据
        ess_powers_history = histories["ess_power"]

        # 提取 Alpha, Beta 和 Loss 的历史数据
        alpha_history = histories.get("alpha", [])
        beta_history = histories.get("beta", [])
        loss_history = histories.get("loss", [])
        p_cmd_history = histories.get("p_cmd", [])  # 指令功率轨迹
        p_actual_history = histories.get("p_actual", [])  # 实际执行功率轨迹

        time_steps = range(len(ess_powers_history))

        # =========================================================
        # 保存绘图数据为 .mat 文件
        # =========================================================
        scipy.io.savemat('plot_data.mat', {
            'alpha': np.array(alpha_history),
            'beta':  np.array(beta_history),
            'loss':  np.array(loss_history),
            'p_cmd': np.array(p_cmd_history),
            'p_actual': np.array(p_actual_history),

        })
        print("绘图数据已保存至 plot_data.mat")

        # =========================================================
        # 绘图设置：大字体、合并图表、时间轴、无标题
        # =========================================================
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker

        # 1. 全局字体设置 (大幅增大，适配论文插图)
        # 使用支持中文和英文的字体组合
        plt.rcParams['font.family'] = 'Arial'
        plt.rcParams['font.size'] = 20  # 基础字号
        plt.rcParams['axes.titlesize'] = 24  # (不显示标题，但保留设置)
        plt.rcParams['axes.labelsize'] = 24  # 轴标签字号
        plt.rcParams['xtick.labelsize'] = 20  # X轴刻度
        plt.rcParams['ytick.labelsize'] = 20  # Y轴刻度
        plt.rcParams['legend.fontsize'] = 20  # 图例字号
        plt.rcParams['lines.linewidth'] = 3.0  # 线条加粗
        plt.rcParams['axes.grid'] = True  # 开启网格
        plt.rcParams['grid.alpha'] = 0.5  # 网格透明度
        plt.rcParams['axes.unicode_minus'] = False


        # 2. 定义时间轴格式化函数 (HH:MM)
        # deltaT = 0.25 小时 (15分钟)
        def time_formatter(x, pos):
            # x 是时间步索引，每步代表 1 分钟
            total_minutes = int(x * 1)
            hours = (total_minutes // 60) % 24
            minutes = total_minutes % 60
            return f"{hours:02d}:{minutes:02d}"


        # 3. 设置X轴范围 (边缘对齐)
        x_limit_min = 0
        x_limit_max = len(time_steps) - 1

        # =========================================================
        # 绘制合并图：上图为参数收敛，下图为误差变化
        # =========================================================
        print("Generating Combined Figure: Parameters & Error...")

        # 创建 2行1列 的子图，共享 X 轴
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), sharex=True)
        plt.subplots_adjust(hspace=0.1)

        # --- 子图 1: 参数收敛 (Alpha & Beta) ---
        if len(alpha_history) > 0 and len(beta_history) > 0:
            ax1.plot(time_steps, alpha_history, label=r'$\alpha$ (Power)', color='#1f77b4')
            ax1.plot(time_steps, beta_history, label=r'$\beta$ (Energy)', color='#ff7f0e', linestyle='--')

            ax1.set_ylabel('Scaling Factor')
            ax1.legend(loc='center right', frameon=True)
            ax1.grid(True)
            ax1.set_xlim(x_limit_min, x_limit_max)

        # --- 子图 2: 误差变化 (Loss) [重点修改] ---
        if len(loss_history) > 0:
            # 1. 改为散点图 (Scatter), s=15大小, alpha=0.6透明度
            ax2.scatter(time_steps, loss_history, label='Loss', color='#d62728', s=15, alpha=0.6)

            # 2. 设置为对数坐标 (Log Scale)
            ax2.set_yscale('log')

            # 3. 添加 10^-4 阈值线
            ax2.axhline(y=1e-4, color='gray', linestyle='--', linewidth=2.5, label=r'Threshold $10^{-4}$')

            ax2.set_ylabel('Loss Value (Log Scale)')
            ax2.set_xlabel('Time (HH:MM)')
            ax2.legend(loc='upper right', frameon=True)

            # 开启主刻度和次刻度的网格，方便看对数
            ax2.grid(True, which="both", ls="-", alpha=0.2)
            ax2.set_xlim(x_limit_min, x_limit_max)

            # 4. 设置时间轴格式
            ax2.xaxis.set_major_formatter(ticker.FuncFormatter(time_formatter))

            # 5. 修改刻度间隔：4小时一个刻度
            # 4小时 * 60分钟/小时 = 240分钟 (即240个时间步)
            ax2.xaxis.set_major_locator(ticker.MultipleLocator(240))

        # 保存并显示
        plt.savefig('Fig_Combined_Convergence_Error.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("绘图完成：Fig_Combined_Convergence_Error.png 已生成。")
        """
        # =========================================================
        # 绘制 TCL 温度验证图
        # =========================================================
        tcl_temp_history = histories.get("tcl_temp", [])
        tcl_power_history = histories.get("tcl_power", [])
        if len(tcl_temp_history) > 0:
            tcl_temp_array = np.array(tcl_temp_history)  # shape: (time_steps, n_tcl)
            tcl_power_array = np.array(tcl_power_history)  # shape: (time_steps, n_tcl)
            n_tcl_plot = tcl_temp_array.shape[1]
            time_steps_tcl = range(len(tcl_temp_array))

            fig2, (ax_pwr, ax3) = plt.subplots(2, 1, figsize=(14, 14), sharex=True)
            plt.subplots_adjust(hspace=0.1)

            # --- 上图：TCL 功率曲线 ---
            for i in range(n_tcl_plot):
                ax_pwr.plot(time_steps_tcl, tcl_power_array[:, i], linewidth=1.5, alpha=0.8,
                            label=f'TCL[{i}] P_rated={tcl_params[i]["P_rated"]:.2f} kW')
            ax_pwr.set_ylabel('TCL Power (kW)')
            ax_pwr.legend(loc='upper right', frameon=True, fontsize=12)
            ax_pwr.yaxis.set_major_locator(ticker.MultipleLocator(0.05))
            ax_pwr.grid(True, which='both', alpha=0.3)
            ax_pwr.set_xlim(0, len(time_steps_tcl) - 1)

            # --- 下图：TCL 温度曲线 ---
            for i in range(n_tcl_plot):
                ax3.plot(time_steps_tcl, tcl_temp_array[:, i], linewidth=1.0, alpha=0.7,
                         label=f'TCL[{i}] T_set={tcl_params[i]["T_set"]:.1f}°C')
                # 画出该台TCL的温度上下限
                T_min_i = tcl_params[i]['T_set'] - tcl_params[i]['delta']
                T_max_i = tcl_params[i]['T_set'] + tcl_params[i]['delta']
                ax3.axhline(y=T_min_i, color='blue', linestyle=':', linewidth=0.8, alpha=0.4)
                ax3.axhline(y=T_max_i, color='red', linestyle=':', linewidth=0.8, alpha=0.4)

            ax3.set_ylabel('Indoor Temperature (°C)')
            ax3.set_xlabel('Time (HH:MM)')
            ax3.legend(loc='upper right', frameon=True, fontsize=12)
            ax3.grid(True, which='both', alpha=0.3)
            ax3.set_xlim(0, len(time_steps_tcl) - 1)

            # 根据实际温度数据范围自动放大 y 轴，留 0.5°C padding
            y_data_min = tcl_temp_array.min()
            y_data_max = tcl_temp_array.max()
            padding = 0.5
            ax3.set_ylim(y_data_min - padding, y_data_max + padding)

            # 主刻度 0.5°C，次刻度 0.05°C
            ax3.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
            ax3.yaxis.set_minor_locator(ticker.MultipleLocator(0.05))
            ax3.grid(True, which='minor', alpha=0.15)

            ax3.xaxis.set_major_formatter(ticker.FuncFormatter(time_formatter))
            ax3.xaxis.set_major_locator(ticker.MultipleLocator(240))

            # 检查并标出违反温度约束的时步
            for i in range(n_tcl_plot):
                T_min_i = tcl_params[i]['T_set'] - tcl_params[i]['delta']
                T_max_i = tcl_params[i]['T_set'] + tcl_params[i]['delta']
                viol_low = np.where(tcl_temp_array[:, i] < T_min_i - 1e-3)[0]
                viol_high = np.where(tcl_temp_array[:, i] > T_max_i + 1e-3)[0]
                if len(viol_low) > 0:
                    print(f"⚠ TCL[{i}] 温度低于下限 {len(viol_low)} 次，首次在步 {viol_low[0]}，温度={tcl_temp_array[viol_low[0], i]:.3f}°C，下限={T_min_i:.3f}°C")
                if len(viol_high) > 0:
                    print(f"⚠ TCL[{i}] 温度高于上限 {len(viol_high)} 次，首次在步 {viol_high[0]}，温度={tcl_temp_array[viol_high[0], i]:.3f}°C，上限={T_max_i:.3f}°C")
                if len(viol_low) == 0 and len(viol_high) == 0:
                    print(f"✓ TCL[{i}] 温度全程在 [{T_min_i:.1f}, {T_max_i:.1f}]°C 内")

            plt.tight_layout()
            plt.savefig('Fig_TCL_Temperature.png', dpi=300, bbox_inches='tight')
            plt.show()
            print("绘图完成：Fig_TCL_Temperature.png 已生成。")

        # =========================================================
        # 绘制 TCL 功率与能量边界曲线（在 current_time=0 时的完整窗口）
        # =========================================================
        ambient_temp_data = histories.get("ambient_temp", None)
        if tcl_params is not None and len(tcl_params) > 0 and ambient_temp_data is not None:
            deltaT_plot = 1 / 60
            T_horizon_plot = 2
            T_plot = int(T_horizon_plot / deltaT_plot)
            t_axis = np.arange(T_plot)  # 0..119，单位：分钟

            for i, tcl_p in enumerate(tcl_params):
                init_temp = tcl_p['T_set']  # 初始温度用设定值（仿真起始状态）
                lp_t, up_t, le_t, ue_t, pb_t = gen_tcl_boundaries(
                    tcl_p, init_temp, 0, T_plot, deltaT_plot, ambient_temp_data
                )

                fig_b, (ax_p, ax_e) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
                plt.subplots_adjust(hspace=0.15)

                # --- 上图：功率边界 ---
                ax_p.plot(t_axis, up_t, label='UP (功率上界)', color='#d62728')
                ax_p.plot(t_axis, lp_t, label='LP (功率下界)', color='#1f77b4', linestyle='--')
                ax_p.plot(t_axis, pb_t, label='P_base (基线)', color='#2ca02c', linestyle=':')
                ax_p.set_ylabel('Power (kW)')
                ax_p.legend(loc='upper right', frameon=True)
                ax_p.grid(True, alpha=0.4)
                ax_p.set_title(f'TCL[{i}]  R={tcl_p["R"]:.3f}  C={tcl_p["C"]:.3f}  P_rated={tcl_p["P_rated"]:.2f} kW  T_set={tcl_p["T_set"]}°C ±{tcl_p["delta"]}°C')

                # 检查功率边界是否有交叉
                cross_p = np.where(lp_t > up_t + 1e-6)[0]
                if len(cross_p) > 0:
                    ax_p.scatter(cross_p, lp_t[cross_p], color='red', zorder=5, s=40, label=f'LP>UP ({len(cross_p)}步)')
                    ax_p.legend(loc='upper right', frameon=True)
                    print(f"⚠ TCL[{i}] 功率边界交叉 {len(cross_p)} 步，首次在步 {cross_p[0]}")

                # --- 下图：能量边界 ---
                ax_e.plot(t_axis, ue_t, label='UE (能量上界)', color='#d62728')
                ax_e.plot(t_axis, le_t, label='LE (能量下界)', color='#1f77b4', linestyle='--')
                ax_e.set_ylabel('Cumulative Energy (kWh)')
                ax_e.set_xlabel('Time (min, from 00:00)')
                ax_e.legend(loc='upper left', frameon=True)
                ax_e.grid(True, alpha=0.4)

                # 检查能量边界是否有交叉
                cross_e = np.where(le_t > ue_t + 1e-6)[0]
                if len(cross_e) > 0:
                    ax_e.scatter(cross_e, le_t[cross_e], color='red', zorder=5, s=40, label=f'LE>UE ({len(cross_e)}步)')
                    ax_e.legend(loc='upper left', frameon=True)
                    print(f"⚠ TCL[{i}] 能量边界交叉 {len(cross_e)} 步，首次在步 {cross_e[0]}")

                if len(cross_p) == 0 and len(cross_e) == 0:
                    print(f"✓ TCL[{i}] 功率与能量边界均无交叉")

                plt.tight_layout()
                fname = f'Fig_TCL{i}_Boundaries.png'
                plt.savefig(fname, dpi=300, bbox_inches='tight')
                plt.show()
                print(f"绘图完成：{fname} 已生成。")"""