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


def calculate_total_agg_boundaries(ess_df, pv_df, ev_list, tcl_list, current_time, deltaT,
                                   T_horizon, ess_soc_list, tcl_current_temps, ambient_temp_data, ev_current_energies):
    """
    基于内接功率边界的聚合边界计算

    为每个设备定义 p_u 和 p_l 变量（均为 N*T 维），满足：
    1. p_u 和 p_l 均满足每个设备的功率和能量边界约束
    2. p_u >= p_l（逐元素）
    3. 内接功率上下边界分别为 sum(p_u) 和 sum(p_l)（按 N 求和）
    4. 目标函数：最大化 (sum(p_u) - sum(p_l)) 按 T 求和

    返回: LP_final, UP_final, LE_final, UE_final, b_upper, b_lower, d_upper, d_lower, P_base
    """
    T = int(T_horizon / deltaT)

    # ==================== 收集所有设备信息 ====================
    devices = []  # 存储所有设备的信息和边界生成函数参数
    device_idx = 0

    # 1. 收集 ESS 设备
    for i, (_, row) in enumerate(ess_df.iterrows()):
        devices.append({
            'type': 'ESS',
            'params': (row.to_dict(), ess_soc_list[i], T, deltaT),
            'func': gen_ess_boundaries,
            'idx': device_idx
        })
        device_idx += 1

    # 2. 收集 PV 设备
    for _, row in pv_df.iterrows():
        devices.append({
            'type': 'PV',
            'params': (row.to_dict(), current_time, T, deltaT),
            'func': gen_pv_boundaries,
            'idx': device_idx
        })
        device_idx += 1

    # 3. 收集 EV 设备
    for i, ev_params in enumerate(ev_list):
        devices.append({
            'type': 'EV',
            'params': (ev_params, current_time, T, deltaT, ev_current_energies[i]),
            'func': lambda *args: gen_ev_window_boundaries(*args)[:4],  # 只取前4个返回值
            'idx': device_idx
        })
        device_idx += 1

    # 4. 收集 TCL 设备
    for i, tcl_params in enumerate(tcl_list):
        devices.append({
            'type': 'TCL',
            'params': (tcl_params, tcl_current_temps[i], current_time, T, deltaT, ambient_temp_data),
            'func': lambda *args: gen_tcl_boundaries(*args)[:4],  # 只取前4个返回值
            'idx': device_idx
        })
        device_idx += 1

    n_devices = len(devices)

    # ==================== 构建优化模型 ====================
    model = pyo.ConcreteModel()
    model.I = pyo.RangeSet(0, n_devices - 1)
    model.T = pyo.RangeSet(0, T - 1)

    # 决策变量：每个设备的上边界和下边界功率
    model.p_u = pyo.Var(model.I, model.T)  # p_u[i,t]: 设备i在时刻t的上边界功率
    model.p_l = pyo.Var(model.I, model.T)  # p_l[i,t]: 设备i在时刻t的下边界功率

    # ==================== 目标函数 ====================
    # 最大化 sum_t(sum_i(p_u[i,t] - p_l[i,t]))
    def obj_rule(m):
        return sum(m.p_u[i, t] - m.p_l[i, t] for i in m.I for t in m.T)

    model.obj = pyo.Objective(rule=obj_rule, sense=pyo.maximize)

    # ==================== 约束条件 ====================
    model.power_constraints_u = pyo.ConstraintList()
    model.power_constraints_l = pyo.ConstraintList()
    model.energy_constraints_u = pyo.ConstraintList()
    model.energy_constraints_l = pyo.ConstraintList()
    model.order_constraints = pyo.ConstraintList()  # p_u >= p_l

    for device in devices:
        i = device['idx']
        lp, up, le, ue = device['func'](*device['params'])

        for t in range(T):
            # --- 功率约束 ---
            # p_u[i,t] 必须在 [lp[t], up[t]] 范围内
            model.power_constraints_u.add(model.p_u[i, t] <= up[t])
            model.power_constraints_l.add(model.p_u[i, t] >= lp[t])

            # p_l[i,t] 必须在 [lp[t], up[t]] 范围内
            model.power_constraints_u.add(model.p_l[i, t] <= up[t])
            model.power_constraints_l.add(model.p_l[i, t] >= lp[t])

            # --- 顺序约束 ---
            # p_u[i,t] >= p_l[i,t]
            model.order_constraints.add(model.p_u[i, t] >= model.p_l[i, t])

        # --- 能量约束 ---
        for t in range(T):
            # 计算累计能量
            energy_u = sum(model.p_u[i, tau] * deltaT for tau in range(t + 1))
            energy_l = sum(model.p_l[i, tau] * deltaT for tau in range(t + 1))

            # p_u 的累计能量必须在 [le[t], ue[t]] 范围内
            model.energy_constraints_u.add(energy_u <= ue[t])
            model.energy_constraints_l.add(energy_u >= le[t])

            # p_l 的累计能量必须在 [le[t], ue[t]] 范围内
            model.energy_constraints_u.add(energy_l <= ue[t])
            model.energy_constraints_l.add(energy_l >= le[t])

    # ==================== 求解 ====================
    solver = pyo.SolverFactory('gurobi')
    solver.options['DualReductions'] = 0
    results = solver.solve(model, tee=False)

    if (results.solver.status != SolverStatus.ok) or (
            results.solver.termination_condition != TerminationCondition.optimal):
        raise RuntimeError(f"内接功率边界优化失败！状态: {results.solver.termination_condition}")

    # ==================== 计算聚合边界 ====================
    LP_final = np.zeros(T)
    UP_final = np.zeros(T)

    for t in range(T):
        for i in range(n_devices):
            LP_final[t] += pyo.value(model.p_l[i, t])
            UP_final[t] += pyo.value(model.p_u[i, t])

    # ==================== 计算 P_base ====================
    P_base = (LP_final + UP_final) / 2.0

    # ==================== 计算能量边界 ====================
    L_matrix = np.tril(np.ones((T, T))) * deltaT
    LE_final = np.dot(L_matrix, LP_final)
    UE_final = np.dot(L_matrix, UP_final)

    # ==================== 计算向后兼容的 b 和 d 向量 ====================
    # 为了保持与原代码接口的兼容性，计算 b_upper, b_lower, d_upper, d_lower
    # P_bar_raw = P_base + b_upper  =>  b_upper = P_bar_raw - P_base
    # P_under_raw = P_base - b_lower  =>  b_lower = -P_under_raw + P_base

    b_upper = UP_final - P_base
    b_lower = P_base - LP_final

    # 同理计算 d_upper 和 d_lower
    E_base = np.dot(L_matrix, P_base)
    d_upper = UE_final - E_base
    d_lower = E_base - LE_final

    return LP_final, UP_final, LE_final, UE_final, b_upper, b_lower, d_upper, d_lower, P_base

def sample_curves_from_boundaries(lp_agg, up_agg, P_cmd_t):
    """
    仅根据功率边界采样功率曲线（不考虑能量约束）

    参数:
        lp_agg: 功率下界
        up_agg: 功率上界
        deltaT: 时间步长
        P_cmd_t: 首步指令功率（可选）

    返回:
        采样的功率曲线（长度为T的数组）
    """
    T = len(lp_agg)
    v = np.random.randn(T)
    model = pyo.ConcreteModel()
    model.T = pyo.RangeSet(0, T - 1)
    model.p = pyo.Var(model.T)

    # 1. 目标函数：沿随机方向最小化
    model.obj = pyo.Objective(rule=lambda m: sum(v[t] * m.p[t] for t in m.T), sense=pyo.minimize)

    model.power_limits = pyo.ConstraintList()

    # 2. 对首步指令进行约束
    if P_cmd_t is not None:
        # 确保 P_cmd_t 在功率边界内
        P_cmd_t_safe = max(lp_agg[0], min(up_agg[0], P_cmd_t))
        model.cmd_constraint = pyo.Constraint(expr=model.p[0] == P_cmd_t_safe)

    # 3. 建立功率约束
    tolerance = 1e-6
    for t in model.T:
        model.power_limits.add(model.p[t] >= lp_agg[t] - tolerance)
        model.power_limits.add(model.p[t] <= up_agg[t] + tolerance)

    # 4. 求解
    solver = pyo.SolverFactory('gurobi')
    solver.options['DualReductions'] = 0
    solver.options['TimeLimit'] = 5

    results = solver.solve(model, tee=False)

    # 5. 状态判定
    if (results.solver.status == pyo.SolverStatus.ok) and \
            (results.solver.termination_condition == pyo.TerminationCondition.optimal):
        return np.array([pyo.value(model.p[t]) for t in model.T])
    else:
        error_msg = (
            f"\n[致命错误] 采样优化失败 (Infeasible)!\n"
            f"求解器状态: {results.solver.termination_condition}\n"
            f"请检查传入的功率边界是否有交叉。"
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
        sqrd_error = np.sum((delta_p0 - delta_p_star)**2)

        return sqrd_error, delta_p_star, p_actual_total, ess_powers, pv_powers, ev_powers, tcl_powers

    else:
        print(f"Critical Warning: Decomposition Optimization failed at step {current_time}.")
        raise RuntimeError(
            f"Decomposition Optimization Infeasible at time {current_time}. Solver condition: {results.solver.termination_condition}")

def rolling_power_decomposition():
    # ================= 参数初始化 =================
    deltaT = 1/60
    T_horizon = 2
    T = int(T_horizon / deltaT)

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
                'T_set': 20.3,  # 统一设定舒适温度为 21℃
                'delta': 5 # 设定温控死区为 ±1℃
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

    all_ess_powers, all_pv_powers = [], []
    all_ev_powers = []  # 记录历史
    all_tcl_powers = []
    all_tcl_temps = []
    all_soc_history = []
    all_loss_history = []
    # 记录指令功率和实际执行功率
    all_p_cmd = []  # 指令功率
    all_p_actual = []  # 实际聚合功率

    print("开始实时滚动功率分解 (Inner Boundary Mode)...")

    current_time = 0

    while current_time + T <= total_simulation_periods:

        print(f"\n--- 时间步 {current_time} ---")

        # 1. 计算内接功率边界（优化求解）
        (LP_agg, UP_agg, _, _, _, _, _, _, P_base) = calculate_total_agg_boundaries(
            ess_df, pv_df, ev_list, tcl_list, current_time, deltaT,
            T_horizon, ess_soc_list, tcl_current_temps, ambient_temp_data, ev_current_energies
        )

        # 2. 采样目标曲线（仅基于功率边界）
        p_cmd_current = np.random.uniform(LP_agg[0], UP_agg[0])
        p_target = sample_curves_from_boundaries(
            LP_agg, UP_agg, P_cmd_t=p_cmd_current)

        # 3. 求解分解问题
        (sqrd_error, _, p_actual_agg,
         ess_powers_horizon, pv_powers_horizon, ev_powers_horizon,
         tcl_powers_horizon) = solve_decomposition_strictly_step2(
            p_target, P_base, ess_df, pv_df, ev_list, tcl_list,
            ev_current_energies, tcl_current_temps, ambient_temp_data,
            current_time, deltaT, T_horizon, ess_soc_list
        )

        # 记录指令功率和实际执行功率（仅记录当前时间步）
        all_p_cmd.append(p_target[0])  # 指令：目标曲线的首步
        all_p_actual.append(p_actual_agg[0])  # 实际：聚合总功率的首步

        # 记录误差
        all_loss_history.append(sqrd_error)

        # 4. 资源状态更新
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
            eta_chg = ess_df.iloc[i]['eta_chg']
            power = first_step_ess_powers[i]
            # 根据你代码中的约定：power > 0 为放电，power < 0 为充电
            if power >= 0:
                ess_soc_list[i] -= power * deltaT / eta_chg
            else:
                ess_soc_list[i] -= power * deltaT * eta_chg
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
        loss_history = histories.get("loss", [])
        p_cmd_history = histories.get("p_cmd", [])  # 指令功率轨迹
        p_actual_history = histories.get("p_actual", [])  # 实际执行功率轨迹

        time_steps = range(len(ess_powers_history))

        # =========================================================
        # 保存绘图数据为 .mat 文件
        # =========================================================
        scipy.io.savemat('plot_data_inner_boundary.mat', {
            'loss': np.array(loss_history),
            'p_cmd': np.array(p_cmd_history),
            'p_actual': np.array(p_actual_history),
        })
        print("绘图数据已保存至 plot_data_inner_boundary.mat")

        # =========================================================
        # 绘图设置
        # =========================================================
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker

        # 1. 全局字体设置 (大幅增大，适配论文插图)
        plt.rcParams['font.family'] = 'Arial'
        plt.rcParams['font.size'] = 20  # 基础字号
        plt.rcParams['axes.labelsize'] = 24  # 轴标签字号
        plt.rcParams['xtick.labelsize'] = 20  # X轴刻度
        plt.rcParams['ytick.labelsize'] = 20  # Y轴刻度
        plt.rcParams['legend.fontsize'] = 20  # 图例字号
        plt.rcParams['lines.linewidth'] = 3.0  # 线条加粗
        plt.rcParams['axes.grid'] = True  # 开启网格
        plt.rcParams['grid.alpha'] = 0.5  # 网格透明度
        plt.rcParams['axes.unicode_minus'] = False

        # 2. 定义时间轴格式化函数 (HH:MM)
        def time_formatter(x, pos):
            total_minutes = int(x * 1)
            hours = (total_minutes // 60) % 24
            minutes = total_minutes % 60
            return f"{hours:02d}:{minutes:02d}"

        # 3. 设置X轴范围 (边缘对齐)
        x_limit_min = 0
        x_limit_max = len(time_steps) - 1

        # =========================================================
        # 绘制图1：功率轨迹（指令 vs 实际）
        # =========================================================
        print("Generating Power Trajectory Figure...")

        fig1, ax1 = plt.subplots(1, 1, figsize=(14, 8))

        if len(p_cmd_history) > 0 and len(p_actual_history) > 0:
            # 绘制指令功率
            ax1.plot(time_steps, p_cmd_history, label='Command Power ($P_{cmd}$)',
                     color='#1f77b4', linewidth=2.0, alpha=0.8)
            # 绘制实际执行功率
            ax1.plot(time_steps, p_actual_history, label='Actual Power ($P_{actual}$)',
                     color='#ff7f0e', linewidth=2.0, linestyle='--', alpha=0.8)

            ax1.set_ylabel('Power (kW)')
            ax1.set_xlabel('Time (HH:MM)')
            ax1.legend(loc='upper right', frameon=True)
            ax1.grid(True, alpha=0.3)
            ax1.set_xlim(x_limit_min, x_limit_max)

            # 设置时间轴格式
            ax1.xaxis.set_major_formatter(ticker.FuncFormatter(time_formatter))
            # 4小时一个刻度
            ax1.xaxis.set_major_locator(ticker.MultipleLocator(240))

        plt.savefig('Fig_Power_Trajectory.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("绘图完成：Fig_Power_Trajectory.png 已生成。")

        # =========================================================
        # 绘制图2：误差变化 (Loss)
        # =========================================================
        print("Generating Loss Figure...")

        fig2, ax2 = plt.subplots(1, 1, figsize=(14, 8))

        if len(loss_history) > 0:
            # 散点图
            ax2.scatter(time_steps, loss_history, label='Loss', color='#d62728', s=15, alpha=0.6)
            # 对数坐标
            ax2.set_yscale('log')
            # 添加 10^-4 阈值线
            ax2.axhline(y=1e-4, color='gray', linestyle='--', linewidth=2.5, label=r'Threshold $10^{-4}$')

            ax2.set_ylabel('Loss Value (Log Scale)')
            ax2.set_xlabel('Time (HH:MM)')
            ax2.legend(loc='upper right', frameon=True)
            ax2.grid(True, which="both", ls="-", alpha=0.2)
            ax2.set_xlim(x_limit_min, x_limit_max)

            # 设置时间轴格式
            ax2.xaxis.set_major_formatter(ticker.FuncFormatter(time_formatter))
            # 4小时一个刻度
            ax2.xaxis.set_major_locator(ticker.MultipleLocator(240))

        plt.savefig('Fig_Inner_Boundary_Loss.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("绘图完成：Fig_Inner_Boundary_Loss.png 已生成。")
