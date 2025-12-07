# 高速公路高D数据 L1 预处理流程

本文档详细描述 `main_01_preprocess.py` 触发的 L1 数据预处理管线，包括入口参数、任务分解、核心转换步骤以及关键公式。

## 入口与调度
- **CLI 入口**：`main_01_preprocess.py` 读取 `--recordings`（录制序号列表或 `all`）与 `--num-workers`（并行进程数）参数，并调用 `data_preproc.preprocessing.run_preprocessing`。【F:main_01_preprocess.py†L1-L34】
- **录制列表解析**：`parse_recording_ids_arg` 支持逗号分隔或 `all`，在未指定时返回 `None`。【F:data_preproc/preprocessing.py†L11-L23】
- **录制列表解析与测试模式**：`run_preprocessing` 内部通过 `_resolve_recording_ids`，在 TEST_MODE 下只处理 `config.TEST_RECORDINGS`，否则默认 1…`MAX_RECORDING_ID`。【F:data_preproc/preprocessing.py†L25-L44】
- **并行控制**：`run_preprocessing` 依据 CLI 覆盖或 `config.NUM_WORKERS` 设置工作进程数，实例化 `L1Builder` 并调用 `build_many`。

## 批处理/并行执行
- **单进程模式**：`build_many` 顺序遍历录制，逐条调用 `build_one`，打印进度。【F:data_preproc/l1_builder.py†L39-L56】
- **多进程模式**：当 `workers>1` 时，使用 `ProcessPoolExecutor` 并行处理，传递原始/输出目录、帧率等参数给 `_process_single_recording_entry`（同构造 `L1Builder` 再调用 `build_one`）。【F:data_preproc/l1_builder.py†L56-L74】【F:data_preproc/l1_builder.py†L205-L228】

## 单条录制的构建流程
`build_one` 对每条录制按下述顺序生成 `L1_master_frame.parquet`：
1. **加载输入**：读取录制元信息、轨迹元数据与逐帧轨迹（`load_recording_meta` / `load_tracks_meta` / `load_tracks`）。【F:data_preproc/l1_builder.py†L77-L96】
2. **元数据合并**：`_merge_meta_to_tracks` 将车辆级元数据（类别、尺寸等）并入帧级轨迹，统一字段命名并新增索引字段：
   - 全局 ID：`global_track_id = recordingId * 10000 + trackId`。
   - 时间：`time = frame / frameRate`。
   - 原始横纵坐标、车道号备份为 `x_raw`/`y_raw`/`laneId_raw`。【F:data_preproc/l1_builder.py†L98-L134】
3. **坐标系统一**：`_unify_coordinates` 使用行驶方向对纵向坐标与速度做符号统一：
   - 车辆中心位置：`x_center = x_raw + width/2`；若行驶方向为 2，则 `s_long = x_center`，否则 `s_long = max_x - x_center`。【F:data_preproc/l1_builder.py†L136-L151】
   - 横向中心：`d_lat = y_raw + height/2`。
   - 纵向速度/加速度：`v_long_raw = sign * xVelocity`，`a_long_raw = sign * xAcceleration`，其中 `sign` 在逆向时取负。【F:data_preproc/l1_builder.py†L147-L151】
4. **轨迹平滑**：`_smooth_kinematics` 对同一 `trackId` 的速度、加速度应用 Savitzky–Golay 滤波：
   - 平滑速度：`v_long_smooth = SG(v_long_raw)`。
   - 平滑加速度：`a_long_smooth = SG(v_long_raw, deriv=1)`。【F:data_preproc/l1_builder.py†L153-L164】
5. **交互特征**：`_compute_interactions` 基于前车信息计算跟驰指标：
   - 通过 `precedingId` 与帧号合并得到前车纵向位置/速度/加速度。
   - 净车头时距：`dist_headway = leader_s_long - s_long - 0.5*(leader_width + width)`。
   - 相对速度：`rel_velocity = v_long_smooth - leader_v_long`。
   - 反应式车头时距：`time_headway = dist_headway / v_long_smooth`（防零除）。【F:data_preproc/l1_builder.py†L166-L178】
6. **安全指标计算**：`_compute_safety_metrics` 计算 TTC、DRAC 及风险等级：
   - **TTC**：若重新计算，`TTC = gap / max(rel_v, 0)`；否则使用原始 ttc 字段并替换 0 为 NaN。【F:data_preproc/l1_builder.py†L180-L193】
   - **DRAC**：调用 `compute_drac`，公式 `DRAC = (rel_vel^2) / (2 * max(gap - rel_vel * t_r, 0))`，其中默认反应时间 `t_r=1s`。【F:utils/misc.py†L23-L44】
   - 风险标注：`TTC < TTC_HIGH_RISK → 2`，`TTC_LOW_RISK > TTC ≥ TTC_HIGH_RISK → 1`，否则 0。【F:data_preproc/l1_builder.py†L188-L194】
7. **排放估计（VT‑CPFM）**：`_compute_emissions_vt_cpfm` 调用 `apply_vt_cpfm_to_df`，按车型（LDV/HDDT）计算牵引功率、燃油与 CO₂ 排放率：
   - 牵引力：`F_t = m(1+λ)·a + m·g·Cr/1000·(c1·v_kmh + c2) + 0.5·ρ·A·Cd·v²`。
   - 功率：`P = F_t · v / (1000·η_d)`（kW）。
   - 燃油率：若 `P ≥ 0`，`FC = α0 + α1·P + α2·P²`，否则 `FC = α0`（L/s）。
   - CO₂ 率：`CO2 = FC · factor`，factor 取汽/柴油碳因子。【F:data_preproc/vt_cpfm.py†L36-L123】【F:data_preproc/vt_cpfm.py†L125-L169】
8. **排放估计（VSP 占位模型）**：`_compute_emissions_vsp` 计算车辆特定功率与分段排放：
   - VSP：`vsp = (v·a + c_air·v³ + c_roll·v + c_drive·v² + g·sin(grade)·v) / mass`。
   - 分段排放：`CO₂ = 0.7·vsp + 0.5 (vsp≥0 否则 0.5)`，`NOx = 0.1·vsp + 0.05 (vsp≥0 否则 0.05)`。【F:data_preproc/vsp.py†L1-L37】
   - 同时附加帧时长 `dt = 1/frameRate`。【F:data_preproc/l1_builder.py†L196-L204】
9. **可视化坐标**：`_add_visual_coordinates` 将米制坐标映射到道路示意图像素：`x_img = (x_raw - x_min)/(x_max - x_min)·img_w`，`y_img` 类似。【F:data_preproc/l1_builder.py†L206-L224】
10. **列重排与持久化**：按 `column_order()` 排列字段，写出 `processed/recording_##/L1_master_frame.parquet`，缺目录则创建。【F:data_preproc/l1_builder.py†L200-L228】

## 关键输入输出
- **输入目录**：`config.RAW_DATA_DIR` 下的 highD 原始 `recordingMeta.csv`、`tracksMeta.csv`、`tracks.csv`。
- **输出目录**：`config.PROCESSED_DATA_DIR/recording_##/L1_master_frame.parquet`。
- **并行性**：受 `--num-workers` 或 `config.NUM_WORKERS` 控制，多进程模式推荐在 Linux/WSL。

## 数学符号说明
- `v`、`a`：车辆纵向速度/加速度（m/s, m/s²）。
- `gap`：本车与前车前后保险杠净距离（m）。
- `rel_v`：本车速度相对前车的差值（m/s）。
- `TTC`：碰撞时间，越小风险越高。
- `DRAC`：避免碰撞所需减速度，越大风险越高。
- `FC`/`CO2`：燃油与排放瞬时率，基于 VT‑CPFM 计算。
