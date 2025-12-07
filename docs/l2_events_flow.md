# 高速公路高D数据 L2 事件构建流程

本报告梳理 `main_02_events.py` 触发的 L2 事件生成逻辑，覆盖入口参数、数据依赖、冲突/基线事件的判定规则及相关公式。

## 入口与调度
- **CLI 入口**：`main_02_events.py` 接收 `--recordings`（录制列表或 `all`）和 `--frame-rate`（帧率，默认 `config.FRAME_RATE_DEFAULT`）。【F:main_02_events.py†L1-L37】
- **录制解析**：`parse_recordings` 支持逗号分隔或 `all`，`resolve_recordings` 在 TEST_MODE 时仅返回 `config.TEST_RECORDINGS`，否则全量录制。【F:main_02_events.py†L9-L33】
- **事件生成**：对每个录制，定位对应的 `L1_master_frame.parquet` 与输出事件目录，调用 `build_events_for_recording` 生成并落盘冲突/基线事件。【F:main_02_events.py†L39-L51】

## 事件构建主流程（`build_events_for_recording`）
1. **读取并过滤 L1**：加载指定录制的 L1 数据，仅保留匹配的 `recordingId`。
2. **冲突事件提取**：调用 `extract_high_interaction_events`，参数使用配置阈值或 CLI 覆盖：
   - `ttc_upper = config.TTC_CONF_THRESH`
   - `min_conf_duration = config.MIN_CONFLICT_DURATION`
   - 事件前后扩展窗口：`pre_event_time = config.PRE_EVENT_TIME`，`post_event_time = config.POST_EVENT_TIME`。
3. **基线事件提取**：调用 `extract_baseline_events`，使用默认滑窗长度、步长、加速度和 TTC 阈值。
4. **写入结果**：分别保存为 `L2_conflict_events.parquet` 与 `L2_baseline_events.parquet` 到 `data/processed/highD/events/recording_##/`。【F:data_preproc/events.py†L260-L308】

## 冲突事件判定逻辑（`extract_high_interaction_events`）
1. **按车辆遍历**：对每个 `(recordingId, trackId)` 组按帧排序，取 TTC 序列与帧号。【F:data_preproc/events.py†L34-L62】
2. **阈值检出**：`mask_conflict = TTC < ttc_upper`，通过 `find_contiguous_segments` 识别连续帧段（输出起止帧）。【F:data_preproc/events.py†L20-L33】【F:data_preproc/events.py†L56-L69】
3. **最短持续时间**：若连续段时长 `< min_conf_duration`（秒）则剔除。时长根据帧跨度或 `dt` 求和计算：`duration = Σ dt` 或 `(f_max - f_min + 1)/frame_rate`。【F:data_preproc/events.py†L35-L52】【F:data_preproc/events.py†L70-L89】
4. **窗口扩展**：在冲突段前后附加 `pre_event_time`、`post_event_time` 对应的帧数，限定在该车辆轨迹范围内。【F:data_preproc/events.py†L70-L89】
5. **事件特征汇总**：为每个事件输出：
   - 帧/时间范围与持续时间；
   - 冲突窗口最小 TTC (`min_TTC_conf`) 及整窗最小 TTC (`min_TTC`)
   - 车道变换次数：`num_lane_changes = max((lane != lane.shift()).sum() - 1, 0)`。【F:data_preproc/events.py†L54-L66】【F:data_preproc/events.py†L93-L125】
   - 能耗/排放：通过 `_energy_from_rate` 按 `dt` 或帧率累计 `cpf_co2_rate_gps`、`cpf_fuel_rate_lps`、`vsp_co2_rate`、`vsp_nox_rate`（单位对应 g 或 L）。【F:data_preproc/events.py†L37-L52】【F:data_preproc/events.py†L106-L121】
6. **数据类型规范**：对 ID、帧号等关键字段进行整型转换，便于下游存储与计算。【F:data_preproc/events.py†L127-L142】

### 相关公式
- **持续时间**：`duration = Σ dt`（若存在 dt），否则 `duration = (frame_max - frame_min + 1) / frame_rate`。【F:data_preproc/events.py†L35-L52】
- **冲突段持续时间**：同上，作用于 `conf_start_frame`~`conf_end_frame`。
- **能量/排放累计**：`E_rate = Σ (rate * dt)`，若无 `dt` 则 `E_rate = Σ rate / frame_rate`。【F:data_preproc/events.py†L37-L52】
- **车道变换计数**：`num_lane_changes = max((lane_i != lane_{i-1})累加 - 1, 0)`，首帧不计变化。【F:data_preproc/events.py†L43-L50】【F:data_preproc/events.py†L106-L121】

## 基线事件判定逻辑（`extract_baseline_events`）
1. **滑动窗口**：窗口长度 `window_size = round(window_time * frame_rate)`，步长 `step_size = round(step_time * frame_rate)`。【F:data_preproc/events.py†L144-L172】
2. **过滤条件**（窗口级）：
   - 最小帧数：窗口帧数不足则跳过。
   - 安全裕度：`min(TTC) > ttc_min`（默认 `config.TTC_BASE_MIN`）。
   - 平稳性：`max(|a_long_smooth|) < acc_thresh`（默认 `config.ACC_SMOOTH_THRESH`）。
   - 车道稳定：窗口内 `laneId_raw` 唯一值数 `≤ 1`（无变道）。【F:data_preproc/events.py†L172-L200】
3. **事件生成**：保留窗口的起止帧、时间、持续时间、TTC 统计（最小/平均）、车道变换数和各排放能量累计，与冲突事件字段保持一致格式。【F:data_preproc/events.py†L172-L228】
4. **类型转换**：与冲突事件相同，对 ID、帧号等字段转为整数。【F:data_preproc/events.py†L214-L228】

## 数据输入输出
- **输入**：`processed/recording_##/L1_master_frame.parquet`（需包含 TTC、平滑加速度、车道号、排放率等字段）。
- **输出**：`data/processed/highD/events/recording_##/L2_conflict_events.parquet` 与 `L2_baseline_events.parquet`。
- **帧率依赖**：持续时间与能量累计均基于传入 `frame_rate` 或帧级 `dt`，需与 L1 生成时保持一致。
