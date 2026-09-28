# 角色动作编排引擎

纯 Python 实现，先覆盖动作播放、条件状态切换和平滑混合；动作资源与角色实例分离，方便多个角色共用同一批片段。

## 模块

- `animation_engine/clip.py`：不可变关键帧动作、循环边界采样、根移动分段精确积分。
- `animation_engine/player.py`：每个角色独立的播放游标、速度、过渡冻结快照与播放状态存取。
- `animation_engine/state_machine.py`：命名状态和基于条件回调的状态切换。
- `animation_engine/library.py`：内置 idle/walk/run 与短动作 `step_dodge`。

## 连续性设计

过渡不是简单平均两个姿态：新切换会先冻结“当前混合结果”的关节值、根速度和角速度，再向目标片段做 smoothstep 混合。根部位移按混合权重对速度积分，循环边界和关键帧边界自动分段，因此在旧过渡未完成时再次切换不会产生位置跳变。

播放状态包含目标片段、局部时间、播放速度、根部位置/朝向、混合已用时间和冻结快照。保存后用新的播放器载入，再以相同时间步进独立采样，会与未中断播放得到一致结果。

## 运行

```bash
python -m unittest discover -s tests -v
python demo.py
```
