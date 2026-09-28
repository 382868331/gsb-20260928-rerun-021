# 角色动作编排引擎（第一阶段）

纯 Python、无第三方依赖。动作资源与角色实例分离：多个角色实例可共用同一
AnimationClip，但各自持有独立播放头、混合权重和根部位置。

## 模块

- anim_engine/clip.py — AnimationClip / Track：采样姿态（四元数 slerp、
  向量 lerp），循环边界接缝插值，基于累积根运动曲线的 root_delta（按完整
  循环周期计入步幅，速度任意也连续）。
- anim_engine/engine.py — AnimInstance：权重化多层混合；播放速度、淡入
  淡出、根位移在同一次 update(dt) 内一起推进。过渡中途再次切换时，旧层
  保留当前权重仅改目标，新层从 0 权重长出，姿态和位置在切换瞬间严格不变。
- anim_engine/rules.py — TransitionRule / evaluate_rules：按条件
  （context dict）按优先级触发平滑过渡。
- anim_engine/builtin_clips.py — 内置短动作 idle / walk / run（约 1 秒、
  循环），带每半周期脚步事件数据。
- anim_engine/mathutil.py — 四元数与向量工具。

## 关键设计

- 根运动不从“混合后的姿态”逐帧反推，而是每层先对自己的运动曲线积分
  （循环感知），再按权重相加——循环接缝和变速都不会造成位置跳变。
- 姿态混合对旋转用最短路径 slerp 累加，不是简单平均四元数分量。
- get_state() / set_state() 以 clip 名字引用资源，快照可 JSON 化；
  恢复后独立采样与连续播放逐位一致（见测试）。
- 脚步/表情事件为预留扩展点：clip 上的 events=[(local_time, name)]
  会在播放头跨过时回调（含循环绕行），表情可沿同一通道接入。

## 运行

    .venv/bin/python demo.py
    .venv/bin/python -m unittest discover -s tests -v

## 测试覆盖

- 保存恢复 == 连续播放（姿态与根位置，12 位小数一致）
- 过渡未完再次切换：切换瞬间零跳变，之后逐帧增量不超过纯 run 基线
- 根运动跨循环接缝连续，距离与步幅精确相符
- 播放速度同时缩放时间与位移
- 共享 clip 的多个实例播放头互相独立
- 条件规则触发、脚步事件跨循环触发
