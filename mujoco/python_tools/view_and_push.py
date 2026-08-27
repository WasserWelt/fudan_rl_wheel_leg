"""
交互查看 infantry_V2 模型，并对指定 link 施加恒定外力（xfrc_applied）。

用法示例：
    python view_and_push.py --link rf1_Link --force 0 0 30
    python view_and_push.py --link r22_Link --force 20 0 0 --spring 300

- --link   : 要施力的 body 名（见下方 BODY_NAMES）
- --force  : 世界坐标系下的力 [fx fy fz] (N)，作用在该 body 质心
- --torque : 世界坐标系下的力矩 [mx my mz] (N·m)，可选
- --spring : 给两个 loop1_motor 写的气弹簧恒定 ctrl（默认 0 = 不出力；脚本原值 300）
- --pause  : 启动即暂停物理，只看造型

在 viewer 里依旧可以用鼠标 Ctrl+拖拽额外施力；本脚本的力是叠加的、恒定的。
"""
import argparse
import time

import mujoco
import mujoco.viewer

MJCF = "../assert_now/infantry_binglian_yuntai/infantry_V2/meshes/mjmodel.xml"

BODY_NAMES = [
    "base_Link_del",
    "rf0_Link", "rf1_Link", "r_wheel_Link",
    "lf0_Link", "lf1_Link", "l_wheel_Link",
    "r20_Link", "r21_Link", "r22_Link", "r23_Link",
    "l20_Link", "l21_Link", "l22_Link", "l23_Link",
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mjcf", default=MJCF)
    p.add_argument("--link", default="rf1_Link", choices=BODY_NAMES)
    p.add_argument("--force", type=float, nargs=3, default=[0.0, 0.0, 0.0])
    p.add_argument("--torque", type=float, nargs=3, default=[0.0, 0.0, 0.0])
    p.add_argument("--spring", type=float, default=0.0)
    p.add_argument("--pause", action="store_true")
    args = p.parse_args()

    m = mujoco.MjModel.from_xml_path(args.mjcf)
    d = mujoco.MjData(m)

    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, args.link)
    if bid < 0:
        raise KeyError(f"body '{args.link}' not found")

    # 气弹簧 motor（可选）
    spring_ids = []
    for name in ("Left_loop1_motor", "Right_loop1_motor"):
        aid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if aid >= 0:
            spring_ids.append(aid)

    wrench = args.force + args.torque  # [fx fy fz mx my mz]
    print(f"对 body '{args.link}' (id={bid}) 施加 xfrc_applied = {wrench}")
    print(f"气弹簧 ctrl = {args.spring}，作用于 actuators {spring_ids}")

    with mujoco.viewer.launch_passive(m, d) as viewer:
        paused = args.pause
        while viewer.is_running():
            step_start = time.time()
            if not paused:
                # 每步重置再写，避免 viewer 的鼠标扰动被覆盖冲突
                d.xfrc_applied[bid, :] = wrench
                for aid in spring_ids:
                    d.ctrl[aid] = args.spring
                mujoco.mj_step(m, d)
            viewer.sync()
            dt = m.opt.timestep - (time.time() - step_start)
            if dt > 0:
                time.sleep(dt)


if __name__ == "__main__":
    main()
