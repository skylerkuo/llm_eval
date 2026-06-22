---
name: move
description: 機器人移動到工廠中的某個地方。
---

# 任務說明

本文件用來處理機器人移動動作。

分類只使用：

- `robot_action`
- `clarify`

當使用者指令清楚時，分類為 `robot_action`。

當使用者指令缺少或方向不明確時，分類為 `clarify`。

---

# 可用動作與 args

## move_to

用途：控制機器人底盤移動到指定平面座標。

action name：`move_to`

args：

obj

args 說明：

- `obj`: str，目標要移動到的物品名稱

根據現有的物品名稱去選擇，且要完全相同的格式。

有的物品: red_cube, black_ball



