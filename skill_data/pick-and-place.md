---
name: pick-and-place
description: 做拿取或是放下移動物品或是整理水果或物件的任務分解。
---

# 任務說明

本 skill 用於 Isaac Sim 桌面型機器人取放任務。當使用者提出自然語言指令時，請根據任務內容，將使用者指令分解為可執行的子任務序列。

# 輸出格式

每一個子任務請使用以下格式：

action object target

其中：

- action 表示任務類型。
- object 表示要處理的物件。
- target 表示目標位置。

若該任務沒有明確 target，則不要額外加入 target。

# 可用 action

只能使用以下 action：

- prepare：整理或準備某一類物件。
- put：將指定物件放到指定目標位置。
- tidyup：整理桌面物件或雜物。

# 可用 object

只能使用以下 object：

- fruits：水果類物件。
- orange：橘子。
- lemon：檸檬。
- cube：方塊。
- eraser：橡皮擦。
- properties：桌面物件或雜物。

# 可用 target

只能使用以下 target：

- tray：托盤。

# 可用子任務指令

只能輸出以下子任務指令，不要自行創造其他指令：

1. prepare fruits
2. put orange tray
3. put lemon tray
4. put cube tray
5. put eraser tray
6. tidyup properties

# 範例 1

使用者問題：幫我把橘子放到托盤。

任務分解：

1. put orange tray

# 範例 2

使用者問題：幫我把橘子、檸檬、方塊和橡皮擦放到托盤。

任務分解：

1. put orange tray
2. put lemon tray
3. put cube tray
4. put eraser tray

# 範例 3

使用者問題：幫我整理水果。

任務分解：

1. prepare fruits

# 範例 4

使用者問題：幫我整理桌面物件，然後把檸檬和橡皮擦放到托盤。

任務分解：

1. tidyup properties
2. put lemon tray
3. put eraser tray