import matplotlib.pyplot as plt
import numpy as np

# 1. 準備數據
categories = ['bird.png', 'girl.png', 'mountain.jpg']
methods = ['sequential', 'OpenMP', 'MPI', 'Cuda']

data = {
    'sequential': [0.36, 7.47, 143.08],
    'OpenMP':     [0.1, 5.83, 62.74],
    'MPI':        [0.74, 1.45, 18.82],
    'Cuda':       [0.21, 0.40, 2.21]
}

x = np.arange(len(categories))  # 標籤位置
width = 0.2  # 長條寬度

fig, ax = plt.subplots(figsize=(10, 6))

# 2. 繪製長條
rects1 = ax.bar(x - 1.5*width, data['sequential'], width, label='Sequential', color='#1f77b4')
rects2 = ax.bar(x - 0.5*width, data['OpenMP'], width, label='OpenMP', color='#ff7f0e')
rects3 = ax.bar(x + 0.5*width, data['MPI'], width, label='MPI', color='#2ca02c')
rects4 = ax.bar(x + 1.5*width, data['Cuda'], width, label='Cuda', color='#d62728')

# 3. 設定圖表細節
ax.set_ylabel('Time (seconds)')
ax.set_title('Performance of different methods')
ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.legend()

# 加入格線方便閱讀
ax.grid(axis='y', linestyle='--', alpha=0.7)

# 4. 加上數值標籤 (因為 mountain 數值太大，這裡選擇性標示或建議看 log 圖)
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 垂直偏移
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8, rotation=90)

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)
autolabel(rects4)

# *** 關鍵設定：設定 Y 軸為對數座標 ***
# 如果你想看一般比例，請把下面這行註解掉
ax.set_yscale('log') 
ax.set_ylabel('Time (seconds) - Log Scale')

plt.tight_layout()
fig.savefig('performance_transparent.png', transparent=True, dpi=300)
