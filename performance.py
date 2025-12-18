import matplotlib.pyplot as plt
import numpy as np

# 1. 準備數據
categories = ['bird.png', 'rain.png', 'mountain.jpg']
# data 結構: '方法': [bird秒數, rain秒數, mountain秒數]
data = {
    'sequential': [0.36, 7.47, 143.08],
    'OpenMP':     [0.1, 5.83, 62.74],
    'MPI':        [0.74, 1.45, 18.82],
    'Cuda':       [0.21, 0.40, 2.21]
}

x = np.arange(len(categories))  # 標籤位置
width = 0.2  # 長條寬度

# 調整畫布大小 (figsize)，讓大字體不會顯得擁擠
fig, ax = plt.subplots(figsize=(12, 8))

# 2. 繪製長條
rects1 = ax.bar(x - 1.5*width, data['sequential'], width, label='Sequential', color='#1f77b4')
rects2 = ax.bar(x - 0.5*width, data['OpenMP'], width, label='OpenMP', color='#ff7f0e')
rects3 = ax.bar(x + 0.5*width, data['MPI'], width, label='MPI', color='#2ca02c')
rects4 = ax.bar(x + 1.5*width, data['Cuda'], width, label='Cuda', color='#d62728')

# 3. 設定圖表細節 (這裡調整字體大小 fontsize)
# 標題
ax.set_title('Performance of Different Methods', fontsize=18)
# Y軸標籤
ax.set_ylabel('Time (seconds) - Log Scale', fontsize=18)
# X軸分類標籤 (bird, rain...)
ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=18)
# Y軸數值刻度
ax.tick_params(axis='y', labelsize=14)
# 圖例
ax.legend(fontsize=16)

# 加入格線
ax.grid(axis='y', linestyle='--', alpha=0.7)

# 4. 加上數值標籤 (字體加大)
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha='center', va='bottom', 
                    fontsize=14,    # 數值標籤字體大小
                    weight='bold')  # 數值標籤加粗

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)
autolabel(rects4)

# 使用對數座標 (Log Scale) 以清楚顯示差異極大的數值
ax.set_yscale('log')
ax.set_ylim(0.05, 400) # 調整 Y 軸範圍，確保上方標籤不被切到

plt.tight_layout()

# 儲存檔案：透明背景
plt.savefig('performance_large_font.png', transparent=True, dpi=300)
