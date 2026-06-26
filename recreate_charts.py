import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os

# Set style
plt.style.use('default')
sns.set_palette("pastel")

# Output dir
os.makedirs('fixed_charts', exist_ok=True)

# 1. Phân bố độ tuổi (Histogram)
fig1, ax1 = plt.subplots(figsize=(7, 5.5))
# Approximate data
bins = [13, 15, 17.5, 20, 22.5, 24.5, 26.5, 29, 31, 33.5, 36, 38, 40, 42.5, 45, 47, 49, 51.5, 53.5, 56, 58, 61, 63.5, 66, 68, 71, 74, 80]
heights = [14600, 23000, 22500, 14600, 15000, 7500, 6200, 5200, 6200, 3800, 3100, 2900, 4100, 2200, 2300, 2000, 1900, 2400, 1400, 1100, 900, 950, 450, 350, 250, 150, 50]

# Generating data to fit the histogram exactly using bar plot to mimic the original histogram bins
widths = np.diff(bins)
centers = [(bins[i] + bins[i+1])/2 for i in range(len(bins)-1)]

ax1.bar(centers, heights, width=widths, color='skyblue', edgecolor='black', alpha=0.9, linewidth=1)
ax1.set_title('Phân bố độ tuổi', fontsize=12)
ax1.set_xlabel('Tuổi', fontsize=10)
ax1.set_ylabel('Số lượng', fontsize=10)
ax1.set_xlim(9, 83)
ax1.set_ylim(0, 24000)
plt.tight_layout()
fig1.savefig('fixed_charts/age_distribution.png', dpi=300)
plt.close(fig1)

# 2. Phân bố giới tính (Bar Chart)
fig2, ax2 = plt.subplots(figsize=(6.5, 5))
genders = ['Nam', 'Nữ', 'Khác']
gender_counts = [49000, 95000, 1500]
colors2 = sns.color_palette("pastel")[0:3]

ax2.bar(genders, gender_counts, color=colors2, edgecolor='white', linewidth=1)
ax2.set_title('Phân bố giới tính', fontsize=12)
ax2.set_xlabel('Giới tính', fontsize=10)
ax2.set_ylabel('Số lượng', fontsize=10)
ax2.set_ylim(0, 100000)
plt.tight_layout()
fig2.savefig('fixed_charts/gender_distribution.png', dpi=300)
plt.close(fig2)

# 3. Phân bố trình độ học vấn (Bar Chart)
fig3, ax3 = plt.subplots(figsize=(6.5, 5))
edu_levels = ['Dưới THPT', 'THPT', 'Đại học', 'Sau đại học']
edu_counts = [22500, 62500, 39000, 20500]
# The original colors seem to be from pastel palette but mapped differently: orange, blue, green, red
colors3 = [sns.color_palette("pastel")[1], sns.color_palette("pastel")[0], sns.color_palette("pastel")[2], sns.color_palette("pastel")[3]]

ax3.bar(edu_levels, edu_counts, color=colors3, edgecolor='white', linewidth=1)
ax3.set_title('Phân bố trình độ học vấn', fontsize=12)
ax3.set_xlabel('Trình độ học vấn', fontsize=10)
ax3.set_ylabel('Số lượng', fontsize=10)
ax3.set_ylim(0, 66000)
plt.tight_layout()
fig3.savefig('fixed_charts/education_distribution.png', dpi=300)
plt.close(fig3)

print("Charts successfully created in fixed_charts/ directory.")
