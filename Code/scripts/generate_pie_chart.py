import matplotlib.pyplot as plt

# Data
labels = ['Normal (pH >= 7.15)', 'Pathological (pH < 7.15)']
sizes = [512, 40]
colors = ['#4CAF50', '#F44336']
explode = (0, 0.1)  # explode the Pathological slice

fig, ax = plt.subplots(figsize=(7, 7))
ax.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.2f%%',
        shadow=True, startangle=140, textprops={'fontsize': 14, 'weight': 'bold'})
ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.title('CTU-UHB Dataset Class Distribution\n(Extreme Imbalance)', fontsize=16, weight='bold', pad=20)
plt.tight_layout()
plt.savefig(r'd:\Research Project\Research_Project\Code\figures\class_distribution.png', dpi=300)
print('Successfully generated class_distribution.png')
