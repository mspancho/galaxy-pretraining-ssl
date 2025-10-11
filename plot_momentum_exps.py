import matplotlib.pyplot as plt

momentums = [0.9, 0.95, 0.99]
best_accs = [74.6899642944336, 77.22660064697266, 79.200]

plt.figure()
plt.plot(momentums, best_accs, marker='o')
plt.title("Peak Top-1 Accuracy for Momentum Experiments")
plt.xlabel("Momentum @ 200 Epochs")
plt.ylabel("Best Top-1 Accuracy")
plt.grid(True)
plt.savefig("figure.png", dpi=300, bbox_inches="tight")
plt.show()