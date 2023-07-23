import matplotlib.pyplot as plt

from matplotlib.patches import Circle

circ = Circle([0, 0], 2, color="green", alpha=0.5)
ax = plt.gca()
ax.add_patch(circ)


circ = Circle([0.6, .5], 3, edgecolor="red", alpha=0.5, fill=False)
ax = plt.gca()
ax.add_patch(circ)
plt.xlim(-5,5)
plt.ylim(-5,5)

plt.savefig("example.png")
plt.close()