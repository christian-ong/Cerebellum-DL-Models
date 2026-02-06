import numpy as np
import matplotlib.pyplot as plt

# load "data/simulated_trajectories.npz"
data = np.load("data/simulated_trajectories.npz")
t = data["t"]
X = data["X"]

print(X.shape)  # should be (n_points, n_trajectories, 2)
print(t.shape)  # should be (n_points,)

# plot the trajectories
plt.figure(figsize=(8, 6))
plt.plot(X[:, 0], X[:, 1])
plt.title("Simulated Trajectories in Phase Space")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid()
plt.axis("equal")
plt.savefig("data/simulated_trajectories_separate_file.png")