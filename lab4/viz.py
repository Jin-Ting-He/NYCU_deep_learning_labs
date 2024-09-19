import numpy as np
import matplotlib.pyplot as plt

class kl_annealing:
    def __init__(self, num_epoch, kl_anneal_cycle, kl_anneal_ratio, kl_anneal_type='Cyclical'):
        self.current_epoch = 0
        self.n_iter = num_epoch
        self.start = 0.0
        self.stop = 1.0
        self.n_cycle = kl_anneal_cycle
        self.ratio = kl_anneal_ratio
        self.type = kl_anneal_type
        
    def update(self):
        self.current_epoch += 1
        
    def get_beta(self):
        if self.type == 'Monotonic':
            return self.monotonic_kl_annealing(0)
        elif self.type == 'Cyclical':
            return self.cyclical_kl_annealing(0)
        else:
            return 1

    def monotonic_kl_annealing(self, start_epoch):
        return min(1, 0.1 * self.current_epoch)

    def cyclical_kl_annealing(self, start_epoch):
        weight = 1.0 / (self.n_cycle * self.ratio)
        return min(1, weight * ((self.current_epoch - start_epoch) % (self.n_cycle+1)))

# Simulation settings
num_epoch = 70
kl_anneal_cycle = 5
kl_anneal_ratio = 1
kl_anneal_type = 'Monotonic'

# Create kl_annealing instance
annealer = kl_annealing(num_epoch, kl_anneal_cycle, kl_anneal_ratio, kl_anneal_type)

# Collect data
betas = []
for epoch in range(num_epoch):
    betas.append(annealer.get_beta())
    annealer.update()

# Plotting
plt.figure(figsize=(10, 5))
plt.plot(betas, marker='o', linestyle='-')
plt.title("Monotonic KL Annealing over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Beta")
plt.grid(True)
plt.savefig("monotonic.png")
