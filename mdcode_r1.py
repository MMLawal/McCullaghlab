import numpy as np

def calculate_force(positions, box_size):
  """
  Calculates the Lennard-Jones force between particles.

  Args:
    positions: A NumPy array of particle positions (N x 3).
    box_size: The size of the simulation box.

  Returns:
    A NumPy array of forces acting on each particle (N x 3).
  """

  N = len(positions)
  forces = np.zeros_like(positions)

  for i in range(N):
    for j in range(i+1, N):
      # Calculate distance vector with minimum image convention
      dr = positions[i] - positions[j]
      dr -= box_size * np.round(dr / box_size) 

      # Calculate distance squared
      r2 = np.dot(dr, dr)

      # Calculate Lennard-Jones potential energy
      r6 = r2 * r2 * r2
      r12 = r6 * r6
      potential = 4.0 * (1.0/r12 - 1.0/r6)

      # Calculate Lennard-Jones force
      force = 48.0 * ((1.0/r12 - 0.5/r6) / r2) * dr
      forces[i] += force
      forces[j] -= force

  return forces

def velocity_verlet(positions, velocities, forces, dt, box_size):
  """
  Performs a single step of the Velocity Verlet algorithm.

  Args:
    positions: A NumPy array of particle positions (N x 3).
    velocities: A NumPy array of particle velocities (N x 3).
    forces: A NumPy array of forces acting on each particle (N x 3).
    dt: The time step.
    box_size: The size of the simulation box.

  Returns:
    Updated positions and velocities.
  """

  # Update velocities (half step)
  velocities += 0.5 * dt * forces

  # Update positions
  positions += dt * velocities

  # Apply periodic boundary conditions
  positions -= box_size * np.floor(positions / box_size)

  # Recalculate forces
  forces = calculate_force(positions, box_size)

  # Update velocities (full step)
  velocities += 0.5 * dt * forces

  return positions, velocities

# Simulation parameters
N = 2  # Number of particles
box_size = 10.0  # Size of the simulation box
dt = 0.001  # Time step
num_steps = 10000  # Number of simulation steps

# Initialize particle positions and velocities
positions = np.random.rand(N, 3) * box_size  # Random initial positions
velocities = np.random.randn(N, 3)  # Random initial velocities

# Run simulation
for step in range(num_steps):
  forces = calculate_force(positions, box_size)
  positions, velocities = velocity_verlet(positions, velocities, forces, dt, box_size)

  # Print progress (optional)
  if (step+1) % 1000 == 0:
    print(f"Step {step+1}/{num_steps}")
