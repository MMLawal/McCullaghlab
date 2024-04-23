import numpy as np
from math import sqrt

class Particle:
  """
  Represents a particle in the simulation.
  """
  def __init__(self, mass, position, velocity):
    self.mass = mass
    self.position = np.array(position)  # Position as a NumPy array
    self.velocity = np.array(velocity)  # Velocity as a NumPy array

def LJ_potential(r, sigma, epsilon):
  """
  Calculates Lennard-Jones potential energy between two particles.

  Args:
      r: Distance between particle centers.
      sigma: Lennard-Jones sigma parameter.
      epsilon: Lennard-Jones epsilon parameter.

  Returns:
      The Lennard-Jones potential energy.
  """
  sigma_6 = sigma**6
  return 4 * epsilon * (sigma_6 / r**12 - sigma_6 / r**6)

def LJ_force(r, sigma, epsilon):
  """
  Calculates Lennard-Jones force acting on one particle due to another.

  Args:
      r: Distance between particle centers.
      sigma: Lennard-Jones sigma parameter.
      epsilon: Lennard-Jones epsilon parameter.

  Returns:
      The Lennard-Jones force as a NumPy array.
  """
  sigma_6 = sigma**6
  F = 24 * epsilon * (2 * sigma_6 / r**13 - sigma_6 / r**7)
  return F * (r / np.linalg.norm(r))  # Scale force by unit vector

def apply_periodic_boundary(particle, box_size):
  """
  Applies periodic boundary conditions to a particle's position.

  Args:
      particle: The Particle object.
      box_size: The simulation box size (side length).
  """
  particle.position = particle.position % box_size  # Wrap position around box

def compute_forces(particles, box_size, sigma, epsilon):
  """
  Calculates the net force acting on each particle.

  Args:
      particles: A list of Particle objects.
      box_size: The simulation box size (side length).
      sigma: Lennard-Jones sigma parameter.
      epsilon: Lennard-Jones epsilon parameter.

  Returns:
      A list of NumPy arrays, where each element represents the force on a particle.
  """
  forces = []
  for i, particle_i in enumerate(particles):
    force = np.zeros_like(particle_i.position)
    for j, particle_j in enumerate(particles):
      if i != j:  # Avoid self-interaction
        r_ij = particle_i.position - particle_j.position
        apply_periodic_boundary(r_ij, box_size)
        r_mag = np.linalg.norm(r_ij)
        force += LJ_force(r_mag, sigma, epsilon)
    forces.append(force)
  return forces

def simulate(particles, box_size, sigma, epsilon, temperature, dt, steps):
  """
  Runs a Lennard-Jones MD simulation for a specified number of steps.

  Args:
      particles: A list of Particle objects.
      box_size: The simulation box size (side length).
      sigma: Lennard-Jones sigma parameter.
      epsilon: Lennard-Jones epsilon parameter.
      temperature: Temperature (in reduced units).
      dt: Time step.
      steps: Number of simulation steps.
  """

  # Initialize velocities based on temperature
  masses = [particle.mass for particle in particles]
  v_squared = temperature * (3 * np.ones(len(particles)))  # Reduced units
  v_rms = np.sqrt(np.sum(v_squared) / len(particles))
  velocities = [v_rms * np.random.randn(3) for _ in particles]  # Randomize direction
  for particle, velocity in zip(particles, velocities):
    particle.velocity = velocity * np.sqrt(particle.mass / masses[0])  # Scale for individual masses

  # Main simulation loop
  for _ in range(steps):
    # Apply periodic boundary conditions
    for particle in particles:
      apply_periodic_boundary(particle, box_size)

    # Calculate forces
    forces = compute_forces(particles, box_size, sigma, epsilon)
