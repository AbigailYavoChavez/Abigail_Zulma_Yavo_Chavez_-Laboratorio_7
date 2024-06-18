import gymnasium as gym  # Librería para crear entornos de simulación para aprendizaje por refuerzo
import numpy as np       # Librería para operaciones numéricas avanzadas
import matplotlib.pyplot as plt  # Librería para la visualización de datos
import math              # Librería estándar de Python para funciones matemáticas

def train(episodes):
    # Inicializa el entorno Taxi-v3
    env = gym.make("Taxi-v3")

    # Crea la tabla Q inicializada con ceros para todas las combinaciones estado-acción
    q_table = np.zeros((env.observation_space.n, env.action_space.n))

    # Inicializa un contador de acciones por cada par estado-acción
    action_counts = np.zeros((env.observation_space.n, env.action_space.n))

    # Define los parámetros del algoritmo Q-learning
    learning_rate = 0.3           # Tasa de aprendizaje: ajusta cuánto se actualiza la tabla Q con cada iteración
    discount_factor = 0.9         # Factor de descuento: reduce el valor de las recompensas futuras en comparación con las actuales
    epsilon = 1.0                # Probabilidad inicial de exploración (epsilon-greedy)
    epsilon_decay_rate = 0.0003   # Tasa de decaimiento de epsilon: reduce la probabilidad de exploración con el tiempo
    rng = np.random.default_rng() # Generador de números aleatorios para decisiones de exploración

    # Array para almacenar las recompensas obtenidas en cada episodio
    rewards_por_episode = np.zeros(episodes)

    # Bucle principal de entrenamiento
    for i in range(episodes):
        # Reinicia el entorno cada 100 episodios, alternando entre modos con y sin renderización
        if (i + 1) % 100 == 0:
            env.close()
            env = gym.make("Taxi-v3", render_mode="human")
        else:
            env.close()
            env = gym.make("Taxi-v3")

        # Reinicia el entorno y obtiene el estado inicial
        state = env.reset()[0]
        terminated = False
        truncated = False

        # Bucle para cada paso dentro de un episodio
        while not terminated and not truncated:
            # Implementación de UCB para la selección de acciones
            if rng.random() < epsilon:
                action = env.action_space.sample()  # Exploración: selecciona una acción aleatoria
            else:
                total_counts = np.sum(action_counts[state, :])
                if total_counts == 0:
                    action = env.action_space.sample()
                else:
                    # Calcula los límites de confianza para cada acción usando UCB
                    confidence_bounds = q_table[state, :] + np.sqrt(2 * np.log(total_counts) / (action_counts[state, :] + 1e-10))
                    action = np.argmax(confidence_bounds)

            # Realiza la acción y obtiene el nuevo estado y la recompensa
            new_state, reward, terminated, truncated, _ = env.step(action)
            action_counts[state, action] += 1

            # Actualiza la tabla Q usando el algoritmo Q-learning
            q_table[state, action] = q_table[state, action] + learning_rate * (reward + discount_factor * np.max(q_table[new_state, :]) - q_table[state, action])

            # Actualiza el estado para el siguiente paso
            state = new_state

        # Reduce epsilon para disminuir la exploración a lo largo del tiempo
        epsilon = max(epsilon - epsilon_decay_rate, 0.01)

        # Registra la recompensa obtenida en este episodio
        rewards_por_episode[i] = reward

        # Imprime el progreso cada 50 episodios
        if (i + 1) % 50 == 0:
            print(f"Episodio: {i + 1} - Recompensa: {rewards_por_episode[i]}")

    # Cierra el entorno al finalizar el entrenamiento
    env.close()

    # Imprime la tabla Q final para la inspección
    print("Tabla Q resultante después del entrenamiento:")
    print(q_table)

    # Calcula y muestra la suma de recompensas acumuladas en bloques de 100 episodios
    suma_rewards = np.zeros(episodes)
    for t in range(episodes):
        suma_rewards[t] = np.sum(rewards_por_episode[max(0, t - 100) :(t + 1)])

    # Grafica la evolución de las recompensas acumuladas durante el entrenamiento
    plt.plot(suma_rewards)
    plt.xlabel('Episodios')
    plt.ylabel('Suma de recompensas acumuladas')
    plt.title('Evolución de las recompensas acumuladas durante el entrenamiento')
    plt.show()

if __name__ == "__main__":
    train(2000)

"""
El método UCB implementado equilibra la exploración inicial con la explotación de acciones prometedoras
basadas en la teoría de límites de confianza superiores. Esto permite al agente aprender de manera 
eficiente en el entorno `Taxi-v3`, priorizando acciones que podrían llevar a mayores recompensas 
mientras reduce gradualmente la exploración conforme acumula más experiencia.
"""