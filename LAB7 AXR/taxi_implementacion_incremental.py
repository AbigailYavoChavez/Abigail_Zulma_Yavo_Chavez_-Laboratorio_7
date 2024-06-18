# Importa el entorno de simulación y las herramientas de RL de Gymnasium
import gymnasium as gym
# Importa la biblioteca NumPy para manipulación de arrays y operaciones matemáticas
import numpy as np
# Importa Matplotlib para graficar datos
import matplotlib.pyplot as plt

def train(episodes):
    # Inicializa el entorno de Taxi-v3
    env = gym.make("Taxi-v3")

    # Crea la tabla Q inicializada con ceros para todas las combinaciones de estado-acción
    q_table = np.zeros((env.observation_space.n, env.action_space.n))

    # Define los parámetros del algoritmo Q-Learning
    learning_rate = 0.1            # Tasa de aprendizaje para actualizar la tabla Q
    epsilon = 1.0                  # Probabilidad inicial de exploración (seleccionar acciones aleatorias)
    epsilon_decay_rate = 0.01      # Tasa de decaimiento de epsilon (para reducir gradualmente la exploración)
    rng = np.random.default_rng()  # Generador de números aleatorios

    # Inicializa un array para almacenar las recompensas obtenidas en cada episodio
    rewards_por_episode = np.zeros(episodes)

    # Bucle principal de entrenamiento que se ejecuta durante el número de episodios especificados
    for i in range(episodes):
        # Reinicia el entorno cada 100 episodios, alternando entre modos con y sin renderización
        if (i + 1) % 100 == 0:
            env.close()  # Cierra el entorno actual
            env = gym.make("Taxi-v3", render_mode="human")  # Reinicia el entorno con renderización
        else:
            env.close()  # Cierra el entorno actual
            env = gym.make("Taxi-v3")  # Reinicia el entorno sin renderización

        # Reinicia el entorno y obtiene el estado inicial
        state = env.reset()[0]

        # Variables para controlar la finalización del episodio
        terminated = False  # Indica si el episodio ha terminado
        truncated = False   # Indica si el episodio ha sido truncado

        # Bucle para cada paso dentro de un episodio
        while not terminated and not truncated:
            # Decidir si tomar una acción exploratoria o explotatoria basada en epsilon
            if rng.random() < epsilon:
                action = env.action_space.sample()     # Exploración: Selecciona una acción aleatoria
            else:
                action = np.argmax(q_table[state, :])  # Explotación: Selecciona la mejor acción basada en la tabla Q

            # Realiza la acción y obtiene el nuevo estado y la recompensa
            new_state, reward, terminated, truncated, _ = env.step(action)

            # Actualiza la tabla Q usando la fórmula de Q-Learning
            q_table[state, action] += learning_rate * (reward - q_table[state, action])

            # Actualiza el estado actual para el próximo paso
            state = new_state

        # Reduce epsilon para disminuir la exploración a lo largo del tiempo, con un mínimo de 0.01
        epsilon = max(epsilon - epsilon_decay_rate, 0.01)

        # Registra la recompensa obtenida en este episodio
        rewards_por_episode[i] = reward

        # Imprime el progreso cada 10 episodios
        if (i + 1) % 10 == 0:
            print(f"Episodio: {i + 1} - Recompensa: {rewards_por_episode[i]}")

    # Cierra el entorno al finalizar el entrenamiento
    env.close()

    # Imprime la tabla Q final para la inspección
    print("Tabla Q resultante después del entrenamiento:")
    print(q_table)

    # Calcula y muestra la suma de recompensas acumuladas en bloques de 10 episodios
    suma_rewards = np.zeros(episodes)
    for t in range(episodes):
        suma_rewards[t] = np.sum(rewards_por_episode[max(0, t - 10):(t + 1)])

    # Grafica la evolución de las recompensas acumuladas durante el entrenamiento
    plt.plot(suma_rewards)
    plt.xlabel('Episodios')
    plt.ylabel('Suma de recompensas acumuladas')
    plt.title('Evolución de las recompensas acumuladas durante el entrenamiento')
    plt.show()

if __name__ == "__main__":
    train(7000)  # Llama a la función de entrenamiento con 7000 episodios


"""
la implementación incremental en Q-Learning para "Taxi-v3" permite que el agente mejore continuamente 
su rendimiento al ajustar las estimaciones de calidad de las acciones según las interacciones en el 
juego. Esto facilita una adaptación eficiente y efectiva del agente a las dinámicas y desafíos 
específicos presentados por el entorno del juego.
"""