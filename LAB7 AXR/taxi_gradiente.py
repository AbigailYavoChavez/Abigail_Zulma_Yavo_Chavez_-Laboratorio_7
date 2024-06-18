import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

def train(episodes):
    # Inicializa el entorno
    env = gym.make("Taxi-v3")

    # Inicializa las preferencias de la política (parámetros theta) como una matriz de ceros
    theta = np.zeros((env.observation_space.n, env.action_space.n))

    # Define los parámetros del algoritmo
    epsilon = 1.0                  # Probabilidad inicial de exploración (acciones aleatorias)
    epsilon_decay_rate = 0.01     # Tasa de decaimiento de epsilon más rápida
    rng = np.random.default_rng()  # Generador de números aleatorios
    alpha = 0.01                   # Tasa de aprendizaje
    gamma = 0.99                   # Factor de descuento

    # Inicializa un array para almacenar las recompensas obtenidas en cada episodio
    rewards_por_episode = np.zeros(episodes)

    def softmax(x):
        return np.exp(x)/sum(np.exp(x))
        #e_x = np.exp(x - np.max(x))  # Aplica el log-sum-exp trick para estabilidad numérica
        #return e_x / e_x.sum(axis=0)

    def choose_action(state):
        probabilities = softmax(theta[state])
        return np.random.choice(len(probabilities), p=probabilities)

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

        # Variables para controlar la finalización del episodio
        terminated = False
        truncated = False
        episode = []

        # Bucle para cada paso dentro de un episodio
        while not terminated and not truncated:
            # Tomar una acción en base a si es explotación o exploración basado en epsilon
            if rng.random() < epsilon:
                action = env.action_space.sample()     # Exploración: Selecciona una acción aleatoria
            else:
                action = choose_action(state)          # Explotación: Selecciona la acción basada en la política

            # Realizar la acción y obtiene el nuevo estado y la recompensa
            new_state, reward, terminated, truncated, _ = env.step(action)

            # Almacena la transición (estado, acción, recompensa)
            episode.append((state, action, reward))

            # Actualizar el estado para el siguiente paso
            state = new_state

        # Reduce epsilon para disminuir la exploración a lo largo del tiempo
        epsilon = max(epsilon - epsilon_decay_rate, 0.01)

        # Calcula la recompensa acumulada a partir de cada paso del episodio
        G = 0
        for t in reversed(range(len(episode))):
            state, action, reward = episode[t]
            G = reward + gamma * G

            # Actualiza los parámetros theta con la fórmula del gradiente de políticas
            probabilities = softmax(theta[state])
            for a in range(env.action_space.n):
                if a == action:
                    theta[state, a] += alpha * (G - np.sum(probabilities * theta[state])) * (1 - probabilities[a])
                else:
                    theta[state, a] += alpha * (G - np.sum(probabilities * theta[state])) * (-probabilities[a])

        # Registra la recompensa obtenida en este episodio
        rewards_por_episode[i] = G

        # Imprime el progreso cada 10 episodios
        if (i + 1) % 10 == 0:
            print(f"Episodio: {i + 1} - Recompensa: {rewards_por_episode[i]}")

    # Cierra el entorno al finalizar el entrenamiento
    env.close()

    # Imprime los parámetros theta finales para la inspección
    print("Parámetros theta resultantes después del entrenamiento:")
    print(theta)

    # Calcula y muestra la suma de recompensas acumuladas en bloques de 100 episodios
    suma_rewards = np.zeros(episodes)
    for t in range(episodes):
        suma_rewards[t] = np.sum(rewards_por_episode[max(0, t - 10) :(t + 1)])

    plt.plot(suma_rewards)
    plt.xlabel('Episodios')
    plt.ylabel('Suma de recompensas acumuladas')
    plt.title('Evolución de las recompensas acumuladas durante el entrenamiento')
    plt.show()

if __name__ == "__main__":
    train(10000)  # Aumenta el número de episodios