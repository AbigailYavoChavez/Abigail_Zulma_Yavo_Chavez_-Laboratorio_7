import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

def train(episodes):
    # Inicializa el entorno Taxi-v3
    env = gym.make("Taxi-v3")

    # Crea la tabla Q inicializada con ceros para todas las combinaciones de estado-acción
    q_table = np.zeros((env.observation_space.n, env.action_space.n))

    # Contadores para la actualización incremental de la tabla Q
    action_counts = np.zeros((env.observation_space.n, env.action_space.n))
    rewards_sum = np.zeros((env.observation_space.n, env.action_space.n))

    # Define los parámetros del algoritmo
    epsilon = 1.0                  # Probabilidad inicial de exploración (acciones aleatorias)
    epsilon_decay_rate = 0.01      # Tasa de decaimiento de epsilon
    rng = np.random.default_rng()  # Generador de números aleatorios

    # Inicializa un array para almacenar las recompensas obtenidas en cada episodio
    rewards_por_episode = np.zeros(episodes)

    # Bucle principal de entrenamiento
    for i in range(episodes):
        # Reinicia el entorno y define el modo de renderización cada 100 episodios
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

        # Bucle para cada paso dentro de un episodio
        while not terminated and not truncated:
            # Decidir la acción basada en la política epsilon-greedy
            if rng.random() < epsilon:
                action = env.action_space.sample()     # Exploración: Selecciona una acción aleatoria
            else:
                action = np.argmax(q_table[state, :])  # Explotación: Selecciona la mejor acción basada en la tabla Q

            # Realizar la acción y obtener el nuevo estado y la recompensa
            new_state, reward, terminated, truncated, _ = env.step(action)

            # Actualiza el contador de acciones y la suma de recompensas para la acción tomada
            action_counts[state, action] += 1
            rewards_sum[state, action] += reward

            # Actualiza la tabla Q utilizando la actualización incremental (método de acción-valor)
            q_table[state, action] = rewards_sum[state, action] / action_counts[state, action]

            # Actualizar el estado para el siguiente paso
            state = new_state

        # Reduce epsilon para disminuir la exploración a lo largo del tiempo
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

    # Calcula y muestra la suma de recompensas acumuladas en bloques de 100 episodios
    suma_rewards = np.zeros(episodes)
    for t in range(episodes):
        suma_rewards[t] = np.sum(rewards_por_episode[max(0, t - 10):(t + 1)])

    plt.plot(suma_rewards)
    plt.xlabel('Episodios')
    plt.ylabel('Suma de recompensas acumuladas')
    plt.title('Evolución de las recompensas acumuladas durante el entrenamiento')
    plt.show()

if __name__ == "__main__":
    train(8000)  # Aumenta el número de episodios

"""
El método de acción-valor utilizado en este código para el entorno `Taxi-v3` ayuda al agente a aprender la mejor forma de actuar basándose en sus experiencias pasadas.

1. **Exploración vs. Explotación**: Inicialmente, el agente realiza muchas acciones aleatorias (exploración) para descubrir qué funciona mejor. Con el tiempo, disminuye la exploración y se enfoca en las acciones que ya sabe que son buenas (explotación).

2. **Actualización de la Tabla Q**: Cada vez que el agente realiza una acción, actualiza su tabla Q, que es una matriz que registra el valor promedio de cada acción en cada estado. Lo hace calculando el promedio de las recompensas obtenidas por cada acción, sin necesitar un factor de aprendizaje adicional.

3. **Mejora Continua**: A lo largo de los episodios, el agente se vuelve mejor tomando decisiones, lo que se refleja en el aumento de las recompensas acumuladas.

Este método permite al agente mejorar su desempeño de manera gradual y continua, balanceando entre explorar nuevas acciones y explotar el conocimiento adquirido para maximizar las recompensas.
"""