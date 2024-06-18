import gymnasium as gym  # Librería para crear y gestionar entornos de simulación
import numpy as np       # Librería para operaciones numéricas avanzadas
import matplotlib.pyplot as plt  # Librería para visualización de datos

def train(episodes):
    # Inicializa el entorno Taxi-v3
    env = gym.make("Taxi-v3")

    # Crea la tabla Q inicializada con valores optimistas para fomentar la exploración inicial
    q_table = np.ones((env.observation_space.n, env.action_space.n)) * 10

    # Define los parámetros del algoritmo
    learning_rate = 0.1            # Tasa de aprendizaje ajustada para la actualización de la tabla Q
    epsilon = 1.0                  # Probabilidad inicial de exploración (acciones aleatorias)
    epsilon_decay_rate = 0.002     # Tasa de decaimiento de epsilon ajustada para reducir gradualmente la exploración
    rng = np.random.default_rng()  # Generador de números aleatorios para decisiones de exploración

    # Inicializa un array para almacenar las recompensas obtenidas en cada episodio
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

        # Variables para controlar la finalización del episodio
        terminated = False
        truncated = False

        # Bucle para cada paso dentro de un episodio
        while not terminated and not truncated:
            # Decidir la acción basada en la política epsilon-greedy (exploración vs. explotación)
            if rng.random() < epsilon:
                action = env.action_space.sample()     # Exploración: Selecciona una acción aleatoria
            else:
                action = np.argmax(q_table[state, :])  # Explotación: Selecciona la mejor acción basada en la tabla Q

            # Realizar la acción y obtener el nuevo estado y la recompensa
            new_state, reward, terminated, truncated, _ = env.step(action)

            # Actualizar la tabla Q usando una implementación incremental con tasa de aprendizaje
            q_table[state, action] += learning_rate * (reward - q_table[state, action])

            # Actualizar el estado para el siguiente paso
            state = new_state

        # Reducir epsilon para disminuir la exploración a lo largo del tiempo
        epsilon = max(epsilon - epsilon_decay_rate, 0.01)

        # Registrar la recompensa obtenida en este episodio
        rewards_por_episode[i] = reward

        # Imprimir el progreso cada 10 episodios
        if (i + 1) % 10 == 0:
            print(f"Episodio: {i + 1} - Recompensa: {rewards_por_episode[i]}")

    # Cerrar el entorno al finalizar el entrenamiento
    env.close()

    # Imprimir la tabla Q final para la inspección
    print("Tabla Q resultante después del entrenamiento:")
    print(q_table)

    # Calcular y mostrar la suma de recompensas acumuladas en bloques de 10 episodios
    suma_rewards = np.zeros(episodes)
    for t in range(episodes):
        suma_rewards[t] = np.sum(rewards_por_episode[max(0, t - 10):(t + 1)])

    # Graficar la evolución de las recompensas acumuladas durante el entrenamiento
    plt.plot(suma_rewards)
    plt.xlabel('Episodios')
    plt.ylabel('Suma de recompensas acumuladas')
    plt.title('Evolución de las recompensas acumuladas durante el entrenamiento')
    plt.show()

if __name__ == "__main__":
    train(10000)

"""
El método de valores optimistas utilizado en este código es una estrategia donde inicialmente se sobreestima el valor esperado de las acciones para fomentar la exploración. Aquí están los puntos clave:

1. **Inicialización con Valores Altos**: La tabla Q se inicializa con valores altos (10 en este caso), lo que hace que el agente considere todas las acciones como muy prometedoras al principio.

2. **Exploración Temprana**: Debido a los valores optimistas, el agente tiende a explorar más agresivamente al principio para confirmar o refutar estas expectativas altas.

3. **Aprendizaje y Ajuste**: Con el tiempo, el agente ajusta gradualmente sus estimaciones basadas en las recompensas reales recibidas, utilizando una tasa de aprendizaje para actualizar la tabla Q.

4. **Reducción de la Exploración**: A medida que el entrenamiento avanza, la probabilidad de tomar acciones aleatorias (exploración) disminuye, ya que el valor de epsilon (probabilidad de exploración) decae a un ritmo establecido.

El método de valores optimistas es una forma efectiva de promover la exploración inicialmente, lo que puede acelerar el descubrimiento de políticas de acción más efectivas en entornos de aprendizaje por refuerzo como el `Taxi-v3` utilizado en este código.
"""