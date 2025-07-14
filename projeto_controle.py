import numpy as np
import matplotlib.pyplot as plt

# Tempo de simulação
T = 2.0
dt = 0.01
t = np.arange(0, T + dt, dt)

# Condições iniciais e finais: [θ1, θ2, d3]
q0 = np.array([0, 0, 0])  # rad, rad, m
qf = np.array([np.deg2rad(135), np.deg2rad(45), 0.4])  # rad, rad, m

# Trajetória desejada (linear)
q_des = np.linspace(q0, qf, len(t))

# Inicialização das variáveis do sistema
q = np.zeros_like(q_des)
dq = np.zeros_like(q_des)
err_int = np.zeros(3)
torques = []

# Parâmetros dinâmicos simplificados
I1 = 0.00858   # kg·m²
I2 = 0.00162   # kg·m²
m3 = 0.2037    # kg

# Ganhos PID para cada junta (ajustáveis)
Kp = np.array([0.858, 0.162, 20.37])
Kd = np.array([0.1201, 0.0227, 2.8518])
Ki = np.array([0.6, 0.1, 15])  # moderate integral gain

# Simulação do controle PID
for i in range(1, len(t)):
    error = q_des[i] - q[i-1]
    derror = (q_des[i] - q_des[i-1]) / dt
    err_int += error * dt

    # PID
    u = Kp * error + Kd * derror + Ki * err_int
    torques.append(u)

    # Modelo de planta: torque → aceleração (simplificado)
    acc = np.array([
        u[0] / I1,
        u[1] / I2,
        u[2] / m3
    ])

    dq[i] = dq[i-1] + acc * dt
    q[i] = q[i-1] + dq[i] * dt

torques = np.array(torques)

# Gráficos
fig, axs = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

# Seguimento da trajetória
axs[0].plot(t, q_des[:, 0], '--', label='θ1 desejado')
axs[0].plot(t, q[:, 0], label='θ1 simulado')
axs[0].plot(t, q_des[:, 1], '--', label='θ2 desejado')
axs[0].plot(t, q[:, 1], label='θ2 simulado')
axs[0].plot(t, q_des[:, 2], '--', label='d3 desejado')
axs[0].plot(t, q[:, 2], label='d3 simulado')
axs[0].set_ylabel("Posição")
axs[0].legend()
axs[0].grid()

# Erro ao longo do tempo
axs[1].plot(t, q_des - q)
axs[1].set_ylabel("Erro")
axs[1].legend(["θ1", "θ2", "d3"])
axs[1].grid()

# Sinais de controle
axs[2].plot(t[1:], torques[:, 0], label="τ1 [Nm]")
axs[2].plot(t[1:], torques[:, 1], label="τ2 [Nm]")
axs[2].plot(t[1:], torques[:, 2], label="f3 [N]")
axs[2].set_ylabel("Controle")
axs[2].legend()
axs[2].grid()

# Comparativo (teoria vs simulação)
axs[3].plot(t, q_des[:, 0] - q[:, 0], '--', label='Erro θ1')
axs[3].plot(t, q_des[:, 1] - q[:, 1], '--', label='Erro θ2')
axs[3].plot(t, q_des[:, 2] - q[:, 2], '--', label='Erro d3')
axs[3].set_xlabel("Tempo [s]")
axs[3].set_ylabel("Teoria vs Simulação")
axs[3].legend()
axs[3].grid()

plt.tight_layout()
plt.savefig("simulacao_pid_marvs.png", dpi=300)
plt.savefig('foo.png')

