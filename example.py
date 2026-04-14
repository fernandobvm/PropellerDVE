from propellerDVE import *

def debug_plot_rotor(rotor, show_points=True):
    """
    Função de plotagem de depuração que desenha a superfície do rotor e,
    opcionalmente, os pontos de canto de cada DVE.
    (Versão com a lógica de reconstrução da malha corrigida)
    """
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    colors = ['skyblue', 'salmon', 'lightgreen', 'gold']

    # Itera sobre cada pá no rotor
    for k, blade in enumerate(rotor.blades):
        blade_color = colors[k % len(colors)]
        
        # --- LÓGICA DE PLOTAGEM DA SUPERFÍCIE (CORRIGIDA) ---
        m_chord = blade.m_chord
        n_span = blade.n_span
        dves = blade.dves

        # Cria as matrizes (meshgrid) para as coordenadas da superfície
        X = np.zeros((n_span, m_chord + 1))
        Y = np.zeros((n_span, m_chord + 1))
        Z = np.zeros((n_span, m_chord + 1))

        # Itera sobre os DVEs para preencher a malha de forma sistemática
        for j in range(n_span - 1):      # Itera sobre as fileiras na envergadura
            for i in range(m_chord):  # Itera sobre as colunas na corda
                dve_index = j * m_chord + i
                dve = dves[dve_index]

                # Preenche a malha usando os pontos de cada DVE.
                # A chave é preencher os pontos da borda de ataque (p1, p4) e depois
                # as bordas de fuga (p2, p3) apenas para a última coluna.
                X[j, i] = dve.p1[0]
                Y[j, i] = dve.p1[1]
                Z[j, i] = dve.p1[2]

                X[j+1, i] = dve.p4[0]
                Y[j+1, i] = dve.p4[1]
                Z[j+1, i] = dve.p4[2]
                
                # Para a última coluna de DVEs, preenchemos a borda de fuga da malha
                if i == m_chord - 1:
                    X[j, i+1] = dve.p2[0]
                    Y[j, i+1] = dve.p2[1]
                    Z[j, i+1] = dve.p2[2]

                    X[j+1, i+1] = dve.p3[0]
                    Y[j+1, i+1] = dve.p3[1]
                    Z[j+1, i+1] = dve.p3[2]
        #Zerar valores muito pequenos para melhor visualização
        #X[X < 1e-9] = 0
        #Y[Y < 1e-9] = 0
        Z[Z < 1e-9] = 0
        ax.plot_surface(X, Y, Z, color=blade_color, edgecolor='dimgray', alpha=0.9, rstride=1, cstride=1, linewidth=0.5)

        # --- LÓGICA PARA PLOTAR OS PONTOS DE CANTO (opcional) ---
        if show_points:
            for dve in blade.dves:
                points = np.array([dve.p1, dve.p2, dve.p3, dve.p4])
                ax.scatter(points[:, 0], points[:, 1], points[:, 2], color='black', s=10)

    # Configurações do gráfico
    ax.set_xlabel("X (Eixo de Rotação)")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Geometria do Rotor (Corrigida)")
    max_radius = rotor.blades[0].R * 1.2
    #ax.set_xlim([-max_radius, max_radius]); ax.set_ylim([-max_radius, max_radius]); ax.set_zlim([-max_radius, max_radius])
    #ax.set_box_aspect((1, 1, 1))
    #ax.view_init(elev=30, azim=-45)
    plt.show()


# Exemplo de pá com torção e afilamento simples
sections = [
    PropellerSection(r=1.0, chord=2, twist_deg=0),
    PropellerSection(r=3.0, chord=2, twist_deg=0),
    PropellerSection(r=5.0, chord=2, twist_deg=0),
    PropellerSection(r=7.0, chord=2, twist_deg=0),
    PropellerSection(r=9.0, chord=2, twist_deg=0),
    PropellerSection(r=11.0, chord=2, twist_deg=0),
]

propeller = PropellerGeometry(R=6.0, sections=sections)
#propeller.plot3d(real_profile=True)
#propeller.plot3d()

propeller.discretize2DVE(m=4, n=8)
#propeller.plotDVE(plot_normal=True)
#propeller.plot3d(real_profile=True)

rotor = Rotor(propeller, 2)
rotor.plot(real_profile=False)
#debug_plot_rotor(rotor, show_points=False)

