import copy
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import interp1d

class PropellerSection:
    def __init__(self, r, chord, twist_deg, profile = "NACA 0012", n_points = 50):
        self.r = r                  # Distância radial [m]
        self.chord = chord          # Corda local [m]
        self.twist = np.deg2rad(twist_deg)  # Torção [rad]
        self.profile = profile # NACA xxxx ou Clark Y
        self.n_points = n_points

        self.generate_coordinates()

    def generate_coordinates(self):
        if self.profile.upper().startswith("NACA"):
            code = self.profile.split(" ")[1]
            x , z = self.generate_naca(code)
            
        elif self.profile.lower() in ["clark y", "clarky", "clark-y"]:
           x, z = self.generate_clarky()
        else:
            raise ValueError(f"Airfoil desconhecido ou formato inválido: '{self.profile}'")
        self.x = x
        self.z = np.mean(z) if np.mean(z) > 0.001 else 0 # Aerofólios finos
        self.z_real = z
 
    def generate_naca(self, code='0012'):
        """
        Gera coordenadas para um perfil NACA 4 dígitos.
        Ex: '2412' => m=0.02, p=0.4, t=0.12
        """
        code = code.strip().upper().replace("NACA", "")
        if len(code) != 4 or not code.isdigit():
            raise ValueError(f"Código NACA inválido: '{code}'")

        m = int(code[0]) / 100.0       # Máximo camber
        p = int(code[1]) / 10.0        # Posição do camber
        t = int(code[2:]) / 100.0      # Espessura relativa

        x = 0.5 * (1 - np.cos(np.linspace(0, np.pi, self.n_points)))  # distribuição de coseno

        # Linha de camber e derivada
        yc = np.zeros_like(x)
        dyc_dx = np.zeros_like(x)

        for i, xi in enumerate(x):
            if xi < p and p != 0:
                yc[i] = m / p**2 * (2 * p * xi - xi**2)
                dyc_dx[i] = 2 * m / p**2 * (p - xi)
            elif p != 0:
                yc[i] = m / (1 - p)**2 * ((1 - 2*p) + 2*p*xi - xi**2)
                dyc_dx[i] = 2 * m / (1 - p)**2 * (p - xi)
            else:
                yc[i] = 0
                dyc_dx[i] = 0

        theta = np.arctan(dyc_dx)

        yt = 5 * t * (
            0.2969 * np.sqrt(x) -
            0.1260 * x -
            0.3516 * x**2 +
            0.2843 * x**3 -
            0.1015 * x**4
        )

        # Superfície superior
        xu = x - yt * np.sin(theta)
        zu = yc + yt * np.cos(theta)

        # Superfície inferior
        xl = x + yt * np.sin(theta)
        zl = yc - yt * np.cos(theta)

        # Combinar as superfícies em sentido horário
        x_coords = np.concatenate([xu[::-1], xl[1:]])
        z_coords = np.concatenate([zu[::-1], zl[1:]])

        return x_coords, z_coords

    def generate_clarky(self):
        """
        Gera coordenadas aproximadas para o perfil Clark Y (normalizado para corda 1).
        Baseado em dados experimentais suavizados.
        """
        # Distribuição de x por coseno para maior resolução perto de LE e TE
        x = 0.5 * (1 - np.cos(np.linspace(0, np.pi, self.n_points)))

        # Aproximação polinomial para Clark Y (perfil assimétrico com fundo plano)
        # Fonte: Airfoil Tools + ajuste visual
        yt = 0.12 * (
            0.2969 * np.sqrt(x) -
            0.1260 * x -
            0.3516 * x**2 +
            0.2843 * x**3 -
            0.1036 * x**4
        )

        # Linha média (camber) aproximada
        camber = 0.035 * np.sin(np.pi * x)  # estimativa simplificada

        # Superfície superior e inferior
        xu = x
        zu = camber + yt

        xl = x
        zl = camber - yt

        # Combinar em sentido horário
        x_coords = np.concatenate([xu[::-1], xl[1:]])
        z_coords = np.concatenate([zu[::-1], zl[1:]])

        return x_coords, z_coords

class PropellerGeometry:
    def __init__(self, R, sections):
        self.R = R                  # Raio do rotor
        self.sections = sections    # Lista de PropellerSection
        self.n_sections = len(sections)
    
    def get_spanwise_coordinates(self):
        return [s.r for s in self.sections]

    def get_chords(self):
        return [s.chord for s in self.sections]

    def get_twists(self):
        return [s.twist for s in self.sections]
    
    def get_profiles(self):
        return [s.profile for s in self.sections]
    
    def plot3d(self, real_profile = False):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax = self.plot_blade(ax, real_profile)

        ax.set_xlabel("X (Chord)")
        ax.set_ylabel("Y (Span)")
        ax.set_zlabel("Z")
        ax.set_title("Propeller Geometry")
        ax.view_init(elev=25, azim=-120)
        plt.tight_layout()
        plt.show()

    def plot_dve_mesh_with_indices(self, ax=None):
        """
        Plota a malha de DVEs da pá, exibindo o índice de cada DVE e
        destacando as arestas da raiz e da ponta.

        Args:
            ax (matplotlib.axes._subplots.Axes3DSubplot, optional): Um eixo 3D existente
                para plotar. Se None, uma nova figura será criada.
        """
        if ax is None:
            fig = plt.figure(figsize=(12, 9))
            ax = fig.add_subplot(111, projection='3d')
            created_fig = True
        else:
            created_fig = False

        m = self.m_chord
        n_panels_span = self.n_span - 1

        # Itera sobre cada DVE para desenhar a malha e os índices
        for i, dve in enumerate(self.dves):
            points = np.array([dve.p1, dve.p2, dve.p3, dve.p4, dve.p1])
            ax.plot(points[:, 0], points[:, 1], points[:, 2], 'b-', linewidth=0.7, alpha=0.5)
            
            cp = dve.control_point
            ax.text(cp[0], cp[1], cp[2], str(i), color='black', fontsize=8, ha='center', va='center')

        # --- NOVO: Destaca as arestas da Raiz e da Ponta ---

        # Itera sobre os DVEs ao longo da corda para desenhar a aresta da raiz
        for i in range(m):
            root_dve = self.dves[i]
            root_edge = np.array([root_dve.p1, root_dve.p4])
            # Apenas a primeira aresta terá a legenda para evitar duplicatas
            label = "Raiz (Root)" if i == 0 else ""
            ax.plot(root_edge[:, 0], root_edge[:, 1], root_edge[:, 2], 
                    'g-', linewidth=3, label=label)

        # Itera sobre os DVEs ao longo da corda para desenhar a aresta da ponta
        for i in range(m):
            tip_dve_index = (n_panels_span - 1) * m + i
            tip_dve = self.dves[tip_dve_index]
            tip_edge = np.array([tip_dve.p2, tip_dve.p3])
            label = "Ponta (Tip)" if i == 0 else ""
            ax.plot(tip_edge[:, 0], tip_edge[:, 1], tip_edge[:, 2], 
                    'r-', linewidth=3, label=label)

        if created_fig:
            ax.set_xlabel("X (Eixo de Rotação)")
            ax.set_ylabel("Y (Envergadura)")
            ax.set_zlabel("Z")
            ax.set_title("Malha de DVEs com Índices, Raiz e Ponta")
            ax.legend()
            ax.set_box_aspect((1, 1, 1))
            plt.show()

    def plot_blade(self, ax, color='lightblue', edgecolor='gray', alpha=0.95, rstride=1, cstride=1):
        """
        Plota a superfície 3D desta pá, com base na geometria atual dos seus DVEs.
        (Versão corrigida que funciona com a rotação do rotor)
        """
        if not hasattr(self, 'dves') or not self.dves:
            print("Aviso: A pá ainda não foi discretizada em DVEs. Nada para plotar.")
            return
        
        # --- ADICIONE ESTE BLOCO DE DEBUG ---
        # Pega o primeiro DVE da pá como amostra para inspeção
        dve_para_inspecionar = self.dves[0]
        print("Coordenadas dos 4 cantos do primeiro DVE no momento do plot:")
        print(f"  p1: {np.round(dve_para_inspecionar.p1, 3)}")
        print(f"  p2: {np.round(dve_para_inspecionar.p2, 3)}")
        print(f"  p3: {np.round(dve_para_inspecionar.p3, 3)}")
        print(f"  p4: {np.round(dve_para_inspecionar.p4, 3)}")
        # --- FIM DO BLOCO DE DEBUG ---

        # Cria as matrizes (meshgrid) para as coordenadas da superfície
        X = np.zeros((self.n_span, self.m_chord + 1))
        Y = np.zeros((self.n_span, self.m_chord + 1))
        Z = np.zeros((self.n_span, self.m_chord + 1))

        # Preenche as primeiras (n_span - 1) linhas da malha de pontos
        for j in range(self.n_span - 1):      # Itera sobre as fileiras de DVEs
            for i in range(self.m_chord):  # Itera sobre os DVEs na fileira
                dve_index = j * self.m_chord + i
                dve = self.dves[dve_index]
                
                # A linha j da malha de pontos é formada pelos pontos p1 dos DVEs
                X[j, i] = dve.p1[0]
                Y[j, i] = dve.p1[1]
                Z[j, i] = dve.p1[2]

            # O último ponto da linha j é o ponto p2 do último DVE da fileira
            last_dve_in_row = self.dves[(j + 1) * self.m_chord - 1]
            X[j, self.m_chord] = last_dve_in_row.p2[0]
            Y[j, self.m_chord] = last_dve_in_row.p2[1]
            Z[j, self.m_chord] = last_dve_in_row.p2[2]

        # Preenche a última linha de pontos da malha (a ponta da pá)
        # Ela é formada pelos pontos p4 e p3 da última fileira de DVEs
        last_row_start_index = (self.n_span - 2) * self.m_chord
        for i in range(self.m_chord):
            dve_index = last_row_start_index + i
            dve = self.dves[dve_index]
            X[self.n_span - 1, i] = dve.p4[0]
            Y[self.n_span - 1, i] = dve.p4[1]
            Z[self.n_span - 1, i] = dve.p4[2]
            
        last_dve_in_blade = self.dves[-1]
        X[self.n_span - 1, self.m_chord] = last_dve_in_blade.p3[0]
        Y[self.n_span - 1, self.m_chord] = last_dve_in_blade.p3[1]
        Z[self.n_span - 1, self.m_chord] = last_dve_in_blade.p3[2]

        # Agora, plota a superfície usando a malha reconstruída
        ax.plot_surface(X, Y, Z, color=color, edgecolor=edgecolor, alpha=alpha, rstride=rstride, cstride=cstride)

    def discretize2DVE(self, m, n):
        """
        Discretiza a pá em uma malha de DVEs (Versão Robusta).
        Gera uma grelha de pontos 3D e depois conecta-os para formar os painéis.
        m: número de painéis na direção da corda (chordwise)
        n: número de estações na direção da envergadura (spanwise)
        """
        self.m_chord = m
        self.n_span = n

        # --- Passo 1: Criar interpoladores para as propriedades da pá ---
        r_sections = [s.r for s in self.sections]
        chord_interp = interp1d(r_sections, [s.chord for s in self.sections], kind='linear')
        # GARANTIR QUE A INTERPOLAÇÃO SEJA FEITA EM GRAUS
        twist_interp = interp1d(r_sections, [s.twist for s in self.sections], kind='linear')

        # --- Passo 2: Gerar a grelha de pontos 3D ---
        span_stations = np.linspace(r_sections[0], r_sections[-1], n)
        chord_fractions = np.linspace(0, 1, m + 1) # m painéis requerem m+1 pontos

        grid_points = np.zeros((n, m + 1, 3)) # Formato (span, chord, xyz)

        for j, r in enumerate(span_stations):
            chord = chord_interp(r)
            twist_deg = twist_interp(r)
            twist_rad = np.deg2rad(twist_deg)

            for i, c_frac in enumerate(chord_fractions):
                # Coordenadas 2D não rotacionadas (placa plana no plano XY)
                # O eixo de torção está no eixo Y
                # A corda estende-se ao longo do eixo X negativo, a partir de x=0
                x_unrotated = -c_frac * chord
                y_unrotated = r
                z_unrotated = 0.0

                # Aplica a rotação de torção (pitch) em torno do eixo Y
                # Nota: para um pitch-up (twist > 0), o bordo de fuga (x<0) deve subir (z>0)
                # Rotação padrão: x' = x*cos - z*sin; z' = x*sin + z*cos
                # Como z_unrotated=0, simplifica-se:
                x_rotated = x_unrotated * np.cos(twist_rad)
                z_rotated = x_unrotated * np.sin(twist_rad)
                
                grid_points[j, i, :] = [x_rotated, y_unrotated, z_rotated]

        # --- Passo 3: Construir os DVEs a partir da grelha de pontos ---
        dves = []
        for j in range(n - 1):      # Itera sobre as fileiras da envergadura
            for i in range(m):      # Itera sobre as colunas da corda
                # Pontos seguindo a sua nova definição
                p1 = grid_points[j, i]          # LE, inboard
                p2 = grid_points[j + 1, i]      # LE, outboard
                p3 = grid_points[j + 1, i + 1]  # TE, outboard
                p4 = grid_points[j, i + 1]      # TE, inboard
                
                dve = DVE(p1, p2, p3, p4)
                dves.append(dve)
                
        self.dves = dves

    def plotDVE(self, figsize=(10, 6), elev=25, azim=-120, plot_normal = False, k_normal = 0.1):
        """
        Plota os DVEs em 3D:
        - Arestas em linhas vermelhas tracejadas
        - Ponto de controle como ponto vermelho
        - Vetor normal como seta azul
        """
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')

        for dve in self.dves:
            # Coordenadas das arestas do painel
            x = [dve.p1[0], dve.p2[0], dve.p3[0], dve.p4[0], dve.p1[0]]
            y = [dve.p1[1], dve.p2[1], dve.p3[1], dve.p4[1], dve.p1[1]]
            z = [dve.p1[2], dve.p2[2], dve.p3[2], dve.p4[2], dve.p1[2]]

            # Desenha as arestas do painel
            ax.plot(x, y, z, 'r--', linewidth=0.8)

            # Ponto de controle (centro do DVE)
            cx, cy, cz = dve.control_point
            ax.scatter(cx, cy, cz, color='red', s=10)

            # Vetor normal (azul)
            # Ajustar escala com base na dimensão local do painel
            if plot_normal:
                edge_length = k_normal* (np.linalg.norm(dve.p2 - dve.p1) + np.linalg.norm(dve.p4 - dve.p1))
                normal_vector = dve.normal * edge_length * 0.5  # ajustar escala

                ax.quiver(
                    cx, cy, cz,  # origem
                    normal_vector[0], normal_vector[1], normal_vector[2],  # direção
                    color='blue', linewidth=1.2, arrow_length_ratio=0.2
                )

        # Ajustes de visualização
        ax.set_xlabel("X (Chord)")
        ax.set_ylabel("Y (Span)")
        ax.set_zlabel("Z")
        ax.set_title("Representação 3D dos DVEs")
        ax.view_init(elev=elev, azim=azim)
        ax.grid(True)
        ax.set_box_aspect([1, 2, 0.5])  # aspecto ajustado
        plt.tight_layout()
        plt.show()

    def get_trailing_edge_dves(self):
        """
        Retorna uma lista dos DVEs localizados na borda de fuga da pá.
        """
        if not hasattr(self, 'dves') or not self.dves:
            return []

        trailing_edge_dves = []
        # Os DVEs são armazenados em ordem: primeiro a corda, depois a envergadura.
        # Para uma fileira j (na envergadura) e uma coluna i (na corda), o índice é j*m + i.
        # Os DVEs da borda de fuga são aqueles na última coluna da malha, ou seja, i = m_chord - 1.
        
        for j in range(self.n_span - 1): # Itera sobre cada fileira ao longo da envergadura
            # O índice do DVE na borda de fuga para a fileira j é (j * m_chord) + (m_chord - 1)
            # O que pode ser simplificado para (j + 1) * m_chord - 1
            te_dve_index = (j + 1) * self.m_chord - 1
            trailing_edge_dves.append(self.dves[te_dve_index])
            
        return trailing_edge_dves

    def __calculate_kse(self):
        """
        Calcula o coeficiente de suavização da borda lateral (kse) para a pá.

        Baseado na recomendação da tese: 1% da envergadura do menor DVE de superfície.
        Este valor é constante para todos os DVEs da pá.
        """
        if not self.dves:
            print("Aviso: Nenhum DVE foi gerado. Retornando kse = 0.")
            return 0.0

        # Encontra a menor envergadura entre todos os DVEs da pá
        min_span = min([dve.half_span * 2 for dve in self.dves])

        # Calcula kse e o armazena como um atributo da geometria
        self.kse = 0.01 * min_span
        
        print(f"kse calculado para a pá: {self.kse:.6f} (baseado na menor envergadura de {min_span:.4f} m)")

class DVE:
    def __init__(self, p1, p2, p3, p4):
        self.p1 = p1  # canto LE inboard
        self.p2 = p2  # canto TE inboard
        self.p3 = p3  # canto TE outboard
        self.p4 = p4  # canto LE outboard

        self.control_point = self.compute_control_point()
        self.normal = self.compute_normal()
        self.half_chord = self.compute_half_chord()
        self.half_span = self.compute_half_span()
        self.compute_sweep_angles()
        self.gamma_coeffs = np.zeros(3)
    
    def compute_control_point(self):
        # Ponto médio LE
        self.mid_le = 0.5*(self.p1 + self.p4)

        # Ponto médio TE
        self.mid_te = 0.5*(self.p2 + self.p3)

        # Ponto de controle em 75%
        #return self.mid_le + 0.75*(self.mid_te - self.mid_le)
        return (self.p1 + self.p2 + self.p3 + self.p4) / 4.0

    def compute_normal(self):
        # Vetor normal estimado com produto vetorial das diagonais
        self.compute_local_frame()
        return self.e_zeta
    
    def compute_local_frame(self):
        """
        Define e armazena os vetores ξ, η, ζ do painel.
        """
        v1 = self.p2 - self.p1
        v2 = self.p4 - self.p1
        self.e_eta = v1 / np.linalg.norm(v1)
        self.e_zeta = np.cross(v2, v1)
        self.e_zeta /= np.linalg.norm(self.e_zeta)
        self.e_xi = np.cross(self.e_eta, self.e_zeta)

        self.local_axes = np.vstack((self.e_xi, self.e_eta, self.e_zeta))  # 3x3

    def local2global(self, x_local):
        x_global = self.local_axes.T @ x_local + self.control_point
        return x_global
    
    def global2local(self, x_global, frame='leading_edge'):
        """
        Converte um ponto global para o sistema de coordenadas local do DVE.
        
        Args:
            x_global (np.array): O ponto em coordenadas globais.
            frame (str): A origem do sistema de coordenadas local a ser usada.
                         Opções: 'leading_edge', 'trailing_edge', 'control_point'.
                         Padrão é 'leading_edge' pois é fundamental para induce_velocity.
        """
        if frame == 'leading_edge':
            origin = self.mid_le
        elif frame == 'trailing_edge':
            origin = self.mid_te
        elif frame == 'control_point':
            origin = self.control_point
        else:
            raise ValueError(f"Argumento 'frame' inválido: '{frame}'. Use 'leading_edge', 'trailing_edge' ou 'control_point'.")

        # Vetor relativo à origem escolhida
        rel_vector = x_global - origin
        
        # Projeção do vetor relativo nos eixos locais para obter as coordenadas locais
        x_local = self.local_axes @ rel_vector
        
        return x_local

    def local2global_vector(self, v_local):
        """Converte um VETOR do sistema local para o global (aplica apenas rotação)."""
        v_global = self.local_axes.T @ v_local
        return v_global

    def compute_sweep_angles(self):
        """
        Calcula os ângulos de enflechamento (sweep angles) da borda de ataque (LE)
        e da borda de fuga (TE) em radianos.
        
        O ângulo de enflechamento é definido como o ângulo entre a aresta 
        e o eixo local eta (envergadura).
        """
        # É necessário que self.local_axes tenha sido computado antes!
        if not hasattr(self, "local_axes"):
            raise ValueError("local_axes devem ser definidos antes de calcular os sweep angles")

        # Vetores das arestas no sistema de coordenadas global
        v_le_global = self.p4 - self.p1
        v_te_global = self.p3 - self.p2

        # Projeta os vetores das arestas nos eixos locais xi e eta
        v_le_xi_comp = np.dot(v_le_global, self.e_xi)
        v_le_eta_comp = np.dot(v_le_global, self.e_eta)
        
        v_te_xi_comp = np.dot(v_te_global, self.e_xi)
        v_te_eta_comp = np.dot(v_te_global, self.e_eta)

        # Calcula o ângulo a partir do eixo eta (e_eta) em direção ao eixo xi (e_xi)
        # A chamada é arctan2(componente_x, componente_y) para medir a partir do eixo Y
        self.sweep_le = np.arctan2(v_le_xi_comp, v_le_eta_comp)
        self.sweep_te = np.arctan2(v_te_xi_comp, v_te_eta_comp)
        
        # Opcional: Calcular o enflechamento da corda média
        v_mc_global = 0.5 * (self.p3 + self.p4) - 0.5 * (self.p1 + self.p2)
        v_mc_xi_comp = np.dot(v_mc_global, self.e_xi)
        v_mc_eta_comp = np.dot(v_mc_global, self.e_eta)
        self.sweep_mc = np.arctan2(v_mc_xi_comp, v_mc_eta_comp)

    def compute_half_chord(self):
        """
        Calcula a meia-corda do DVE no centro da envergadura.
        """
        # Ponto médio da borda de ataque (entre p1 e p4)
        mid_le_point = 0.5 * (self.p1 + self.p4)
        
        # Ponto médio da borda de fuga (entre p2 e p3)
        mid_te_point = 0.5 * (self.p2 + self.p3)
        
        # Vetor da corda no centro da envergadura
        chord_vector = mid_te_point - mid_le_point
        
        return 0.5 * np.linalg.norm(chord_vector)

    def compute_half_span(self):
        """
        Calcula a meia-envergadura do DVE.
        """
        # Ponto médio da aresta inboard (entre p1 e p2)
        mid_inboard_point = 0.5 * (self.p1 + self.p2)

        # Ponto médio da aresta outboard (entre p4 e p3)
        mid_outboard_point = 0.5 * (self.p4 + self.p3)

        # Vetor da envergadura
        span_vector = mid_outboard_point - mid_inboard_point

        return 0.5 * np.linalg.norm(span_vector)

    def control_points_spanwise(self):
        """
        Retorna 3 pontos de controle ao longo da linha de corda central
        em η = -h/2, 0, +h/2 (em coordenadas globais).
        """
        points = []
        for eta in [-self.half_span / 2, 0.0, self.half_span / 2]:
            x_local = np.array([0.0, eta, 0.0])  # ξ=0, ζ=0 → corda média
            x_global = self.p1 + self.local_axes.T @ x_local
            points.append(x_global)
        return points
        
    def induced_velocity(self, x_global, kse, kle):
        # Coeficientes da circulação no LE
        coeffs_le = self.gamma_coeffs

        # Coeficientes da circulação no TE (sinal invertido)
        coeffs_te = -self.gamma_coeffs

        # ---- Borda de Ataque (Leading Edge) -----
        # Transforma o ponto global para o sistema de coordenadas local da LE
        local_le = self.global2local(x_global, frame='leading_edge')
        v_filament_le = self.__compute_filament_velocity(local_le, coeffs_le, self.sweep_le)
        v_sheet_le = self.__compute_sheet_velocity(local_le, coeffs_le, self.sweep_le, kse, kle)

        # ---- Borda de Fuga (Trailing Edge) -----
        # Transforma o ponto global para o sistema de coordenadas local da TE
        local_te = self.global2local(x_global, frame='trailing_edge')
        v_filament_te = self.__compute_filament_velocity(local_te, coeffs_te, self.sweep_te)
        v_sheet_te = self.__compute_sheet_velocity(local_te, coeffs_te, self.sweep_te, kse, kle)

        # ----- Soma as contribuições em coordenadas locais ----
        v_total_local = v_filament_le + v_sheet_le + v_filament_te + v_sheet_te

        # ----- Transforma o vetor de velocidade de volta para o sistema global ----
        v_total_global = self.local2global_vector(v_total_local)

        return v_total_global

    def _induced_velocity_by_filament(self, p1, p2, p, kse=1e-8):
        """
        Calcula a velocidade induzida por um filamento de vórtice reto de força unitária.
        (Implementação da Lei de Biot-Savart)

        Args:
            p1 (np.array): Ponto inicial do filamento.
            p2 (np.array): Ponto final do filamento.
            p (np.array): Ponto onde a velocidade é calculada.
            kse (float): Termo de suavização para evitar singularidades (vortex core).

        Returns:
            np.array: Vetor de velocidade induzida.
        """
        r1 = p - p1
        r2 = p - p2
        r1_norm = np.linalg.norm(r1)
        r2_norm = np.linalg.norm(r2)
        
        # Vetor do filamento
        r0 = p2 - p1
        r0_norm = np.linalg.norm(r0)
        
        # Produto vetorial entre os vetores de distância
        r1_cross_r2 = np.cross(r1, r2)
        r1_cross_r2_norm_sq = np.dot(r1_cross_r2, r1_cross_r2)
        
        # Evita divisão por zero se o ponto p estiver sobre o filamento
        if r1_cross_r2_norm_sq < kse:
            return np.zeros(3)
            
        # Termo principal da equação de Biot-Savart
        term1 = (r1_norm + r2_norm) / (r1_norm * r2_norm * (r1_norm * r2_norm + np.dot(r1, r2)))
        
        # Velocidade final (sem o Gamma / 4*pi)
        vel = (r1_cross_r2 / (4 * np.pi * r1_cross_r2_norm_sq)) * (np.dot(r0, r1) - np.dot(r0, r2))
        
        return vel
    
    def get_induced_velocity_by_coeff(self, p, coeff_idx):
        """
        Calcula o vetor de velocidade de influência em um ponto 'p',
        assumindo que o coeficiente de circulação solicitado é 1 e os outros são 0.

        Args:
            p (np.array): Ponto onde a velocidade é calculada.
            coeff_idx (int): Índice do coeficiente (0 para A, 1 para B, 2 para C).

        Returns:
            np.array: Vetor de velocidade de influência.
        """
        # Velocidades induzidas pelas 4 arestas do DVE (vórtice em anel)
        # A direção segue a regra da mão direita para um vórtice positivo (para cima)
        v_le = self._induced_velocity_by_filament(self.p1, self.p4, p) # Bordo de ataque
        v_te = self._induced_velocity_by_filament(self.p3, self.p2, p) # Bordo de fuga
        v_inboard = self._induced_velocity_by_filament(self.p2, self.p1, p) # Lado interno
        v_outboard = self._induced_velocity_by_filament(self.p4, self.p3, p) # Lado externo

        if coeff_idx == 0:
            # Coeficiente A: Circulação constante Gamma = 1.
            # A influência é a de um anel de vórtice de força unitária.
            return v_le + v_te + v_inboard + v_outboard
        
        elif coeff_idx == 1:
            # Coeficiente B: Circulação linear Gamma = eta.
            # A influência vem da variação de circulação nas laterais.
            # A vorticidade emitida é d(Gamma)/d(eta) = 1.
            eta = self.half_span
            return (v_outboard - v_inboard) * eta
            
        elif coeff_idx == 2:
            # Coeficiente C: Circulação quadrática Gamma = eta^2.
            # A influência também vem da variação nas laterais.
            # A vorticidade emitida é d(Gamma)/d(eta) = 2*eta.
            eta = self.half_span
            return (v_outboard + v_inboard) * 2 * (eta**2)
            
        else:
            raise ValueError("coeff_idx deve ser 0, 1 ou 2.")
    
    def __compute_filament_velocity(self, point, gamma_coeffs, sweep_angle):
        """
        Calcula a velocidade induzida por um único filamento de vórtice.

        Args:
            local_point (np.array): Ponto de destino em coordenadas locais.
            coeffs (np.array): Coeficientes de circulação [A, B, C].
            sweep_angle (float): Ângulo de enflechamento (Lambda) do filamento em radianos.
        """
        A, B, C = gamma_coeffs
        xi, eta, zeta = point

        tan_lambda = np.tan(sweep_angle)

        # ----- A.4 -----        
        a1 = 1 + tan_lambda**2
        b1 = -(eta + xi*tan_lambda)
        c1 = xi**2 + eta**2 + zeta**2
        
        # ----- A.3 -----
        G11 = self.__calculate_G11(a1, b1, c1, self.half_span) - self.__calculate_G11(a1, b1, c1, -self.half_span)
        G12 = self.__calculate_G12(a1, b1, c1, self.half_span) - self.__calculate_G12(a1, b1, c1, -self.half_span)
        G13 = self.__calculate_G13(a1, b1, c1, self.half_span) - self.__calculate_G13(a1, b1, c1, -self.half_span)

        # ----- A.2 ----
        a1xi   = -G11*zeta
        a1eta  = G11*zeta*tan_lambda
        a1zeta = G11*(xi - eta*tan_lambda)
        b1xi   = -G12*zeta
        b1eta  = G12*zeta*tan_lambda
        b1zeta = G12*(xi - eta*tan_lambda)
        c1xi   = -G13*zeta
        c1eta  = G13*zeta*tan_lambda
        c1zeta = G13*(xi - eta*tan_lambda)

        # ----- A.1 -----
        M = np.array([[a1xi, a1eta, a1zeta], [b1xi, b1eta, b1zeta], [c1xi, c1eta, c1zeta]]).T

        return (-1/4*np.pi)*M@gamma_coeffs.T

    def __compute_sheet_velocity(self, point, gamma_coeffs, sweep_angle, kse, kle):
        A, B, C = gamma_coeffs
        xi, eta, zeta = point
        etai = self.half_span

        tan_lambda = np.tan(sweep_angle)

        # ----- A.7 ------
        t1 = eta + etai
        t2 = eta - etai

        # ----- A.8 ------
        a2 = 1 + tan_lambda**2
        b2 = (xi + eta*tan_lambda)*tan_lambda
        c2 = (xi + eta*tan_lambda)**2 + zeta**2

        # ----- A.9 ------
        epsilon = self.__calculate_epsilon(xi, eta, zeta, tan_lambda)
        rho = self.__calculate_rho(epsilon, zeta, b2)
        beta1, beta2 = self.__calculate_betas(epsilon, rho, zeta, b2)
        gamma1 = self.__calculate_gamma1(rho, a2, b2, beta1, beta2, zeta)
        gamma2 = self.__calculate_gamma2(rho, a2, b2, beta1, beta2, zeta)
        delta1 = self.__calculate_delta1(rho, b2, c2, beta1, beta2, zeta)
        delta2 = self.__calculate_delta2(rho, b2, c2, beta1, beta2, zeta)
        G21 = self.__calculate_G21(rho, beta1, beta2, t2, gamma1, gamma2, delta1, delta2, zeta, a2, b2, c2, kse) - self.__calculate_G21(rho, beta1, beta2, t1, gamma1, gamma2, delta1, delta2, zeta, a2, b2, c2, kse)
        G22 = self.__calculate_G22(rho, beta1, beta2, t2, gamma1, gamma2, delta1, delta2, zeta, a2, b2, c2, kse) - self.__calculate_G22(rho, beta1, beta2, t1, gamma1, gamma2, delta1, delta2, zeta, a2, b2, c2, kse)

        # ----- A.11 -----
        G23 = self.__calculate_G23(a2, b2, c2, t2, kle) - self.__calculate_G23(a2, b2, c2, t1, kle)
        G24 = self.__calculate_G24(a2, b2, c2, t2, kle) - self.__calculate_G24(a2, b2, c2, t1, kle)

        # ----- A.13 -----
        G25 = self.__calculate_G25(zeta, t2, kse) - self.__calculate_G25(zeta, t1, kse)
        G26 = self.__calculate_G26(zeta, t2) - self.__calculate_G26(zeta, t1)
        G27 = self.__calculate_G27(t2) - self.__calculate_G27(t1) 


        # ----- A.14 -----
        b21 = -xi + eta*tan_lambda
        b22 = zeta**2*tan_lambda
        b24 = -tan_lambda
        b23 = 0; b25 = -1; b26 = 0; b27 = 0
        c21 = -2*(zeta**2*tan_lambda + eta*(xi - eta*tan_lambda))
        c22 = -2*zeta**2*(xi - 2*eta*tan_lambda)
        c23 = 2*tan_lambda
        c24 = 2*(xi - 2*eta*tan_lambda)
        c25 = -2*eta
        c26 = -2*zeta**2
        c27 = 2

        # ----- A.6 -----
        b2eta = -zeta*(G21*b24 + G22*b21 + G26*b25)
        c2eta = -zeta*(G21*c24 + G22*c21 + G24*c23 + G25*c27 + G26*c25)
        b2zeta = G21*b21 + G22*b22 + G23*b23 + G24*b24 + G25*b25 + G26*b26 + G27*b27
        c2zeta = G21*c21 + G22*c22 + G23*c23 + G24*c24 + G25*c25 + G26*c26 + G27*c27
        
        M = np.array([[0, 0, 0], [0, b2eta, c2eta], [0, b2zeta, c2zeta]])

        return (-1/(4*np.pi))*M@gamma_coeffs.T
        
    def calculate_kle(self):
        """
        Calcula o coeficiente de suavização da borda de ataque (kle).

        Como a tese não especifica um valor, usamos uma estimativa razoável:
        1% da corda local do DVE.
        """
        # A corda total do DVE no centro da envergadura
        full_chord = self.half_chord * 2
        
        # kle é 1% da corda local
        kle = 0.01 * full_chord
        
        return kle    

    def __calculate_epsilon(self, xi, eta, zeta, tan_lambda):
        return (xi - eta*tan_lambda)**2 - (zeta*tan_lambda)**2

    def __calculate_rho(self, epsilon, zeta, b):
        return np.sqrt(epsilon**2 + 4*(zeta*b)**2)

    def __calculate_betas(self, epsilon, rho, zeta, b):
        target_product = zeta*b

        # Verificar se os termos dentro das raízes são positivos
        term1 = (rho + epsilon) / 2.0
        term2 = (rho - epsilon) / 2.0

        if term1 < 0 or term2 < 0:
            # Esta situação pode ocorrer e precisa ser tratada.
            # Por enquanto, vamos retornar zero ou lançar um erro.
            # Em uma implementação real, pode ser necessário usar cmath aqui se isso acontecer.
            # Mas seguindo a lógica do documento como estritamente real:
            print(f"Aviso: Termo negativo dentro da raiz quadrada ao calcular betas.")
            return 0.0, 0.0

        # Calcular as magnitudes (valores absolutos)
        beta1_mag = np.sqrt(term1)
        beta2_mag = np.sqrt(term2)

        # Testar as duas possibilidades de produto: positivo ou negativo
        product_mag = beta1_mag * beta2_mag

        # Se o produto alvo e o produto das magnitudes tiverem o mesmo sinal
        if (target_product >= 0 and product_mag >= 0) or \
        (target_product < 0 and product_mag < 0):
            # Os sinais de beta1 e beta2 devem ser iguais.
            # Por convenção, podemos escolher (+, +). O par (-, -) dá o mesmo produto.
            beta1 = beta1_mag
            beta2 = beta2_mag
        else:
            # Os sinais de beta1 e beta2 devem ser opostos.
            # Por convenção, podemos escolher (+, -).
            beta1 = beta1_mag
            beta2 = -beta2_mag

        # Verificação final (opcional, para depuração)
        # if not math.isclose(beta1 * beta2, target_product, rel_tol=1e-9):
        #     print(f"Aviso: A condição do produto beta não foi satisfeita com precisão.")

        return beta1, beta2
    
    def __calculate_gamma1(self, rho, a2, b2, beta1, beta2, zeta):
        # Adicionar esta verificação de singularidade
        if abs(rho) < 1e-9:
            return 0.0
        
        return (1/rho)*(a2*beta2*zeta + b2*beta1)

    def __calculate_gamma2(self, rho, a2, b2, beta1, beta2, zeta):
        # Adicionar a mesma verificação
        if abs(rho) < 1e-9:
            return 0.0
            
        return (1/rho)*(a2*beta1*zeta - b2*beta2)

    def __calculate_delta1(self, rho, b2, c2, beta1, beta2, zeta):
        # Adicionar a mesma verificação
        if abs(rho) < 1e-9:
            return 0.0
            
        return (1/rho)*(b2*beta2*zeta + c2*beta1)

    def __calculate_delta2(self, rho, b2, c2, beta1, beta2, zeta):
        # Adicionar a mesma verificação
        if abs(rho) < 1e-9:
            return 0.0
            
        return (1/rho)*(b2*beta1*zeta - c2*beta2)
    
    def __calculate_gamma1_old(self, rho, a2, b2, beta1, beta2, zeta):
        return (1/rho)*(a2*beta2*zeta + b2*beta1)
    
    def __calculate_gamma2_old(self, rho, a2, b2, beta1, beta2, zeta):
        return (1/rho)*(a2*beta1*zeta - b2*beta2)
    
    def __calculate_delta1_old(self, rho, b2, c2, beta1, beta2, zeta):
        return (1/rho)*(b2*beta2*zeta + c2*beta1)
    
    def __calculate_delta2_old(self, rho, b2, c2, beta1, beta2, zeta):
        return (1/rho)*(b2*beta1*zeta - c2*beta2)
    
    def __calculate_mu1(self, gamma1, gamma2, t, delta1, delta2, zeta, a, b, c, kse):
        return ((gamma1*t + delta1 - self.__calculate_r_t(a, b, c, t))**2 + (gamma2*t + delta2)**2)/(t**2 + zeta**2 + kse)
    
    def __calculate_mu2(self, gamma1, gamma2, t, delta1, delta2, zeta, a, b, c):
        return np.atan(zeta/t) + np.atan((gamma2*t + delta2)/(gamma1*t+delta1 - self.__calculate_r_t(a, b, c, t)))
    
    def __calculate_r_eta(self, a, b, c, eta):
        return np.sqrt(a*eta**2 + 2*b*eta + c)
    
    def __calculate_r_t(self, a, b, c, t):
        return np.sqrt(t**2*a + 2*t*b + c)
    
    def __calculate_mu3(self, a, b, c, t, kle):
        return a*t + b + np.sqrt(a)*self.__calculate_r_t(a, b, c, t) + kle

    def __calculate_G11(self, a, b, c, eta):
        # O denominador da fórmula
        discriminant = a*c - b**2
        r_eta = self.__calculate_r_eta(a, b, c, eta)
        denominator = discriminant * r_eta

        # --- Adicionar esta verificação de singularidade ---
        if abs(denominator) < 1e-9:
            # Esta condição ocorre quando o ponto de avaliação está na linha do filamento.
            # A contribuição da velocidade induzida nesta situação é zero.
            return 0.0
        # --- Fim da verificação ---

        return (a * eta + b) / denominator

    def __calculate_G12(self, a, b, c, eta):
        # O denominador é o mesmo de G11
        discriminant = a*c - b**2
        r_eta = self.__calculate_r_eta(a, b, c, eta)
        denominator = discriminant * r_eta

        # --- Adicionar a mesma verificação ---
        if abs(denominator) < 1e-9:
            return 0.0
        # --- Fim da verificação ---

        return -(b * eta + c) / denominator # Nota: A fórmula no apêndice tem um sinal negativo

    def __calculate_G13(self, a, b, c, eta):
        # Esta função é mais complexa, mas a primeira parte tem o mesmo denominador
        discriminant = a*c - b**2
        r_eta = self.__calculate_r_eta(a, b, c, eta)
        denominator1 = discriminant * a * r_eta

        if abs(denominator1) < 1e-9:
            term1 = 0.0
        else:
            term1 = ((2*b**2 - a*c)*eta + b*c) / denominator1
        
        # ... O segundo termo com o logaritmo geralmente não é problemático ...
        # mas uma verificação de segurança no argumento do log pode ser adicionada
        log_arg = np.sqrt(a)*r_eta + a*eta + b
        if log_arg <= 0:
            term2 = 0.0
        else:
            term2 = (1 / np.sqrt(a**3)) * np.log(log_arg)
            
        return term1 + term2

    def __calculate_G11_old(self, a, b, c, eta):
        return (a*eta + b)/((a*c - b**2)*self.__calculate_r_eta(a, b, c, eta))
    
    def __calculate_G12_old(self, a, b, c, eta):
        return -(b*eta + c)/((a*c - b**2)*self.__calculate_r_eta(a, b, c, eta))
    
    def __calculate_G13_old(self, a, b, c, eta):
        return ((2*b**2 - a*c)*eta + b*c)/((a*c - b**2)*a*self.__calculate_r_eta(a, b, c, eta)) + (1/np.sqrt(a**3))*np.log(np.sqrt(a)*self.__calculate_r_eta(a, b, c, eta) + a*eta + b)

    
    def __calculate_G21(self, rho, beta1, beta2, t, gamma1, gamma2, delta1, delta2, zeta, a, b, c, kse):
        # --- Adicionar esta verificação de singularidade no início ---
        # Se rho é zero, beta1 também será zero, resultando em 0/0.
        # A contribuição total da integral neste caso singular é 0.
        if abs(rho) < 1e-9:
            return 0.0
        # --- Fim da verificação ---

        # O resto do código agora está seguro contra a divisão por rho.
        
        # Termo 1
        mu1 = self.__calculate_mu1(gamma1, gamma2, t, delta1, delta2, zeta, a, b, c, kse)
        if mu1 <= 0: # Segurança para o logaritmo
            term1 = 0.0
        else:
            term1 = (beta1 / (2 * rho)) * np.log(mu1)

        # Termo 2 (ainda precisa da verificação de zeta)
        if abs(zeta) < 1e-9:
            term2 = 0.0
        else:
            mu2 = self.__calculate_mu2(gamma1, gamma2, t, delta1, delta2, zeta, a, b, c)
            term2 = (beta2 / (rho * zeta)) * mu2
                
        return term1 + term2

    def __calculate_G22(self, rho, beta1, beta2, t, gamma1, gamma2, delta1, delta2, zeta, a, b, c, kse):
        # --- Adicionar a mesma verificação de singularidade no início ---
        if abs(rho) < 1e-9:
            return 0.0
        # --- Fim da verificação ---

        # O resto do código agora está seguro.
        
        if abs(zeta) < 1e-9:
            # Se zeta é zero, ambos os termos são zero
            return 0.0
        else:
            mu1 = self.__calculate_mu1(gamma1, gamma2, t, delta1, delta2, zeta, a, b, c, kse)
            if mu1 <= 0:
                term1_log = 0.0
            else:
                term1_log = np.log(mu1)
            term1 = (-beta2 / (2 * rho * zeta)) * term1_log

            mu2 = self.__calculate_mu2(gamma1, gamma2, t, delta1, delta2, zeta, a, b, c)
            term2 = (beta1 / (rho * zeta)) * mu2
            
            return term1 + term2
    
    def __calculate_G23(self, a, b, c, t, kle):
        mu3 = self.__calculate_mu3(a, b, c, t, kle)
        
        # --- Adicionar esta verificação de segurança ---
        if mu3 <= 1e-14: # Usar uma pequena tolerância em vez de <= 0
            return 0.0
        # --- Fim da verificação ---
        
        # O cálculo agora é seguro
        log_mu3 = np.log(mu3)
        r_t = self.__calculate_r_t(a, b, c, t)
        
        return (r_t / a - (b / np.sqrt(a**3)) * log_mu3)
     
    def __calculate_G24(self, a, b, c, t, kle):
        mu3 = self.__calculate_mu3(a, b, c, t, kle)

        # --- Adicionar a mesma verificação de segurança ---
        if mu3 <= 1e-14:
            return 0.0
        # --- Fim da verificação ---

        # O cálculo agora é seguro
        log_mu3 = np.log(mu3)
        return (1 / np.sqrt(a)) * log_mu3
    
    def __calculate_G21_old(self, rho, beta1, beta2, t, gamma1, gamma2, delta1, delta2, zeta, a, b, c, kse):
        return (beta1/(2*rho))*np.log(self.__calculate_mu1(gamma1, gamma2, t, delta1, delta2, zeta, a, b, c, kse)) + (beta2/(rho*zeta))*self.__calculate_mu2(gamma1, gamma2, t, delta1, delta2, zeta, a, b, c)

    def __calculate_G22_old(self, rho, beta1, beta2, t, gamma1, gamma2, delta1, delta2, zeta, a, b, c, kse):
        return (-beta2/(2*rho*zeta))*np.log(self.__calculate_mu1(gamma1, gamma2, t, delta1, delta2, zeta, a, b, c, kse)) + (beta1/(rho*zeta))*self.__calculate_mu2(gamma1, gamma2, t, delta1, delta2, zeta, a, b, c)

    def __calculate_G23_old(self, a, b, c, t, kle):
        mu3 = self.__calculate_mu3(a, b, c, t, kle)
        return (self.__calculate_r_t(a, b, c, t)/a - (b/np.sqrt(a**3))*np.log(mu3))
    
    def __calculate_G24_old(self, a, b, c, t, kle):
        mu3 = self.__calculate_mu3(a, b, c, t, kle)
        return (1/np.sqrt(a))*np.log(mu3)
    
    def __calculate_G25(self, zeta, t, kse):
        return 0.5*np.log(t**2 + zeta**2 + kse)
    
    def __calculate_G26(self, zeta, t):
        """
        Calcula a integral G26, com tratamento para a singularidade em zeta=0.
        """
        # Adicionar a verificação para o caso singular zeta = 0
        if abs(zeta) < 1e-9:
            return 0.0  # O limite da expressão quando zeta -> 0 é zero.
        
        # Fórmula corrigida da resposta anterior
        return (1 / zeta) * np.arctan(t / zeta)
    
    def __calculate_G26_old(self, zeta, t):
        return (1/zeta)*(np.atan(t/zeta))
    
    def __calculate_G27(self, t):
        return t

    def calculate_forces(self, all_dves, V_freestream, omega_vec, rho):
        """
        Calcula a força e o momento aerodinâmico neste DVE usando Kutta-Joukowski.
        A integral é resolvida numericamente com Quadratura de Gauss-Legendre.
        """
        # Pontos e pesos de Gauss-Legendre para n=3 (no intervalo [-1, 1])
        eta_points_norm = np.array([-0.77459667, 0.0, 0.77459667])
        weights = np.array([0.55555556, 0.88888889, 0.55555556])

        total_force = np.zeros(3)
        total_moment = np.zeros(3)

        # Contribuição de ambos os filamentos (LE e TE)
        filaments = [
            {'origin': self.mid_le, 'edge_vector': self.p4 - self.p1, 'coeffs': self.gamma_coeffs},
            {'origin': self.mid_te, 'edge_vector': self.p3 - self.p2, 'coeffs': -self.gamma_coeffs}
        ]

        for filament in filaments:
            # Vetor unitário ao longo do filamento
            s_hat = filament['edge_vector'] / np.linalg.norm(filament['edge_vector'])
            
            # Integral de Kutta-Joukowski via quadratura de Gauss-Legendre
            force_integral_sum = np.zeros(3)
            moment_integral_sum = np.zeros(3)

            for i in range(len(eta_points_norm)):
                # Mapear o ponto de quadratura do intervalo [-1, 1] para [-half_span, +half_span]
                eta_local = self.half_span * eta_points_norm[i]

                # 1. Posição GLOBAL do ponto de quadratura no filamento
                point_global = filament['origin'] + eta_local * s_hat

                # 2. Velocidade do escoamento (livre + rotação) no ponto
                r_vec = point_global
                V_rotation = np.cross(omega_vec, r_vec)
                V_onset = V_freestream + V_rotation

                # 3. Velocidade induzida por TODOS os DVEs
                V_induced = np.zeros(3)
                for other_dve in all_dves:
                    kse = 0.0 # Simplificado para o teste
                    kle = 0.0 # Simplificado para o teste
                    V_induced += other_dve.induced_velocity(point_global, kse, kle)
                
                # 4. Velocidade total no ponto
                V_total = V_onset + V_induced

                # 5. Força da circulação Γ(η) no ponto
                A, B, C = filament['coeffs']
                circulation_strength = A + B * eta_local + C * eta_local**2

                # 6. Integrando para a força: rho * (V_total x s_hat) * Gamma(eta)
                integrand_force = np.cross(V_total, s_hat) * circulation_strength
                force_integral_sum += integrand_force * weights[i]

                # 7. Integrando para o momento: r x (rho * (V_total x s_hat) * Gamma(eta))
                integrand_moment = np.cross(r_vec, integrand_force)
                moment_integral_sum += integrand_moment * weights[i]
                
            # Multiplicar pelo fator de escala da quadratura e pela densidade
            total_force += rho * force_integral_sum * self.half_span
            total_moment += rho * moment_integral_sum * self.half_span

        return total_force, total_moment

    def recalculate_geometry(self):
        """
        Recalcula todos os atributos geométricos do DVE após seus pontos
        de canto (p1, p2, p3, p4) terem sido movidos/rotacionados.
        """
        self.control_point = self.compute_control_point()
        self.normal = self.compute_normal() # compute_normal já chama compute_local_frame
        self.half_chord = self.compute_half_chord()
        self.half_span = self.compute_half_span()
        self.compute_sweep_angles()
        
class DVESolver:
    def __init__(self, rotor, wake, U_inf, omega, theta=10.0, rho=1.225):
        """
        Inicializa o solver principal da simulação.

        Args:
            rotor (Rotor): A instância do objeto Rotor.
            wake (Wake): A instância do objeto Wake (com a esteira inicial).
            U_inf (float): Velocidade do escoamento livre (vento) [m/s].
            omega (float): Velocidade de rotação [rad/s].
            theta (float): O passo angular da simulação | Discretização da esteira [graus].
            rho (float): Densidade do ar [kg/m^3].
        """
        self.rotor = rotor
        self.wake = wake
        self.U_inf = U_inf
        self.omega = omega
        self.theta = theta
        self.rho = rho

        # Parâmetros da simulação
        self.V_inf_vector = np.array([U_inf, 0.0, 0.0])
        self.omega_vector = np.array([omega, 0.0, 0.0]) # Rotação em torno do eixo X
        
        # Atributos do sistema
        self.blade_dves = self.rotor.get_all_dves()
        self.n_dves_total = len(self.blade_dves)
        self.n_unknowns = 3 * self.n_dves_total
        
        self.D = np.zeros((self.n_unknowns, self.n_unknowns))
        self.R = np.zeros(self.n_unknowns)
        self.gamma = np.zeros(self.n_unknowns)

    def assemble_system(self):
        """
        Monta a matriz de influência [D] e o vetor {R} para o rotor completo,
        considerando a malha 2D de DVEs em cada pá.
        """
        print(f"Montando sistema linear para {self.rotor.num_blades} pás ({self.n_unknowns} incógnitas)...")
        
        # --- Parte 1: Condição Cinemática (Tangência de Fluxo) ---
        # Esta parte já está correta, pois a influência é global.
        for i, dve_i in enumerate(self.blade_dves):
            r_vec = dve_i.control_point
            V_rotation = np.cross(self.omega_vector, r_vec)
            V_onset = self.V_inf_vector + V_rotation
            

            V_induced_by_wake = np.zeros(3)
            for wake_dve in self.wake.get_all_dves():
                # A circulação de um DVE da esteira é constante (só tem coeficiente A)
                # e o seu valor 'gamma' é conhecido.
                vel = wake_dve.get_induced_velocity_by_coeff(dve_i.control_point, coeff_idx=0)
                V_induced_by_wake += vel * wake_dve.gamma_coeffs[0]

            # MODIFICA o cálculo de R[i] para incluir a influência da esteira
            total_onset_velocity = V_onset + V_induced_by_wake
            self.R[i] = -np.dot(total_onset_velocity, dve_i.normal)
            ### Ignorando a esteira para este teste de validação
            ##self.R[i] = -np.dot(V_onset, dve_i.normal)

            for j, dve_j in enumerate(self.blade_dves):
                vel_A = dve_j.get_induced_velocity_by_coeff(dve_i.control_point, coeff_idx=0)
                self.D[i, 3*j + 0] = np.dot(-vel_A, dve_i.normal)
                
                vel_B = dve_j.get_induced_velocity_by_coeff(dve_i.control_point, coeff_idx=1)
                self.D[i, 3*j + 1] = np.dot(-vel_B, dve_i.normal)
                
                vel_C = dve_j.get_induced_velocity_by_coeff(dve_i.control_point, coeff_idx=2)
                self.D[i, 3*j + 2] = np.dot(-vel_C, dve_i.normal)
        
        # --- Parte 2: Condições de Contorno ---
        row_index = self.n_dves_total
        dve_offset = 0
        
        for blade in self.rotor.blades:
            m_chord = blade.m_chord
            n_span_stations = blade.n_span
            n_span_panels = n_span_stations - 1

            # APLICAR CONDIÇÕES PARA CADA COLUNA AO LONGO DA CORDA
            for i_chord in range(m_chord):
                # APLICAR CONDIÇÕES PARA CADA JUNÇÃO AO LONGO DA ENVERGADURA
                for j_span in range(n_span_panels):
                    
                    # --- Condições de Raiz e Ponta ---
                    is_root_dve = (j_span == 0)
                    is_tip_dve = (j_span == n_span_panels - 1)

                    if is_root_dve:
                        dve_idx = dve_offset + j_span * m_chord + i_chord
                        dve = self.blade_dves[dve_idx]
                        eta = dve.half_span
                        
                        # Equação da Raiz: Γ(-η) = 0
                        self.D[row_index, 3*dve_idx+0] = 1; self.D[row_index, 3*dve_idx+1] = -eta; self.D[row_index, 3*dve_idx+2] = eta**2
                        self.R[row_index] = 0.0
                        row_index += 1
                    
                    if is_tip_dve:
                        dve_idx = dve_offset + j_span * m_chord + i_chord
                        dve = self.blade_dves[dve_idx]
                        eta = dve.half_span
                        
                        # Equação da Ponta: Γ(η) = 0
                        self.D[row_index, 3*dve_idx+0] = 1; self.D[row_index, 3*dve_idx+1] = eta; self.D[row_index, 3*dve_idx+2] = eta**2
                        self.R[row_index] = 0.0
                        row_index += 1

                    # --- Condições de Continuidade (apenas para DVEs internos) ---
                    if j_span < n_span_panels - 1:
                        idx1 = dve_offset + j_span * m_chord + i_chord
                        idx2 = dve_offset + (j_span + 1) * m_chord + i_chord
                        
                        dve1 = self.blade_dves[idx1]; dve2 = self.blade_dves[idx2]
                        eta1 = dve1.half_span; eta2 = dve2.half_span

                        # Continuidade de Gamma: Γ₁(η₁) - Γ₂(-η₂) = 0
                        self.D[row_index, 3*idx1+0] = 1; self.D[row_index, 3*idx1+1] = eta1; self.D[row_index, 3*idx1+2] = eta1**2
                        self.D[row_index, 3*idx2+0] = -1; self.D[row_index, 3*idx2+1] = eta2; self.D[row_index, 3*idx2+2] = -eta2**2
                        self.R[row_index] = 0.0
                        row_index += 1

                        # Continuidade da Vorticidade: γ₁(η₁) - γ₂(-η₂) = 0
                        self.D[row_index, 3*idx1+1] = 1; self.D[row_index, 3*idx1+2] = 2 * eta1
                        self.D[row_index, 3*idx2+1] = -1; self.D[row_index, 3*idx2+2] = 2 * eta2
                        self.R[row_index] = 0.0
                        row_index += 1
        

        dve_offset += len(blade.dves)
    
    def solve(self):
        try:
            self.gamma = np.linalg.solve(self.D, self.R)
            for i, dve in enumerate(self.blade_dves):
                dve.gamma_coeffs = self.gamma[i*3:(i+1)*3]
            print("Sistema resolvido com sucesso. Coeficientes de circulação atualizados.")

        except np.linalg.LinAlgError:
            print("ERRO: A matriz de influência [D] é singular.")
            print("Isso geralmente indica um problema na definição das condições de contorno.")
            # Você pode querer parar a execução ou lidar com o erro de outra forma
            self.gamma = np.zeros(len(self.R)) # Define como zero para evitar mais erros

    def compute_performance(self):
        """
        Calcula os coeficientes aerodinâmicos totais do rotor.
        Deve ser chamado DEPOIS de self.solve().
        """
        rotor_total_force = np.zeros(3)
        rotor_total_torque = np.zeros(3)
        
        # 1. Coleta todos os DVEs que induzem velocidade (pás e esteira)
        all_inducing_dves = self.rotor.get_all_dves()
        # No futuro, adicione os DVEs da esteira:
        # all_inducing_dves.extend(self.wake.get_all_dves())

        # 2. Itera sobre os DVEs das pás para somar as forças que atuam SOBRE eles
        for dve in self.blade_dves:
            # O DVE calcula sua própria força com base na influência de todos os DVEs
            force, torque = dve.calculate_forces(
                all_dves=all_inducing_dves, 
                V_freestream=self.V_inf_vector, 
                omega_vec=self.omega_vector, 
                rho=self.rho
            )
            rotor_total_force += force
            rotor_total_torque += torque

        # 3. Extrai as componentes axiais de força (Empuxo) e momento (Torque)
        thrust = rotor_total_force[0]
        torque_axial = rotor_total_torque[0]
        
        # 4. Calcula a potência
        power = torque_axial * self.omega

        # 5. Normaliza para obter os coeficientes de desempenho
        # Pega o raio de uma das pás como referência
        rotor_radius = self.rotor.blades[0].R
        rotor_area = np.pi * rotor_radius**2
        
        # Denominadores para os coeficientes
        # Usar self.U_inf_scalar para evitar divisões por zero se U_inf for 0
        thrust_denom = 0.5 * self.rho * rotor_area * self.U_inf**2
        power_denom = 0.5 * self.rho * rotor_area * self.U_inf**3

        # Calcula os coeficientes, com proteção para evitar divisão por zero
        Ct = thrust / thrust_denom if thrust_denom > 1e-6 else 0.0
        Cp = power / power_denom if power_denom > 1e-6 else 0.0

        return Cp, Ct
    
    def plot_influence_matrix(self):
        """
        Plota a matriz de influência [D] como uma imagem para inspeção visual.
        """
        if self.D is None:
            print("A matriz [D] ainda não foi montada.")
            return
            
        plt.figure(figsize=(8, 8))
        plt.imshow(self.D, cmap='viridis', interpolation='none')
        plt.colorbar(label="Magnitude da Influência")
        plt.title("Visualização da Matriz de Influência [D]")
        plt.xlabel("Índice da Incógnita (Coluna)")
        plt.ylabel("Índice da Equação (Linha)")
        plt.show()

class Rotor:
    def __init__(self, blade, num_blades):
        self.num_blades = num_blades
        self.blades = []

        self.__create_blades(blade)

    def __create_blades(self, blade):
        azimuth_step = 2*np.pi/self.num_blades

        for i in range(self.num_blades):
            azimuth_angle = i * azimuth_step

            new_blade = copy.deepcopy(blade)

            print(f"\n--- Processando Pá {i+1} (Ângulo: {np.rad2deg(azimuth_angle):.1f}°) ---")
        
            # --- INSPECIONAR ANTES ---
            # Pega o primeiro ponto do primeiro DVE como amostra
            p1_antes = new_blade.dves[0].p1
            #print(f"Coordenadas ANTES da rotação: {np.round(p1_antes, 3)}")
            print(f"Coordenadas ANTES da rotação: p1 = {np.round(new_blade.dves[0].p1, 3)} | p2 = {np.round(new_blade.dves[0].p2, 3)}")

            
            # Gira a nova pá para sua posição inicial
            self.__rotate_blade(new_blade, azimuth_angle)
            
            # --- INSPECIONAR DEPOIS ---
            p1_depois = new_blade.dves[0].p1
            #print(f"Coordenadas DEPOIS da rotação: {np.round(p1_depois, 3)}")
            print(f"Coordenadas DEPOIS da rotação: p1 = {np.round(new_blade.dves[0].p1, 3)} | p2 = {np.round(new_blade.dves[0].p2, 3)}")
            
            self.blades.append(new_blade)

    def __rotate_blade(self, blade, angle):
        rot_matrix = np.array([
            [1, 0, 0],
            [0, np.cos(angle), -np.sin(angle)],
            [0, np.sin(angle), np.cos(angle)]
        ])

        for dve in blade.dves:
            dve.p1 = rot_matrix @ dve.p1
            dve.p2 = rot_matrix @ dve.p2
            dve.p3 = rot_matrix @ dve.p3
            dve.p4 = rot_matrix @ dve.p4
            
            # Recalcula a geometria interna do DVE após a rotação
            dve.recalculate_geometry()

    def update_position(self, dt, omega):
        """
        Avança a posição angular do rotor em um passo de tempo.
        
        Args:
            dt (float): Passo de tempo [s].
            omega (float): Velocidade de rotação [rad/s] em torno do eixo X.
        """
        rotation_angle = omega * dt
        
        # Gira cada pá pelo ângulo do passo de tempo
        for blade in self.blades:
            self.__rotate_blade(blade, rotation_angle)

    def get_all_dves(self):
        """
        Retorna uma lista plana com todos os DVEs de todas as pás.
        Útil para o solver.
        """
        all_dves = []
        for blade in self.blades:
            all_dves.extend(blade.dves)
        return all_dves
    
    def get_trailing_edge_dves(self):
        """
        Retorna uma lista com todos os DVEs de borda de fuga de todas as pás do rotor.
        """
        all_te_dves = []
        for blade in self.blades:
            all_te_dves.extend(blade.get_trailing_edge_dves())
        return all_te_dves

    def plot(self, real_profile = False):

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        colors = ['skyblue', 'salmon', 'lightgreen', 'gold', 'plum', 'lightcoral']
        for i, blade in enumerate(self.blades):
            # ADICIONE ESTA LINHA PARA SABER QUAL PÁ ESTÁ SENDO PLOTADA
            print(f"\n--- Preparando para plotar a Pá {i + 1} ---")
            blade_color = colors[i % len(colors)]
            blade.plot_blade(ax, color=blade_color)

        ax.set_xlabel("X (Chord)")
        ax.set_ylabel("Y (Span)")
        ax.set_zlabel("Z")
        ax.view_init(elev=25, azim=-120)
        ax.set_title(f"Rotor Geometry with {self.num_blades} Blades")
        plt.tight_layout()
        plt.show()

class Wake:
    def __init__(self, rotor, U_inf, theta, omega, type = 'fixed', n_rows=20, initial_induction_factor=1/3):
        """
        Inicializa a esteira.
        
        Args:
            rotor (Rotor): A instância do rotor.
            U_inf (float): Velocidade do escoamento livre [m/s].
            theta (float): Ângulo de discretização da esteira
            omega (float): Velocidade de rotação [rad/s].
            n_rows (int): Número de fileiras de DVEs para gerar na esteira inicial.
            initial_induction_factor (float): Fator 'a' para estimar a velocidade na esteira.
        """
        self.rows = []
        self.type = type
        self.n_rows = n_rows
        self.__generate_wake(type, rotor, U_inf, theta, omega, n_rows, initial_induction_factor)

    def __generate_wake(self, type, rotor, U_inf, theta, omega, n_rows, initial_induction_factor):
        match type:
            case 'fixed':
                self.__generate_initial_fixed_wake(rotor, U_inf, theta, omega, n_rows, initial_induction_factor)
            case 'relaxed':
                self.__generate_initial_relaxed_wake()
            case 'semi':
                self.__generate_initial_semirelaxed_wake()
            case _:
                self.__generate_initial_fixed_wake(rotor, U_inf, theta, omega, n_rows, initial_induction_factor)

    def __generate_initial_fixed_wake(self, rotor, U_inf, theta, omega, n_rows, induction_factor):
        """
        Gera uma geometria de esteira helicoidal inicial com base na cinemática do rotor.
        """
        print(f"Gerando esteira inicial fixa com {n_rows} fileiras...")

        # 1. Velocidade efetiva e passo de tempo para a construção da esteira
        V_eff = U_inf * (1 - induction_factor)
        # Define um passo de tempo baseado em uma rotação de, por exemplo, 10 graus
        if abs(omega) > 1e-6:
            dt = np.deg2rad(theta) / omega
        else:
            dt = 1.0
        # 2. Pega os pontos da borda de fuga das pás na posição t=0 como base
        te_dves_t0 = rotor.get_trailing_edge_dves()
        
        # Extrai os nós (pontos de canto) da borda de fuga.
        # Para uma pá, teremos uma linha de nós (p2, p3)
        # Precisamos organizar isso por pá.
        
        helical_lines = []
        for blade in rotor.blades:
            te_nodes_blade = []
            te_dves_blade = blade.get_trailing_edge_dves()
            # Adiciona o primeiro ponto (p2 do primeiro DVE da TE)
            te_nodes_blade.append(te_dves_blade[0].p2)
            # Adiciona os pontos p3 de todos os DVEs da TE
            for dve in te_dves_blade:
                te_nodes_blade.append(dve.p3)
            helical_lines.append(np.array(te_nodes_blade))

        # 3. "Viaja para trás no tempo", gerando as "costelas" da esteira
        wake_lines_by_age = [helical_lines] # A linha de idade 0 está em t=0
        
        for k in range(1, n_rows + 1): # Para cada passo de tempo no passado
            age = k * dt
            rotation_angle = omega * age
            axial_translation = np.array([V_eff * age, 0, 0])

            rot_matrix = np.array([
                [1, 0, 0],
                [0, np.cos(rotation_angle), -np.sin(rotation_angle)],
                [0, np.sin(rotation_angle), np.cos(rotation_angle)]
            ])

            # Calcula a posição das linhas helicoidais nesta "idade"
            current_age_lines = []
            for line_t0 in helical_lines:
                # Aplica rotação e translação
                rotated_line = (rot_matrix @ line_t0.T).T
                translated_line = rotated_line + axial_translation
                current_age_lines.append(translated_line)
            
            wake_lines_by_age.append(current_age_lines)

        # 4. Constrói as "peles" (DVEs) entre as "costelas"
        for j in range(n_rows): # Para cada fileira de DVEs
            new_wake_row = []
            inboard_lines = wake_lines_by_age[j]
            outboard_lines = wake_lines_by_age[j+1]

            for blade_idx in range(rotor.num_blades): # Para cada pá
                line_in = inboard_lines[blade_idx]
                line_out = outboard_lines[blade_idx]

                for i in range(len(line_in) - 1): # Para cada DVE ao longo da envergadura
                    p1 = line_in[i]
                    p2 = line_in[i+1]
                    p3 = line_out[i+1]
                    p4 = line_out[i]
                    
                    wake_dve = DVE(p1, p2, p3, p4)
                    wake_dve.gamma_coeffs = [0, 0, 0]
                    new_wake_row.append(wake_dve)
            
            self.rows.append(new_wake_row)
        
        print("Esteira inicial gerada com sucesso.")

    def __generate_initial_semirelaxed_wake(self):
        pass
       
    def __generate_initial_relaxed_wake(self):
        pass

    
    
    def get_all_dves(self):
        """Retorna uma lista plana com todos os DVEs da esteira."""
        all_dves = []
        for row in self.rows:
            all_dves.extend(row)
        return all_dves