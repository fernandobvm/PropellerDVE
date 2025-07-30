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


        X, Y, Z = [], [], []

        for sec in self.sections:

            # Escalar pela corda local
            x_scaled = (sec.x - 0.25) * sec.chord  # referência em 1/4 da corda
            if real_profile:
                z_scaled = sec.z_real * sec.chord
            else:
                z_scaled = sec.z * sec.chord

            # Aplicar torção
            x_rot = x_scaled * np.cos(sec.twist) - z_scaled * np.sin(sec.twist)
            z_rot = x_scaled * np.sin(sec.twist) + z_scaled * np.cos(sec.twist)

            # Posicionar no raio atual
            y_coords = np.full_like(x_rot, sec.r)

            X.append(x_rot)
            Y.append(y_coords)
            Z.append(z_rot)

        X = np.array(X)
        Y = np.array(Y)
        Z = np.array(Z)

        ax.plot_surface(X, Y, Z, color='lightblue', edgecolor='gray', alpha=0.95, rstride=1, cstride=1)

        ax.set_xlabel("X (Chord)")
        ax.set_ylabel("Y (Span)")
        ax.set_zlabel("Z")
        ax.set_title("Propeller Geometry")
        ax.view_init(elev=25, azim=-120)
        plt.tight_layout()
        plt.show()

    def discretize2DVE(self, m=5, n=10):
        """
        Discretiza a pá em DVEs, utilizando os perfis armazenados nas seções (pré-processados).
        m -> chordwise
        n -> spanwise
        """

        r_vals = np.linspace(self.sections[0].r, self.sections[-1].r, n)
        profiles = []

        # Criar interpoladores para corda e twist
        r_sections = [s.r for s in self.sections]
        chord_interp = interp1d(r_sections, [s.chord for s in self.sections], kind='linear')
        twist_interp = interp1d(r_sections, [s.twist for s in self.sections], kind='linear')

        # Lista auxiliar com os perfis normalizados
        x_profiles = [s.x for s in self.sections]
        z_profiles = [s.z for s in self.sections]

        # Para cada r do span, encontrar perfil mais próximo
        for r in r_vals:
            # Seleciona seção mais próxima (você pode usar interpolação de perfil se quiser algo mais avançado)
            closest_section = min(self.sections, key=lambda s: abs(s.r - r))
            x2d = closest_section.x
            z2d = closest_section.z

            chord = chord_interp(r)
            twist = twist_interp(r)

            # Centraliza em x=0.25, escala e aplica twist
            x_scaled = (x2d - 0.25) * chord
            z_scaled = z2d * chord
            x_rot = x_scaled * np.cos(twist) - z_scaled * np.sin(twist)
            z_rot = x_scaled * np.sin(twist) + z_scaled * np.cos(twist)
            y = np.full_like(x_rot, r)

            profile_3d = np.vstack((x_rot, y, z_rot)).T
            profiles.append(profile_3d)

        # Construção dos DVEs
        dves = []
        for j in range(n - 1):      # spanwise
            for i in range(m):      # chordwise
                p1 = profiles[j][i]
                p2 = profiles[j][i+1]
                p3 = profiles[j+1][i+1]
                p4 = profiles[j+1][i]
                dve = DVE(np.array(p1), np.array(p2), np.array(p3), np.array(p4))
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

class DVE:
    def __init__(self, p1, p2, p3, p4):
        self.p1 = p1  # canto LE inboard
        self.p2 = p2  # canto TE inboard
        self.p3 = p3  # canto TE outboard
        self.p4 = p4  # canto LE outboard

        self.control_point = self.compute_control_point()
        self.normal = self.compute_normal()
    
    def compute_control_point(self):
        # Ponto de controle: centro do painel (média dos vértices)
        return 0.25 * (self.p1 + self.p2 + self.p3 + self.p4)

    def compute_normal(self):
        # Vetor normal estimado com produto vetorial das diagonais
        v1 = self.p2 - self.p1
        v2 = self.p4 - self.p1
        n = np.cross(v2, v1)
        return n / np.linalg.norm(n)


