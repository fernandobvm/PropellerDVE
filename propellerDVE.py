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
        self.__calculate_kse()

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
        return self.mid_le + 0.75*(self.mid_te - self.mid_le)

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
        self.e_xi = v1 / np.linalg.norm(v1)
        self.e_zeta = np.cross(v1, v2)
        self.e_zeta /= np.linalg.norm(self.e_zeta)
        self.e_eta = np.cross(self.e_zeta, self.e_xi)

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
        Calcula os sweep angles do painel DVE em radianos.
        """
        # Leading edge vector: p4 - p1
        v_le = self.p4 - self.p1
        # Trailing edge vector: p3 - p2
        v_te = self.p3 - self.p2
        # Mid-chord vector: (p3+p4)/2 - (p1+p2)/2
        v_mc = 0.5 * (self.p3 + self.p4) - 0.5 * (self.p1 + self.p2)

        # Direção da corda: p2 - p1
        chord_vector = self.p2 - self.p1
        e_chord = chord_vector / np.linalg.norm(chord_vector)

        # Função auxiliar para calcular o ângulo entre um vetor e a corda no plano (xi, eta)
        def angle_with_chord(v):
            proj = v - np.dot(v, e_chord) * e_chord  # componente perpendicular à corda
            return np.arctan2(np.dot(proj, self.local_axes[1]), np.dot(v, e_chord))

        # É necessário que self.local_axes tenha sido computado antes!
        if not hasattr(self, "local_axes"):
            raise ValueError("local_axes devem ser definidos antes de calcular os sweep angles")

        # Leading-edge sweep
        self.sweep_le = angle_with_chord(v_le)
        # Trailing-edge sweep
        self.sweep_te = angle_with_chord(v_te)
        # Mid-chord sweep
        self.sweep_mc = angle_with_chord(v_mc)

    def compute_half_chord(self):
        leading_edge_mid = 0.5 * (self.p1 + self.p2)  # p1, p2
        trailing_edge_mid = 0.5 * (self.p3 + self.p4)  # p3, p4
        chord_vector = trailing_edge_mid - leading_edge_mid
        return 0.5 * np.linalg.norm(chord_vector)
    
    def compute_half_span(self):
        span_vector = self.p2 - self.p1  # p2 - p1
        return 0.5 * np.linalg.norm(span_vector)

    def compute_quadratic_dve_velocity(self, x_local):
        """
        Calcula a velocidade induzida no ponto x_local pelo DVE atual com vorticidade γ(η) = A + Bη + Cη²
        """
        A, B, C = self.gamma_coeffs
        F1, F2, F3 = self.get_influence_terms_quadratic_eta(x_local)
        return A * F1 + B * F2 + C * F3

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

        # Coeficientes da circulação no TE
        coeffs_te = -self.gamma_coeffs

        # ---- Leading Edge -----
        local_le = self.global2local(x_global, frame='leading_edge')
        v_filament_le = self.__compute_filament_velocity(local_le, coeffs_le, self.sweep_le)
        v_sheet_le = self.__compute_sheet_velocity(local_le, coeffs_le, self.sweep_le, kse, kle)

        # ---- Trailing Edge -----
        local_te = self.global2local(x_global, frame='trailing_edge')
        v_filament_te = self.__compute_filament_velocity(local_te, coeffs_te, self.sweep_te)
        v_sheet_te = self.__compute_sheet_velocity(local_te, coeffs_te, self.sweep_te, kse, kle)

        # ----- Somar contribuições em coordenadas locais ----
        v_total_local = v_filament_le + v_sheet_le + v_filament_te + v_sheet_te

        # ----- Transformar de volta as coordenadas globais
        v_total_global = self.local2global_vector(v_total_local)

        return v_total_global


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
        return (1/rho)*(a2*beta2*zeta + b2*beta1)
    
    def __calculate_gamma2(self, rho, a2, b2, beta1, beta2, zeta):
        return (1/rho)*(a2*beta1*zeta - b2*beta2)
    
    def __calculate_delta1(self, rho, b2, c2, beta1, beta2, zeta):
        return (1/rho)*(b2*beta2*zeta + c2*beta1)
    
    def __calculate_delta2(self, rho, b2, c2, beta1, beta2, zeta):
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
        return (a*eta + b)/((a*c - b**2)*self.__calculate_r_eta(a, b, c, eta))
    
    def __calculate_G12(self, a, b, c, eta):
        return -(b*eta + c)/((a*c - b**2)*self.__calculate_r_eta(a, b, c, eta))
    
    def __calculate_G13(self, a, b, c, eta):
        return ((2*b**2 - a*c)*eta + b*c)/((a*c - b**2)*a*self.__calculate_r_eta(a, b, c, eta)) + (1/np.sqrt(a**3))*np.log(np.sqrt(a)*self.__calculate_r_eta(a, b, c, eta) + a*eta + b)

    def __calculate_G21(self, rho, beta1, beta2, t, gamma1, gamma2, delta1, delta2, zeta, a, b, c, kse):
        return (beta1/(2*rho))*np.log(self.__calculate_mu1(gamma1, gamma2, t, delta1, delta2, zeta, a, b, c, kse)) + (beta2/(rho*zeta))*self.__calculate_mu2(gamma1, gamma2, t, delta1, delta2, zeta, a, b, c)

    def __calculate_G22(self, rho, beta1, beta2, t, gamma1, gamma2, delta1, delta2, zeta, a, b, c, kse):
        return (-beta2/(2*rho*zeta))*np.log(self.__calculate_mu1(gamma1, gamma2, t, delta1, delta2, zeta, a, b, c, kse)) + (beta1/(rho*zeta))*self.__calculate_mu2(gamma1, gamma2, t, delta1, delta2, zeta, a, b, c)

    def __calculate_G23(self, a, b, c, t, kle):
        mu3 = self.__calculate_mu3(a, b, c, t, kle)
        return (self.__calculate_r_t(a, b, c, t)/a - (b/np.sqrt(a**3))*np.log(mu3))
    
    def __calculate_G24(self, a, b, c, t, kle):
        mu3 = self.__calculate_mu3(a, b, c, t, kle)
        return (1/np.sqrt(a))*np.log(mu3)
    
    def __calculate_G25(self, zeta, t, kse):
        return 0.5*np.log(t**2 + zeta**2 + kse)
    
    def __calculate_G26(self, zeta, t):
        return (1/zeta)*(np.atan(t/zeta))
    
    def __calculate_G27(self, t):
        return t
class DVESolver:
    def __init__(self, dves, V_inf=np.array([1.0, 0.0, 0.0])):
        self.dves = dves
        self.V_inf = V_inf
        self.A = None
        self.b = None
        self.gamma = None

    def assemble(self):
        N = len(self.dves)
        rows = []  # lista de linhas da matriz A
        rhs = []   # lado direito do sistema (vetor b)

        # === (1) Condição cinemática: não-penetração em 3 pontos por DVE ===
        for j, dve_j in enumerate(self.dves):
            x_cp_list = dve_j.control_points_spanwise()
            n_j = dve_j.normal

            for m, x_j in enumerate(x_cp_list):
                row = np.zeros(3 * N)
                rhs_value = -np.dot(n_j, self.V_inf)

                for i, dve_i in enumerate(self.dves):
                    x_j_local = dve_i.global2local(x_j)
                    F1, F2, F3 = dve_i.get_influence_terms_quadratic_eta(x_j_local)

                    for k, Fk in enumerate([F1, F2, F3]):
                        col = i * 3 + k
                        row[col] = np.dot(n_j, dve_i.local2global(Fk))

                rows.append(row)
                rhs.append(rhs_value)

        # === (2) Continuidade de vorticidade entre DVEs vizinhos (spanwise) ===
        for i in range(N - 1):
            dve_left = self.dves[i]
            dve_right = self.dves[i + 1]

            # Eq: γ_left(+h/2) = γ_right(-h/2)
            eta = dve_left.half_span / 2
            row = np.zeros(3 * N)
            row[i * 3 + 0] = 1             # A_left
            row[i * 3 + 1] = eta           # B_left
            row[i * 3 + 2] = eta ** 2     # C_left

            row[(i+1) * 3 + 0] = -1             # -A_right
            row[(i+1) * 3 + 1] = -(-eta)        # -B_right
            row[(i+1) * 3 + 2] = -(-eta) ** 2   # -C_right

            rows.append(row)
            rhs.append(0.0)

            # Eq: dγ_left/dη(+h/2) = dγ_right/dη(-h/2)
            row = np.zeros(3 * N)
            row[i * 3 + 1] = 1                 # B_left
            row[i * 3 + 2] = 2 * eta           # 2*C_left

            row[(i+1) * 3 + 1] = -1            # -B_right
            row[(i+1) * 3 + 2] = -2 * (-eta)   # -2*C_right

            rows.append(row)
            rhs.append(0.0)

        # === (3) Condições de contorno na raiz e ponta ===
        # Exemplo: γ = 0 na ponta (último painel)
        eta_tip = self.dves[-1].half_span / 2
        row = np.zeros(3 * N)
        row[-3] = 1                      # A_last
        row[-2] = eta_tip               # B_last
        row[-1] = eta_tip ** 2          # C_last
        rows.append(row)
        rhs.append(0.0)

        # Exemplo: dγ/dη = 0 na raiz (primeiro painel)
        eta_root = -self.dves[0].half_span / 2
        row = np.zeros(3 * N)
        row[1] = 1                      # B_0
        row[2] = 2 * eta_root           # 2*C_0
        rows.append(row)
        rhs.append(0.0)

        # === Finalização do sistema ===
        self.A = np.vstack(rows)
        self.b = np.array(rhs)

    def solve(self):
        self.gamma = np.linalg.solve(self.A, self.b)
        for i, dve in enumerate(self.dves):
            dve.gamma_coeffs = self.gamma[i*3:(i+1)*3]

    def run(self):
        self.assemble()
        self.solve()

