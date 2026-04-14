import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from propellerDVE import PropellerSection, PropellerGeometry, Rotor, Wake, DVE

def get_wake_points(wake):
    """
    Extrai todos os pontos dos painéis da esteira para visualização.
    Retorna listas de coordenadas X, Y, Z.
    """
    wx, wy, wz = [], [], []
    for dve in wake.get_all_dves():
        # Adiciona os 4 pontos de cada DVE para plotar o contorno
        pts = np.array([dve.p1, dve.p2, dve.p3, dve.p4, dve.p1])
        wx.extend(pts[:, 0])
        wy.extend(pts[:, 1])
        wz.extend(pts[:, 2])
        # Adiciona um NaN para que o matplotlib não conecte o final de um DVE ao início do próximo
        wx.append(np.nan)
        wy.append(np.nan)
        wz.append(np.nan)
    return wx, wy, wz

def validate_wake_geometry(rotor, wake, U_inf, omega, dt):
    """
    Verifica numericamente se a esteira segue a trajetória helicoidal esperada,
    considerando a posição inicial do bordo de fuga e o fator de indução.
    """
    print("\n--- INICIANDO VALIDAÇÃO NUMÉRICA DA ESTEIRA (AJUSTADA) ---")
    
    # 1. VERIFICAÇÃO DE CONEXÃO
    blade_te_dves = rotor.blades[0].get_trailing_edge_dves()
    n_span = len(blade_te_dves)
    
    # Ponto de referência na pá (Ponta do Bordo de Fuga)
    # Usamos o último DVE da pá
    blade_tip_dve = blade_te_dves[-1]
    blade_tip_point = blade_tip_dve.p3 # Ponto mais externo do TE
    
    # Ponto correspondente na esteira (Primeira fileira)
    wake_first_row = wake.rows[0]
    # O DVE da ponta na esteira é o último correspondente àquela pá
    wake_tip_dve_start = wake_first_row[n_span-1] 
    
    # Nota: A esteira conecta t=0 (P1/P2) a t=dt (P4/P3)
    # Ponto P2 do DVE da esteira deve conectar ao P3 da pá
    wake_start_point = wake_tip_dve_start.p2 
    
    dist_error = np.linalg.norm(blade_tip_point - wake_start_point)
    if dist_error < 1e-6:
        print(f"\n[Teste 1] Conectividade Pá-Esteira: [SUCESSO] (Erro = {dist_error:.2e} m)")
    else:
        print(f"\n[Teste 1] Conectividade Pá-Esteira: [FALHA]")
        print(f"  Ponto Pá: {np.round(blade_tip_point, 3)}")
        print(f"  Ponto Esteira: {np.round(wake_start_point, 3)}")

    # 2. VERIFICAÇÃO DA TRAJETÓRIA AXIAL
    print("\n[Teste 2] Trajetória Axial e Indução:")
    
    # Pega a posição X inicial (onde a esteira nasce)
    x_start = blade_tip_point[0]
    
    # Pega um ponto na esteira em uma fileira mais avançada (ex: fileira 5)
    row_idx = 4
    if len(wake.rows) <= row_idx:
        row_idx = len(wake.rows) - 1
        
    target_dve = wake.rows[row_idx][n_span-1]
    p_eval = target_dve.p3 # Ponto na idade 'age'
    
    age = (row_idx + 1) * dt
    
    # Deslocamento Real
    dx_real = p_eval[0] - x_start
    
    # Velocidade Efetiva Real
    V_eff_real = abs(dx_real / age)
    
    # Fator de Indução Calculado (a = 1 - V_real/U_inf)
    induction_calculated = 1 - (V_eff_real / U_inf)
    
    print(f"  Posição Inicial (TE): X = {x_start:.3f} m")
    print(f"  Posição na Fileira {row_idx+1} (t={age:.3f}s): X = {p_eval[0]:.3f} m")
    print(f"  Deslocamento Total: {dx_real:.3f} m")
    print(f"  Velocidade de Convecção Média: {V_eff_real:.2f} m/s")
    print(f"  Fator de Indução Implícito: a = {induction_calculated:.3f}")
    
    # Verifica direção
    if dx_real < 0:
        print("  Direção: [OK] Negativa (Coerente com geometria da pá)")
    else:
        print("  Direção: [ALERTA] Positiva (Verificar se geometria da pá é positiva)")

    # Validação baseada na expectativa de indução
    # Se você configurou induction=1/3 (0.333) no Wake, esperamos isso.
    if 0.30 < induction_calculated < 0.36:
        print("  Validação: [SUCESSO] Velocidade consistente com indução de 1/3.")
    elif abs(induction_calculated) < 0.05:
         print("  Validação: [SUCESSO] Velocidade consistente com indução de 0 (Free wake speed).")
    else:
         print(f"  Validação: [NOTA] Indução de {induction_calculated:.2f} detectada.")

    # 3. VERIFICAÇÃO RADIAL
    R_tip = np.sqrt(blade_tip_point[1]**2 + blade_tip_point[2]**2)
    R_wake = np.sqrt(p_eval[1]**2 + p_eval[2]**2)
    
    if abs(R_wake - R_tip) < 1e-3:
        print(f"\n[Teste 3] Conservação Radial: [SUCESSO] R={R_wake:.3f}m")
    else:
        print(f"\n[Teste 3] Conservação Radial: [FALHA] R_pá={R_tip:.3f} vs R_wake={R_wake:.3f}")
def plot_validation_case():
    # --- SETUP DO CASO DE TESTE ---
    print("Gerando geometria para teste...")
    U_inf = 10.0
    RPM = 600
    omega = RPM * 2 * np.pi / 60.0
    R = 1.0
    
    # Pá simples
    sections = [
        PropellerSection(r=0.2, chord=0.1, twist_deg=0),
        PropellerSection(r=1.0, chord=0.1, twist_deg=0)
    ]
    geo = PropellerGeometry(R, sections)
    geo.discretize2DVE(m=2, n=5) # Malha grosseira para visualizar bem as linhas
    
    rotor = Rotor(geo, num_blades=1)
    
    # Gerar Esteira Fixa
    # Theta = passo angular da esteira. Ex: 15 graus por passo.
    theta_deg = 15
    n_rows = 24 # 1 volta completa (360/15 = 24)
    
    dt = np.deg2rad(theta_deg) / omega
    
    wake = Wake(rotor, U_inf, theta_deg, omega, type='fixed', n_rows=n_rows, initial_induction_factor= 0.0)
    
    # --- VALIDAÇÃO NUMÉRICA ---
    validate_wake_geometry(rotor, wake, U_inf, omega, dt)
    
    # --- VALIDAÇÃO VISUAL ---
    print("\nGerando gráfico 3D...")
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 1. Plotar a Pá (Vermelho)
    for dve in rotor.blades[0].dves:
        pts = np.array([dve.p1, dve.p2, dve.p3, dve.p4, dve.p1])
        ax.plot(pts[:,0], pts[:,1], pts[:,2], 'r-', linewidth=2, label='Pá' if dve == rotor.blades[0].dves[0] else "")
        
        # Plota vetores normais da pá para referência
        c = dve.control_point
        n = dve.normal * 0.2
        ax.quiver(c[0], c[1], c[2], n[0], n[1], n[2], color='black', alpha=0.3)

    # 2. Plotar a Esteira (Azul)
    wx, wy, wz = get_wake_points(wake)
    ax.plot(wx, wy, wz, 'b-', linewidth=0.5, alpha=0.6, label='Esteira')
    
    # 3. Destacar a "Costela" (Rib) mais recente e a mais antiga
    # Primeira fileira (conectada à pá)
    first_row = wake.rows[0]
    for dve in first_row:
        pts = np.array([dve.p1, dve.p2, dve.p3, dve.p4, dve.p1])
        ax.plot(pts[:,0], pts[:,1], pts[:,2], 'c-', linewidth=2, label='Wake Row 0' if dve==first_row[0] else "")

    # Configurações do Gráfico
    ax.set_xlabel('X (Axial)')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Validação da Esteira Fixa\nU={U_inf} m/s, RPM={RPM}')
    
    # Ajuste de escala para ver a hélice sem distorção
    ax.set_box_aspect([2, 1, 1]) 
    plt.legend()
    plt.show()

if __name__ == "__main__":
    plot_validation_case()