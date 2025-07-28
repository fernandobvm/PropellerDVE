from propellerDVE import *

# Exemplo de pá com torção e afilamento simples
sections = [
    PropellerSection(r=1.0, chord=1.2, twist_deg=14),
    PropellerSection(r=3.0, chord=1.1, twist_deg=12),
    PropellerSection(r=5.0, chord=1.0, twist_deg=10),
    PropellerSection(r=7.0, chord=0.9, twist_deg=7),
    PropellerSection(r=9.0, chord=0.8, twist_deg=5),
    PropellerSection(r=11.0, chord=0.6, twist_deg=2),
]

propeller = PropellerGeometry(R=6.0, sections=sections)
propeller.plot3d()