from propellerDVE import *

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
propeller.plotDVE(plot_normal=True)
