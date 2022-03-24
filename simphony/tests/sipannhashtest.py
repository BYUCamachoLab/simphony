from simphony.libraries.sipann import *

hr1siepic = HalfRing(gap=2e-7,radius=1e-5,width=5e-7,thickness=2.2e-7)
hr2siepic = HalfRing(gap=2e-7,radius=1e-5,width=5e-7,thickness=2.2e-7)
print(f'Identical halfrings:{hr1siepic.__hash__()}, {hr2siepic.__hash__()}\n')

hr1siepic = HalfRing(gap=1.5e-7,radius=10e-6,width=5.2e-7,thickness=2.1e-7)
hr2siepic = HalfRing(gap=2e-7,radius=1e-5,width=5e-7,thickness=2.2e-7)
print(f'Non-identical halfrings:{hr1siepic.__hash__()}, {hr2siepic.__hash__()}\n')


hra1siepic = HalfRacetrack(gap=1e-7,radius=1e-5,width=5e-7,thickness=2.2e-7,length=1e-7)
hra2siepic = HalfRacetrack(gap=1e-7,radius=1e-5,width=5e-7,thickness=2.2e-7,length=1e-7)
print(f'Identical halfracetracks:{hra1siepic.__hash__()}, {hra2siepic.__hash__()}\n')

hra1siepic = HalfRacetrack(gap=1.5e-7,radius=10e-6,width=5.2e-7,thickness=2.1e-7,length=1e-7)
hra2siepic = HalfRacetrack(gap=2e-7,radius=1e-5,width=5e-7,thickness=2.2e-7,length=2e-7)
print(f'Non-identical halfracetracks:{hra1siepic.__hash__()}, {hra2siepic.__hash__()}\n')


sc1 = StraightCoupler(width=5e-7, thickness=2e-7, gap=2e-7, length=1e-7)
sc2 = StraightCoupler(width=5e-7, thickness=2e-7, gap=2e-7, length=1e-7)
print(f'Identical straight coupler:{sc1.__hash__()}, {sc2.__hash__()}\n')

sc1 = StraightCoupler(width=5.8e-7, thickness=2e-7, gap=3e-7, length=2e-7)
sc2 = StraightCoupler(width=5e-7, thickness=2.3e-7, gap=2e-7, length=1e-7)
print(f'Non-identical straight coupler:{sc1.__hash__()}, {sc2.__hash__()}\n')


# wg1 and wg2 hash should be different
wg1 = Waveguide(length=150e-6,width=5e-7,thickness=2e-7)
wg2 = Waveguide(length=50e-6,width=6e-7,thickness=2.3e-7)
print(f'Non-identical waveguides:{wg1.__hash__()},{wg2.__hash__()}\n')

# wg1 and wg2 hash should be the same
wg1 = Waveguide(length=150e-6,width=5e-7,thickness=2e-7)
wg2 = Waveguide(length=150e-6,width=5e-7,thickness=2e-7)
print(f'Identical waveguides:{wg1.__hash__()},{wg2.__hash__()}\n')



stcoup1 = Standard(width=5e-7,thickness=2.3e-7,gap=2e-7,length=10e-6,horizontal=1e-6,vertical=1e-6)
stcoup2 = Standard(width=5e-7,thickness=2.3e-7,gap=2e-7,length=10e-6,horizontal=1e-6,vertical=1e-6)
print(f'Identical standard couplers:{stcoup1.__hash__()},{stcoup2.__hash__()}\n')

stcoup1 = Standard(width=5.8e-7,thickness=2.2e-7,gap=3e-7,length=9e-6,horizontal=2e-6,vertical=2e-6)
stcoup2 = Standard(width=5e-7,thickness=2.3e-7,gap=2e-7,length=10e-6,horizontal=1e-6,vertical=1e-6)
print(f'Non-identical standard couplers:{stcoup1.__hash__()},{stcoup2.__hash__()}\n')

dhr1 = DoubleHalfRing(width=5e-7, thickness=2.3e-7, radius=1e-5,gap=2e-7)
dhr2 = DoubleHalfRing(width=5e-7, thickness=2.3e-7, radius=1e-5,gap=2e-7)
print(f'Identical double half ring:{dhr1.__hash__()},{dhr2.__hash__()}\n')

dhr1 = DoubleHalfRing(width=5.8e-7, thickness=2.2e-7, radius=9e-6,gap=3e-7)
dhr2 = DoubleHalfRing(width=5e-7, thickness=2.3e-7, radius=1e-5,gap=2e-7)
print(f'Non-identical double half ring:{dhr1.__hash__()},{dhr2.__hash__()}\n')


ahr1 = AngledHalfRing(width=5e-7, thickness=2.3e-7, radius=1e-5,theta=45,gap=2e-7)
ahr2 = AngledHalfRing(width=5e-7, thickness=2.3e-7, radius=1e-5,theta=45,gap=2e-7)
print(f'Identical angled half ring:{ahr1.__hash__()},{ahr2.__hash__()}\n')

ahr1 = AngledHalfRing(width=5.8e-7, thickness=2.2e-7, radius=9e-6,theta=50,gap=3e-7)
ahr2 = AngledHalfRing(width=5e-7, thickness=2.3e-7, radius=1e-5,theta=45,gap=2e-7)
print(f'Non-identical angled half ring:{ahr1.__hash__()},{ahr2.__hash__()}\n')
