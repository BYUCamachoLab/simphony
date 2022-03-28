from simphony.libraries.siepic import *


bdc1 = BidirectionalCoupler()
bdc2 = BidirectionalCoupler()
print(f'\nIdentical Bidirectional couplers:{bdc1.__hash__()}, {bdc2.__hash__()}\n')

bdc1 = BidirectionalCoupler(2.1e-7)
bdc2 = BidirectionalCoupler(2.3e-7)
print(f'Non-identical Bidirectional couplers:{bdc1.__hash__()}, {bdc2.__hash__()}\n')

# FIXME
hr1siepic = HalfRing(gap=1e-7,radius=1e-5,width=5e-7,thickness=2.2e-7)
hr2siepic = HalfRing(gap=1e-7,radius=1e-5,width=5e-7,thickness=2.2e-7)
print(f'Identical halfrings:{hr1siepic.__hash__()}, {hr2siepic.__hash__()}\n')

hr1siepic = HalfRing(gap=8e-8,radius=10e-6,width=5.2e-7,thickness=2.1e-7)
hr2siepic = HalfRing(gap=1e-7,radius=1e-5,width=5e-7,thickness=2.2e-7,couple_length=0)
print(f'Non-identical halfrings:{hr1siepic.__hash__()}, {hr2siepic.__hash__()}\n')


dc1 = DirectionalCoupler()
dc2 = DirectionalCoupler()
print(f'Identical Directional couplers:{dc1.__hash__()}, {dc2.__hash__()}')

dc1 = DirectionalCoupler(gap=2e-7, Lc=10e-6)    # this autocorrects to default attribute values everytime, not sure why
dc2 = DirectionalCoupler()
print(f'Non-identical Directional couplers:{dc1.__hash__()}, {dc2.__hash__()}')


term1 = Terminator()
term2 = Terminator()
print(f'Identical terminators:{term1.__hash__()}, {term2.__hash__()}')

term1 = Terminator(w1=5e-7)     # this autocorrects to default attribute values everytime
term2 = Terminator()
print(f'Non-identical terminators:{term1.__hash__()}, {term2.__hash__()}')


# wg1 and wg2 hash should be different
wg1 = Waveguide(length=150e-6)
wg2 = Waveguide(length=50e-6)
print(f'Non-identical waveguides:{wg1.__hash__()},{wg2.__hash__()}')

# wg1 and wg2 hash should be the same
wg1 = Waveguide(length=150e-6)
wg2 = Waveguide(length=150e-6)
print(f'Identical waveguides:{wg1.__hash__()},{wg2.__hash__()}')
