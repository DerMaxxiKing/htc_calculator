from .construction import Solid, Fluid

water = Fluid(name='water',
              id='f001',
              mol_weight=18,
              density=1000,
              specific_heat_capacity=4181,
              hf=0,
              mu=959*10e-6,
              pr=6.62)

aluminium = Solid(name='aluminium',
                  id='s001',
                  mol_weight=27,
                  density=2700,
                  heat_conductivity=200,
                  specific_heat_capacity=900)

concrete = Solid(name='concrete',
                 id='s002',
                 density=2600,
                 specific_heat_capacity=1000,
                 heat_conductivity=2.5)

rockwool = Solid(name='rockwool',
                 id='s003',
                 density=250,
                 specific_heat_capacity=840,
                 heat_conductivity=0.034)

plaster = Solid(name='plaster',
                id='s004',
                density=1500,
                specific_heat_capacity=960,
                heat_conductivity=0.60)
