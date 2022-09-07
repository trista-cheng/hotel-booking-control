#!/usr/bin/env python3.7

# Copyright 2022, Gurobi Optimization, LLC

# Solve a multi-commodity flow problem.  Two products ('Pencils' and 'Pens')
# are produced in 2 cities ('Detroit' and 'Denver') and must be sent to
# warehouses in 3 cities ('Boston', 'New York', and 'Seattle') to
# satisfy demand ('inflow[h,i]').
#
# Flows on the transportation network must respect arc capacity constraints
# ('capacity[i,j]'). The objective is to minimize the sum of the arc
# transportation costs ('cost[i,j]').

import time
import gurobipy as gp
from gurobipy import GRB, LinExpr, quicksum
from itertools import product

# Base data
commodities = ['Pencils', 'Pens']
s_commodities = 'a b c d e f g h i j k l m n o p q r s t u v w x y z'.split()
commodities += s_commodities
nodes = ['Detroit', 'Denver', 'Boston', 'New York', 'Seattle']

arcs, capacity = gp.multidict({
    ('Detroit', 'Boston'):   100,
    ('Detroit', 'New York'):  80,
    ('Detroit', 'Seattle'):  120,
    ('Denver',  'Boston'):   120,
    ('Denver',  'New York'): 120,
    ('Denver',  'Seattle'):  120})

c_dict = {
    'Boston': ['Detroit', 'Denver'],
    'New York': ['Detroit', 'Denver'],
    'Seattle': ['Detroit', 'Denver'],
    'Detroit': [],
    'Denver': []
}
c2_dict = {
    'Detroit': ['Boston', 'New York', 'Seattle'],
    'Denver': ['Boston', 'New York', 'Seattle'],
    'Boston': [],
    'New York': [],
    'Seattle': [],
}

# Cost for triplets commodity-source-destination
cost = {
    ('Pencils', 'Detroit', 'Boston'):   10,
    ('Pencils', 'Detroit', 'New York'): 20,
    ('Pencils', 'Detroit', 'Seattle'):  60,
    ('Pencils', 'Denver',  'Boston'):   40,
    ('Pencils', 'Denver',  'New York'): 40,
    ('Pencils', 'Denver',  'Seattle'):  30,
    ('Pens',    'Detroit', 'Boston'):   20,
    ('Pens',    'Detroit', 'New York'): 20,
    ('Pens',    'Detroit', 'Seattle'):  80,
    ('Pens',    'Denver',  'Boston'):   60,
    ('Pens',    'Denver',  'New York'): 70,
    ('Pens',    'Denver',  'Seattle'):  30}
for c in s_commodities:
    for i, j in product(['Detroit', 'Denver'], ['Boston', 'New York', 'Seattle']):
        cost[(c, i, j)] = -10

# Demand for pairs of commodity-city
inflow = {
    ('Pencils', 'Detroit'):   50,
    ('Pencils', 'Denver'):    60,
    ('Pencils', 'Boston'):   -50,
    ('Pencils', 'New York'): -50,
    ('Pencils', 'Seattle'):  -10,
    ('Pens',    'Detroit'):   60,
    ('Pens',    'Denver'):    40,
    ('Pens',    'Boston'):   -40,
    ('Pens',    'New York'): -30,
    ('Pens',    'Seattle'):  -30}

for c, i in product(s_commodities, nodes):
    if c in 'a b c d e f g h i j k l m'.split():
        inflow[(c, i)] = 0
    else:
        inflow[(c, i)] = 0

start = time.time()
for i in range(500):
    # Create optimization model
    m = gp.Model('netflow')

    # Create variables
    flow = m.addVars(commodities, arcs, obj=cost, name="flow")

    # Arc-capacity constraints
    
    for i, j in arcs:
        l = LinExpr(quicksum(flow[h, i, j] for h in commodities))
        m.addConstr(
        (l <= capacity[i, j] ), "cap")

    # Equivalent version using Python looping
    # for i, j in arcs:
    #   m.addConstr(sum(flow[h, i, j] for h in commodities) <= capacity[i, j],
    #               "cap[%s, %s]" % (i, j))


    # Flow-conservation constraints
    for h in commodities:
        for j in nodes:
            expr = LinExpr(quicksum(flow[h, j1, j] for j1 in c_dict[j]))
            expr.add(inflow[h, j])
            
            r = LinExpr(quicksum(flow[h, j, j2] for j2 in c2_dict[j]))

            m.addConstr(
                (
                    expr == r
                ), 
                "node"
            )

    # Alternate version:
    # m.addConstrs(
    #   (gp.quicksum(flow[h, i, j] for i, j in arcs.select('*', j)) + inflow[h, j] ==
    #     gp.quicksum(flow[h, j, k] for j, k in arcs.select(j, '*'))
    #     for h in commodities for j in nodes), "node")

    # Compute optimal solution
    m.optimize()

    # # Print solution
    # if m.Status == GRB.OPTIMAL:
    #     solution = m.getAttr('X', flow)
    #     for h in commodities:
    #         print('\nOptimal flows for %s:' % h)
    #         for i, j in arcs:
    #             if solution[h, i, j] > 0:
    #                 print('%s -> %s: %g' % (i, j, solution[h, i, j]))

print(time.time() - start)