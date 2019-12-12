#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script uses nqueens.py module for solving 8-queens problem by using 
a genetic algorithm
"""
import sys
print('Python version:', sys.version)

import nqueens as nq

solver=nq.Solver_8_queens(0.3,0.4,160)
best_sol,best_fit, epoch_num, visualization = solver.solve(0.9,100)

print("Best solution:",best_sol)
print("Fitness:", best_fit)
print("Iterations:", epoch_num)
print(visualization)
