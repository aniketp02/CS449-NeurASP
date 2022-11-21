% Sudoku Solver n=9

#const n=9.

1{filled(R, C, 1..n)}1:- R = 0..n-1, C = 0..n-1.

R1 = R2 :- filled(R1, C1, X), filled(R2, C1, X).
C1 = C2 :- filled(R1, C1, X), filled(R1, C2, X).
:- filled(R1, C1, X), filled(R2, C2, X), R1/3 = R2/3, C1/3 = C2/3, R1 != R2, C1 != C2.

:- given (R ,C , X), not filled (R ,C , X), not invalid(R, C, X).

#show filled/3.