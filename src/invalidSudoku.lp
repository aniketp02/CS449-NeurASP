invalid(R1, C1, X) :- given(R1, C1, X), given(R2, C1, X), R1 != R2.
invalid(R1, C1, X) :- given(R1, C1, X), given(R1, C2, X), C1 != C2.
invalid(R1, C1, X) :- given(R1, C1, X), given(R2, C2, X), R1/3 = R2/3, C1/3 = C2/3, R1 != R2, C1 != C2.
#show invalid/3.
% if everything is ok, returns empty, not unsat.