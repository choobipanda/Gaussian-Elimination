import sys
import numpy as np
'''
The program should take as input a file which contains data for a linear system in the following format:

n
a11 a12 ... a1n
a21 a22 ... a2n
...
an1 an2 ... ann
b1   b2   ... bn

The file should have extension .lin, for example, sys1.lin could a suitable file name. The program should use Naive Gaussian Elimination by default and should place the solution in a file with the same name as the input, but with extension .sol and with the following format:

x1 x2 ... xn

Additionally, the user should be able to modify the programs behavior with optional flag --spp, in which case the program will use Scaled Partial Pivoting to produce the solution. For example, for a system placed in file sys1.lin, the user could run:

> gaussian sys1.lin

or,

> gaussian --spp sys1.lin

An additional flag: --double, will specify the precision used. By default, the program runs using single precision, but when the flag --double is used, it runs using double precision.

In the first case the program will use NGE, and in the second it will use SPP. In both cases, the solution will be placed in file sys1.sol.

notes:
read .lin file as input
n: system size
matrix: coefficient matrix
b: constant

choose algorithm
default is naive gaussian
--spp switches to scaled partial pivoting

choose precision
default is single precision
--double switches to double precision 

solve the system

write solution(x) to .sol file

'''

#Argument function helpers
def parse_args():
    '''
    get filename, method, and precision
    '''
    args = sys.argv[1:]

    method = "spp" if "--spp" in args else "naive"
    precision = "double" if "--double" in args else "single"

    filename = None
    for arg in args:
        if arg.endswith(".lin"):
            filename = arg

    if filename is None:
        print("error: missing .lin file")
        sys.exit(1)

    return filename, method, precision

def get_datatype(precision):
    '''
    returns specified type based on precision
    '''
    return np.float64 if "double" in precision else np.float32

#File i/o helpers
def read_linear_system(filename, datatype):
    '''
    reads n, matrix, b
    returns matrix, b
    '''
    with open(filename, "r") as f:
        lines = f.readlines()
    
    n = int(lines[0].strip())

    matrix = []
    for i in range(1, n+1):
        row = [datatype(x) for x in lines[i].split()]
        matrix.append(row)

    b = [datatype(x) for x in lines[n+1].split()]

    return matrix, b

def write_solution(filename, x):
    '''
    writes solution x to .sol file
    '''
    out_file = filename.replace(".lin", ".sol")
    with open(out_file, "w") as f:
        f.write(" ".join(str(val) for val in x))

#gaussian helpers
def compute_multiplier(matrix, i, k):
    '''
    multiplier = matrix[i][k] / matrix[k][k]
    '''
    return matrix[i][k] / matrix[k][k]

def eliminate_row(matrix, b, i, k, mult):
    '''
    row_i: row_i - m_ik*row_k
    distribute multiplier 
    then subtract that row with pivot row
    '''
    n = len(matrix)
    for j in range(k+1, n):
        matrix[i][j] = matrix[i][j] - mult*matrix[k][j]
    b[i] = b[i] - mult*b[k]

def back_substitution(matrix, b):
    '''
    solve for x,
    starting from row 3, solve for x_3
    use x_3 to solve for x_2,
    then use those values to solve for x_1
    '''
    n = len(matrix)
    x = [0]*n

    x[n-1] = b[n-1] / matrix[n-1][n-1]

    for i in range(n-1, -1, -1):
        s = b[i]
        for j in range(i+1, n):
            s-= matrix[i][j] * x[j]
        x[i] = s / matrix[i][i]

    return x

def naive_forward_elimination(matrix, b):
    '''
    forward elimination w no pivoting
    '''
    n = len(matrix)
    for k in range(n-1):
        for i in range(k+1, n):
            mult = compute_multiplier(matrix, i, k)
            eliminate_row(matrix, b, i, k, mult)
            matrix[i][k] = 0

def spp_forward_elimination(matrix, b):
    '''
    forward elimination w scaled partial pivoting
    '''
    n = len(matrix)
    #scaling factors
    s = [max(abs(val) for val in row) for row in matrix]
    index = list(range(n))

    for k in range(n-1):
        #find pivot row
        max_ratio = 0
        pivot = k

        for i in range(k, n):
            ratio = abs(matrix[i][k]) / s[index[i]]
            if ratio > max_ratio:
                max_ratio = ratio
                pivot = i
        
        #swap rows
        index[k], index[pivot] = index[pivot], index[k]
        
        #eliminate below the pivot
        for i in range(k+1, n):
            mult = matrix[index[i]][k] / matrix[index[k]][k]
            for j in range(k, n):
                matrix[index[i]][j] -= mult * matrix[index[k]][j]
            b[index[i]] -= matrix * b[index[k]]


def naive_gaussian(matrix, b):
    naive_forward_elimination(matrix, b)
    return back_substitution(matrix, b)

def spp(matrix, b):
    spp_forward_elimination(matrix, b)
    return back_substitution(matrix, b)

def gaussian():
    '''
    Main function: 
    read input
    solve
    write output
    '''
    filename, method, precision = parse_args()
    datatype = get_datatype(precision)

    matrix, b = read_linear_system(filename, datatype)
    
    x = naive_gaussian(matrix, b) if method == "naive" else spp(matrix, b)

    write_solution(filename, x)

if (__name__ == "__main__"):
    gaussian()