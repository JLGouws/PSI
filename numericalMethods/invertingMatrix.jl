##
## SPD
##
## solving Ax = b
##
## iterative method
##
## Gaussian iteration
##
## x^*: true solution
##
## x: approximate solution
##
## e:= x - x^* (error)
##
## r:= b - Ax (residual)
##
## Ax = b
## G:= (1/2)x^T A x - b^T x
##
## grad G = Ax - b = - r

## overshoot then overshoot
## conjugate gradient only look in one direction

function conjugateDescent(A, b, N)
  sol = A \ b
  x = zero(b)
  r_0 = r_1 = b - A * x
  p = r_0
  beta = 0
  for n in 1:N
    alpha =  (r_0' * r_0 ) / (p' * A * p)#(x' * A * r - b' * r) / (r' * A * r)
    x += alpha * p
    r_1 -= alpha * A * p
    beta = r_1' * r_1 / (r_0' * r_0)
    p = r_1 + beta * p
    r_0 = r_1
#    x += 0.05 * r 
    println("e: " * string(sum((x - sol) .* (x - sol))))
    println()

  end
  return x
end

function gradientDescent(A, b, N)
  x = zero(b)
  sol = A \ b
  for n in 1:N
    r = b - A * x
    s = A * r
    println("r: " * string(sum(r .* r)))
    alpha =  (r' * s ) / (s' * s)#(x' * A * r - b' * r) / (r' * A * r)
    println(alpha)
    println("e: " * string(sum((x - sol) .* (x - sol))))
    println()
    x += alpha * r 
#    x += 0.05 * r 

  end
  return x
end

A = randn(10, 10)
A = A + A'
A = A * A
println(gradientDescent(A, randn(10) , 1000))
