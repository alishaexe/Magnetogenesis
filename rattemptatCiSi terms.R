library(pracma)

# # Sine Integral Si(x)
# Si <- function(x) {
#   integrand <- function(t) {return(sin(t) / t)}
#   integral <- integrate(integrand, lower = 0, upper = x, subdivisions = 1000L, rel.tol = 1e-8)$value
#   return(integral())
# }
# 
# # Cosine Integral Ci(x)
# Ci <- function(x) {
#   gamma <- 0.5772156649
#   logx <- log(x)
#   integrand <- function(t){return((1 - cos(t)) / t)}
#   integral <- integrate(integrand, lower = 0, upper = x, subdivisions = 1000L, rel.tol = 1e-8)$value
#   return (gamma + logx - integral)
# }

k <- logseq(0.01, 10000, n = 1000)
xstar <- 1e-3

term <- k**2*Ci(k*xstar)**2+(pi/2+Si(k*xstar))**2
plot(k, term,log = "xy", lty = 1, type="l")
# lines(k, term, lty = 1)

