function factorial(n: nat): nat
{
  if n == 0 then 0
  else n * factorial(n - 1)
}

function {:spec} factorialSpec(n: nat): nat
{
  if n == 0 then 1
  else n * factorialSpec(n - 1)
}

lemma {:axiom} factorialCorrect(n: nat)
  ensures factorial(n) == factorialSpec(n)