idea = "Write (1) a datatype for arithmetic expressions, comparising constants, variables, and binary additions, (2) an evaluator function that takes an expression and an environment (function mapping variable to value) and return an integer value, (3) an optimizer function that removes addition by zero, (4) a lemma that ensures the optimizer preserves the semantics as defined by the evaluator."

spec = """datatype Expr =
  | Const(value: int)
  | Var(name: string)
  | Add(left: Expr, right: Expr)

type Environment = string -> int

function eval(e: Expr, env: Environment): int

function optimize(e: Expr): Expr

lemma {:axiom} optimizePreservesSemantics(e: Expr, env: Environment)
ensures eval(optimize(e), env) == eval(e, env)
"""

program_without_proof = """datatype Expr =
  | Const(value: int)
  | Var(name: string)
  | Add(left: Expr, right: Expr)

type Environment = string -> int

function eval(e: Expr, env: Environment): int
{
  match e
  case Const(val) => val
  case Var(name) => env(name)
  case Add(e1, e2) => eval(e1, env) + eval(e2, env)
}

function optimize(e: Expr): Expr
{
  match e
  case Add(e1, e2) =>
    var o1 := optimize(e1);
    var o2 := optimize(e2);
    if o2 == Const(0) then o1 else
    if o1 == Const(0) then o2 else Add(o1, o2)
  case _ => e
}

lemma {:axiom} optimizePreservesSemantics(e: Expr, env: Environment)
ensures eval(optimize(e), env) == eval(e, env)
"""

program_with_bugs = """datatype Expr =
  | Const(value: int)
  | Var(name: string)
  | Add(left: Expr, right: Expr)

predicate {:spec} optimal(e: Expr)
{
  match e
  case Add(Const(0), _) => false
  case Add(_, Const(0)) => false
  case Add(e1, e2) => optimal(e1) && optimal(e2)
  case _ => true
}

function optimize(e: Expr): Expr
{
  match e
  case Add(Const(0), e2) => optimize(e2)
  case Add(e1, Const(0)) => optimize(e1)
  case Add(e1, e2) => Add(optimize(e1), optimize(e2))
  case _ => e
}

lemma {:axiom} optimizeOptimal(e: Expr)
ensures optimal(optimize(e))
"""

spec_opt = """datatype Expr =
  | Const(value: int)
  | Var(name: string)
  | Add(left: Expr, right: Expr)

predicate {:spec} optimal(e: Expr)
{
  match e
  case Add(Const(0), _) => false
  case Add(_, Const(0)) => false
  case Add(e1, e2) => optimal(e1) && optimal(e2)
  case _ => true
}

function optimize(e: Expr): Expr

lemma {:axiom} optimizeOptimal(e: Expr)
ensures optimal(optimize(e))
"""

program_with_obvious_bug = """
function magic_number(): int {
    33
}

lemma {:axiom} magic_number_is_42()
ensures magic_number() == 42
"""

spec_all = """datatype Expr =
  | Const(value: int)
  | Var(name: string)
  | Add(left: Expr, right: Expr)

type Environment = string -> int

function eval(e: Expr, env: Environment): int

function optimize(e: Expr): Expr

lemma {:axiom} optimizePreservesSemantics(e: Expr, env: Environment)
ensures eval(optimize(e), env) == eval(e, env)

predicate {:spec} optimal(e: Expr)
{
  match e
  case Add(Const(0), _) => false
  case Add(_, Const(0)) => false
  case Add(e1, e2) => optimal(e1) && optimal(e2)
  case _ => true
}

lemma {:axiom} optimizeOptimal(e: Expr)
ensures optimal(optimize(e))
"""

def run(solver):
    if True:
        print('GIVEN PROGRAM WITH SUBTLE BUGS')
        solver(program_with_bugs)
    if False:
        print('GIVEN PROGRAM WITH BUGS')
        solver(program_with_obvious_bug)
    if False:
        print('GIVEN SPEC')
        solver(spec_all)
