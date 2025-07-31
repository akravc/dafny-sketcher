abstract module BSTSpec {

  // Types and equality
  type A(!new) // not a heap datatype for soundness of BST predicate
  datatype Tree = Leaf | Node(value: A, left: Tree, right: Tree)

  function lessThan(v1: A, v2: A): bool
  function greaterThan(v1: A, v2: A): bool
  function eq(v1: A, v2: A): bool

  // BST Predicates
  ghost predicate IsBST(t: Tree)
    decreases t
  {
    match t
    case Leaf => true
    case Node(v, l, r) =>
      IsBST(l) && IsBST(r) &&
      (forall x :: InTree(x, l) ==> lessThan(x, v)) &&
      (forall x :: InTree(x, r) ==> greaterThan(x, v))
  }

  predicate InTree(x: A, t: Tree)
    decreases t
  {
    match t
    case Leaf => false
    case Node(v, l, r) =>
      eq(x, v) || InTree(x, l) || InTree(x, r)
  }

  // BST Interface to be Implemented
  function Contains(t: Tree, key: A): bool
    requires IsBST(t)
    decreases t

  function Insert(t: Tree, x: A): Tree
    requires IsBST(t)
    ensures InTree(x, Insert(t, x))
  
  function Min(t: Tree): A
    requires IsBST(t)
    requires t.Node?
}

module BSTImpl refines BSTSpec{
  type A = int

  function lessThan(v1: A, v2: A): bool { v1 < v2 }
  function greaterThan(v1: A, v2: A): bool { v1 > v2 }
  function eq(v1: A, v2: A): bool { v1 == v2 }

  function Contains(t: Tree, key: A): bool
  {
    match t
    case Leaf => false
    case Node(v, l, r) =>
      if eq(key, v) then true
      else if lessThan(key, v) then Contains(l, key)
      else Contains(r, key)
  }

  function Insert(t: Tree, x: A): Tree
  {
    match t
    case Leaf => Node(x, Leaf, Leaf)
    case Node(v, l, r) => 
        if eq(x, v) then t
        else if lessThan(x, v) then Node(v, Insert(l, x), r)
        else Node(v, l, Insert(r, x))
  }

  function Min(t: Tree): A 
  {
    match t
    case Node(v, Leaf, _) => v
    case Node(v, l, _) => Min(l)
  }

    // Impl Correctness lemmas
  lemma Contains_Correct(t: Tree, k: A)
    requires IsBST(t)
    ensures Contains(t, k) <==> InTree(k, t)
    decreases t
  {}

  lemma Insert_Preserves_BST(t: Tree, x: A)
    requires IsBST(t)
    ensures IsBST(Insert(t, x))  
  {}

  lemma Min_Absolute(t: Tree)
    requires IsBST(t)
    ensures forall v :: InTree(v, t) ==> lessThan(Min(t), v) || eq(Min(t), v)
  {}

  lemma Insert_Preserves_Node(t: Tree, x: A)
    requires IsBST(t)
    requires t.Node?
    ensures Insert(t, x).Node?
  {}

  
  lemma Insert_New_Min(t: Tree, x: A)
    requires IsBST(t)
    requires t.Node?
    requires !Contains(t, x)
    requires lessThan(x, Min(t))
    ensures eq(Min(Insert(t, x)), x)
  {}
}
