datatype Tree = Leaf | Node(value: int, left: Tree, right: Tree)

// BST Predicates
ghost predicate IsBST(t: Tree)
  decreases t
{
  match t
  case Leaf => true
  case Node(v, l, r) =>
    IsBST(l) && IsBST(r) &&
    (forall x :: InTree(x, l) ==> x < v) &&
    (forall x :: InTree(x, r) ==> x > v)
}

predicate InTree(x: int, t: Tree)
  decreases t
{
  match t
  case Leaf => false
  case Node(v, l, r) =>
    x == v || InTree(x, l) || InTree(x, r)
}

// BST Implementation

function Contains(t: Tree, key: int): bool
  requires IsBST(t)
  decreases t
{
  match t
  case Leaf => false
  case Node(v, l, r) =>
    if key == v then true
    else if (key < v) then Contains(l, key)
    else Contains(r, key)
}

function Insert(t: Tree, x: int): Tree
  requires IsBST(t)
  ensures InTree(x, Insert(t, x))
{
  match t
  case Leaf => Node(x, Leaf, Leaf)
  case Node(v, l, r) => 
      if (x == v) then t
      else if (x < v) then Node(v, Insert(l, x), r)
      else Node(v, l, Insert(r, x))
}

function Min(t: Tree): int 
  requires IsBST(t)
  requires t.Node?
{
  match t
  case Node(v, Leaf, _) => v
  case Node(v, l, _) => Min(l)
}

  // Impl Correctness lemmas
lemma Contains_Correct(t: Tree, k: int)
  requires IsBST(t)
  ensures Contains(t, k) <==> InTree(k, t)
  decreases t
{}

lemma Insert_Preserves_BST(t: Tree, x: int)
  requires IsBST(t)
  ensures IsBST(Insert(t, x))  
{}

lemma Min_intbsolute(t: Tree)
  requires IsBST(t)
  ensures forall v :: InTree(v, t) ==> Min(t) < v || Min(t) == v
{}

lemma Insert_Preserves_Node(t: Tree, x: int)
  requires IsBST(t)
  requires t.Node?
  ensures Insert(t, x).Node?
{}


lemma Insert_New_Min(t: Tree, x: int)
  requires IsBST(t)
  requires t.Node?
  requires !Contains(t, x)
  requires x < Min(t)
  ensures Min(Insert(t, x)) == x
{}