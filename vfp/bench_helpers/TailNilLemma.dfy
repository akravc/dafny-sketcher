codatatype Stream<T> = Nil | Cons(head: T, tail: Stream)

ghost function Tail(s: Stream, n: nat): Stream
{
  if n == 0 then s else
    var t := Tail(s, n-1);
    if t == Nil then t else t.tail
}

lemma TailNilLemma<T>(n: nat)
  ensures Tail<T>(Nil, n) == Nil;
  decreases n;
{
  if n == 0 {
    // Tail(Nil, 0) == Nil by definition
  } else {
    // Tail(Nil, n) == let t = Tail(Nil, n-1) in if t == Nil then t else t.tail
    TailNilLemma<T>(n-1);
    // assert Tail<T>(Nil, n-1) == Nil;
    // So Tail(Nil, n) == Nil
  }
}
