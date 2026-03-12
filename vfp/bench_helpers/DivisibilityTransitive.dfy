// Helper lemma: divisibility is transitive
lemma DivisibilityTransitive(a: int, b: int, n: int)
  requires a > 0 && b > 0
  requires exists k :: b == a * k  // a divides b
  requires exists m :: n == b * m  // b divides n
  ensures exists p :: n == a * p   // a divides n
{
  var k :| b == a * k;
  var m :| n == b * m;
  assert n == (a * k) * m;
  assert n == a * (k * m);
}
