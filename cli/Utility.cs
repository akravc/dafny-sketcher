using System.Linq;
using Microsoft.Dafny;

namespace DafnySketcherCli {
  public static class Utility {

    /// <summary>
    /// Recursively collects all (member, modulePath) pairs from the default module
    /// and all nested/refining modules.
    /// </summary>
    public static IEnumerable<(MemberDecl member, string modulePath)> GetAllMembers(Microsoft.Dafny.Program resolvedProgram) {
      if (resolvedProgram.DefaultModuleDef is DefaultModuleDefinition defaultModule) {
        foreach (var pair in GetMembersRecursive(defaultModule, "")) {
          yield return pair;
        }
      }
    }

    private static IEnumerable<(MemberDecl member, string modulePath)> GetMembersRecursive(ModuleDefinition module, string prefix) {
      // Yield accessible members from this module (filter to MemberDecl)
      foreach (var (decl, _) in module.AccessibleMembers) {
        if (decl is MemberDecl member) {
          yield return (member, prefix);
        }
      }
      // Recurse into nested modules (including refinements)
      // Track visited modules to avoid infinite loops
      var visited = new HashSet<ModuleDefinition> { module };
      foreach (var topDecl in module.TopLevelDecls) {
        // Check ModuleDecl which wraps module definitions (literal modules, imports, refinements)
        if (topDecl is ModuleDecl moduleDecl) {
          var sig = moduleDecl.AccessibleSignature(false);
          if (sig?.ModuleDef != null && !visited.Contains(sig.ModuleDef)) {
            visited.Add(sig.ModuleDef);
            var subPrefix = string.IsNullOrEmpty(prefix) ? moduleDecl.Name : prefix + "." + moduleDecl.Name;
            foreach (var pair in GetMembersRecursive(sig.ModuleDef, subPrefix)) {
              yield return pair;
            }
          }
        }
      }
    }

    /// <summary>
    /// Like GetAllMembers but deduplicates by source position (startLine, startColumn).
    /// When the same declaration appears in multiple modules (via inheritance/refinement),
    /// only the first occurrence (from the most specific/concrete module) is returned.
    /// </summary>
    public static IEnumerable<(MemberDecl member, string modulePath)> GetAllMembersDeduped(Microsoft.Dafny.Program resolvedProgram) {
      var seen = new HashSet<(int line, int col)>();
      foreach (var (member, modulePath) in GetAllMembers(resolvedProgram)) {
        var key = (member.StartToken.line, member.StartToken.col);
        if (seen.Add(key)) {
          yield return (member, modulePath);
        }
      }
    }

    public static Method? GetMethodByName(Microsoft.Dafny.Program resolvedProgram, string name) {
      // First try unqualified name (default module)
      if (resolvedProgram.DefaultModuleDef is DefaultModuleDefinition defaultModule) {
        foreach (var (member, _) in defaultModule.AccessibleMembers) {
          var method = member as Method;
          if (method != null && method.Name == name) {
            return method;
          }
        }
      }
      // Then try all modules (qualified: "Module.Method" or just "Method" in any module)
      foreach (var (member, modulePath) in GetAllMembers(resolvedProgram)) {
        var method = member as Method;
        if (method == null) continue;
        var qualifiedName = string.IsNullOrEmpty(modulePath) ? method.Name : modulePath + "." + method.Name;
        if (qualifiedName == name || method.Name == name) {
          return method;
        }
      }
      return null;
    }

    public static Method GetEnclosingMethodByPosition(Microsoft.Dafny.Program resolvedProgram, int line, int col) {
      // Search default module first
      if (resolvedProgram.DefaultModuleDef is DefaultModuleDefinition defaultModule) {
        foreach (var (member, _) in defaultModule.AccessibleMembers) {
          var method = member as Method;
          if (method != null && IsPositionInRange(method.StartToken, method.EndToken, line, col)) {
            return method;
          }
        }
      }
      // Then search all modules
      foreach (var (member, _) in GetAllMembers(resolvedProgram)) {
        var method = member as Method;
        if (method != null && IsPositionInRange(method.StartToken, method.EndToken, line, col)) {
          return method;
        }
      }
      return null;
    }
    private static bool IsPositionInRange(IOrigin startToken, IOrigin endToken, int line, int col) {
      return line >= startToken.line && line <= endToken.line &&
            (line != startToken.line || col >= startToken.col) &&
            (line != endToken.line || col <= endToken.col);
    }

    /// <summary>
    /// Returns a copy of the source with the method body emptied.
    /// This is useful for induction sketchers that need to see an empty body
    /// to make the correct choice between structural and rule induction.
    /// </summary>
    public static string EmptyMethodBody(string source, Method method) {
      if (method.Body == null) {
        return source; // Already empty
      }

      // Use the exact token positions to locate the body content
      var bodyStartToken = method.Body.StartToken;
      var bodyEndToken = method.Body.EndToken;

      // Calculate character positions in the source string
      var lines = source.Split('\n');
      int startPos = 0;

      // Find the start position of the body's opening brace
      for (int i = 0; i < bodyStartToken.line - 1; i++) {
        startPos += lines[i].Length + 1; // +1 for newline
      }
      startPos += bodyStartToken.col;

      // Find the end position of the body's closing brace
      int endPos = 0;
      for (int i = 0; i < bodyEndToken.line - 1; i++) {
        endPos += lines[i].Length + 1; // +1 for newline
      }
      endPos += bodyEndToken.col;

      // Build result: everything before body + "{" + "\n" + "}" + everything after body
      var result = new System.Text.StringBuilder();
      result.Append(source.Substring(0, startPos)); // Everything before the body opening brace
      result.Append("{\n\n}"); // Empty body
      if (endPos + 1 < source.Length) {
        result.Append(source.Substring(endPos + 1)); // Everything after the body closing brace
      }

      return result.ToString();
    }
  }
}
