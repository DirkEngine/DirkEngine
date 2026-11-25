# Cleanup pass: judgement checks

The rules of the standard that no check can enforce. Each is a question
about the code you own: answer it with the `file:line` and the change you
propose. Fix what you can without changing behaviour; report the rest.

1. **Simplicity.** Which abstractions have one implementation, only forward
   calls, or could be a plain function or a macro? Which repeated
   boilerplate should be a macro?
2. **Duplication.** Which logic is implemented twice, including across
   crates and between languages, such as a Rust and a TypeScript SDK?
3. **Entry-point checks.** Which guard runs at the top of several handlers
   instead of once in middleware, an extractor or a constructor?
4. **Consistency.** Where does new code ignore a pattern the code around it
   already has: a second idiom for the same thing in one crate, or a
   structure unlike the existing code that does something similar?
5. **Lint exceptions.** Is each `#[expect]`, and each Dylint exception
   (`#[cfg_attr(dylint_lib = "standard_lints", expect(…))]`), on the
   smallest item, and does its reason say why the lint is wrong there rather
   than what the lint does? Is each `allow` in `[workspace.lints]` still
   worth allowing everywhere? Is each deviation in `standard.toml` scoped to
   the files or crates that need it?
6. **Invariants.** Is each `// INVARIANT:` comment still true, does the code
   still rely on it, and could a type or a test hold it instead?
7. **Workarounds.** Which code is contorted only to pass a rule: a function
   split only to stay under a cap, text hidden in a string literal, a
   `let _ =` or a rename only to silence a lint? Undo it, and report the
   rule as friction: it needs a deviation, a change, or a better fix.
8. **Errors.** Are failures a caller can tell apart separate variants, and
   indistinguishable ones merged? Does each rejection name the parameter or
   limit, without leaking a secret or an account's existence? Is an error
   logged more than once? Does each `expect` message say what held?
9. **Validation.** Which functions take a raw `String`, `&str` or number for
   something a validated type exists for, or re-check a validated value?
10. **Surfaces.** Which features are missing from a surface `AGENTS.md`
    lists, or lack their reverse (delete, disable, resume) or a way to see
    them? Which surface lists are still kept by hand?
11. **Tests.** Which tests check that something ran rather than its outcome,
    test the same behaviour at two layers or on several surfaces, could be a
    row in an existing table, or are smoke tests or "stays removed" tests?
    Which type rules a public API relies on lack a compile-fail test? Which
    behaviours lack a case in the repository's shared suites, such as SDK
    vectors or contract scenarios? Was any test changed only to pass?
12. **Docs.** Which docs name items, flags, settings or behaviour the code
    no longer has? Which doc comments narrate the code or justify it, rather
    than say what goes in and what comes out or happens? Which user-facing
    doc comments speak to developers? Which user-guide pages break the `docs`
    skill's writing rules or the feature outline? Which screenshots are
    stale?
13. **Dependencies.** Which dependencies keep default features they don't
    need? Which new platform library is missing from `deny.toml`'s bans?
    Which secret values sit in non-zeroizing types?
14. **Domain rules.** Does the code follow the domain rules `AGENTS.md` and
    the developer guide state, which no check enforces?
15. **Skills.** Does any skill copy its guide instead of pointing to it, or
    drive the app in a way that costs money or reaches production?

Who answers which:

- Each area agent: 1 to 11, 13 and 14, in its area.
- The docs lane: 12 and 15.
- The boundary lane: 2, 4 and 10 across areas.
