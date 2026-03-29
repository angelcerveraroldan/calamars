<h1>🐙 Calamars 🌕</h1>


Calamars is an experimental programming language that’s still very
much in development, but with some clear goals:


- **Typst-powered doc blocks**. Documentation and scientific writing
    should feel effortless. You can include Typst right inside your
    code for beautiful, math-friendly docs.
- **Safe mutability**. Nothing changes unless you mean it. Every
    mutation is explicit and visible.
- **Clean, functional-inspired syntax**. Simple constructs, first
    class functions, and a style that encourages composition and
    clarity.
- **Compiled _and_ interpreted**. Calamars is meant to run both as a
    compiled language and inside things such as notebooks. This should
    allow for easy testing and exploring, and for fast binaries.

# Usage

Today the CLI parses Calamars source to AST, lowers to HIR,
type-checks, and prints MIR. There is a basic VM that can be used to
run some calamars files, see the `testing/` directory for examples.

## Quickstart

Run the some code that contains no errors, you should see the MIR in
text form printed:

`cargo run -p calamars_cli -- build --emit-mir <path_to_calamars_file>`

If you would like to actually run calamars code, then you can use the
VM:

`cargo run -p calamars_cli -- build --run-vm <path_to_calamars_file>`

# Pretty Error Reporting

Currently supports pretty error reporting using [`ariadne`](https://github.com/zesterer/ariadne).

```
Error: Wrong type returned
   ╭─[ tmp:1:13 ]
   │
 5 │ def foo i = if foo i then 0 else "hello"
   │                ──┬──  
   │                  ╰──── Expected to find `Bool` but found `Int`
───╯
Error: Both branches in if statement must return the same type
   ╭─[ tmp:1:13 ]
   │
 5 │ def foo i = if foo i then 0 else "hello"
   │                           ┬      ───┬───  
   │                           ╰─────────────── Int returned here
   │                                     │     
   │                                     ╰───── String returned here
───╯
```

# Branch Structure

Branches are named as `<prefix>/<name>`. When creating a new branch,
choose a name that describes the changes being made, to choose the
prefix, reference the follwing table:

| Prefix   | Usage                         |
|----------|-------------------------------|
| feat     | A new feature                 |
| fix      | Fix a bug or something else   |
| impr     | Improvements                  | 
| docs     | Add documentation             |
| refactor | Reorganization                |
| test     | Add new tests                 |
| chore    | Dependencies, formatting, ... |

# File structure

A Calamars project has to have the following structure:

```
├── project.cm
└── src
    └── main.cm
```

Where `project.cm` is the config file for the project.

# Roadmap

Currently working:
- Lexer and parser to AST
- HIR lowering with identifier resolution
- Type checker on the HIR
- Mir lowerer
- Mir to text
- Pretty diagnostics
- Virtual Machine (limited functionality)

Things that I want to work on soon, but are not yet implemented:
- Compile down to a binary (For now, a VM is supported, later
  something like cranelift will be used for this)
- Imports / modules (Currently, we just support one file, need to
  think about how to handle many files)

# Syntax Highlighting

To play around with Calamars syntax, you can use the following code in
you vim config. This is just a temporary solution, since the language
is changing very fast, and the syntax is not fully stable yet.

```vim
" Keywords
syntax keyword calamarsKeyword def typ if then else or and xor
" Types (after :: in type sigs)
syntax match calamarsType /\v::\s*\zs[A-Z][a-zA-Z0-9_]*/
" Strings
syntax region calamarsString start=/"/ skip=/\\"/ end=/"/
" Comments (single-line)
syntax match calamarsComment /^--.*/
" Integer (e.g., 123, 0, 42)
syntax match calamarsNumber /\v\<\d+\>/
" Float (e.g., 3.14, 2.0, 0.001)
syntax match calamarsFloat /\v\<\d+\.\d+\>/
" Function names after 'def'
syntax match calamarsFunction /\vdef\s+\zs\w+/
syntax match calamarsFuncDecl /\vdef\s+\zs\w+/
syntax match calamarsFuncCall /\<\h\w*\>\ze\s*(/
" Link to highlight groups
highlight link calamarsKeyword Keyword
highlight link calamarsType Type
highlight link calamarsString String
highlight link calamarsComment Comment
highlight link calamarsFunction Function
highlight link calamarsFuncDecl Function
highlight link calamarsFuncCall Identifier
highlight link calamarsNumber Number
highlight link calamarsFloat Float
```

If you use emacs, you can use the following simple `calamars-mode`,
which should provide syntax highlighting. Once again, this is for 
the time being, a more compleme emacs mode will be developed at some point.

```elisp
;;; calamars-mode.el --- major mode for calamars -*- lexical-binding: t; -*-
;;; This is a very simple mode that basically only provides basic syntax highlighting,
;;; a more complete major mode will be released once the language is actually usable :)

(defconst cm/comment-prefix "--")

;; Set of calamars keywords
(defconst cm/keywords
  '("typ"    "def"   "mut"    "given"  "match"
	"else"   "let"   "return" "module" "import"
	"trait"  "and"   "or"     "xor"    "not"
	"struct" "enum"  "true"   "false"  "if"
	"then"   "match" "case"))

;; Basic calamars types
(defconst cm/primitive-types
  '("Int" "Unit" "String" "Bool"))

(defconst cm/font-lock-def
  (append
   (mapcar (lambda (x)
             (cons (concat "\\_<" (regexp-quote x) "\\_>")
                   'font-lock-keyword-face))
           cm/keywords)
   (mapcar (lambda (x)
             (cons (concat "\\_<" (regexp-quote x) "\\_>")
                   'font-lock-type-face))
           cm/primitive-types)))

(define-derived-mode calamars-mode
  prog-mode
  "calamars"
  "Major mode for calamars files."
  (setq font-lock-defaults '((cm/font-lock-def)))
  (setq-local comment-start cm/comment-prefix))
```

# Syntax

Note: This section will use syntax that is not yet supported by the
language, but which will be supported at a later date.

When defining functions, you will write two lines, one with the type
information and docs, and another with the actual function definition.

```
typ <name> :: <type>
def <name> <inputs>* = <expression>
```

For example:

```
typ add :: Int -> Int -> Int
def add x y = x + y
```

If you have not used functional programming languages before, you may
ask why the type signature has two arrows, rather than looking
something like `(Int, Int) -> Int`. The reason for this, is that we
will be [currying](https://en.wikipedia.org/wiki/Currying) by default.

Here is an example of using currying:

```
typ addFive :: Int -> Int
def addFive = add 5
```

The above would be equivalent to:

```
typ addFive :: Int -> Int
def addFive x = add 5 x
```

## Expressions

Basically anything is an expression; with the exception of type
definitions / declarations. Becausee of this, you can write something
like:

```
typ divide :: Int -> Int -> Int
def divide x y =
	-- Returning -1 does not make a lot of sense, but this is just an
	-- example
	if y == 0 then -1
	else Some (x/y)
```

You can also do more "procedural" style coding by using blocks,
similar to rusts, where the last line of the block is what is
returned.

```
typ someWeirdFunction :: Int -> Int -> Int
def someWeirdFunction x y = {
	def result :: Int = x / y;
	if isEven result then result
	else None
}
```

Above, you may have noticed some sugar! When inside blocks, you dont
need to write the type and then the declaration, you can do both in
one line.
