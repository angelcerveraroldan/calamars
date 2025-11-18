<h1>🐙 Calamars 🌕</h1>


Calamars is an experimental programming language that’s still very much in development, but with some clear goals:


- **Typst-powered doc blocks**. Documentation and scientific writing should feel effortless. You can include Typst right inside your code for beautiful, math-friendly docs.
- **Safe mutability**. Nothing changes unless you mean it. Every mutation is explicit and visible.
- **Clean, functional-inspired syntax**. Simple constructs, first class functions, and a style that encourages composition and clarity.
- **Compiled _and_ interpreted**. Calamars is meant to run both as a compiled language and inside things such as notebooks. This should allow for easy testing and exploring, and for fast binaries.

# Usage

Today the CLI parses Calamars source to AST, lowers to HIR, type-checks, and prints MIR. There is no native codegen/backend yet (but it is coming soon!).

## Quickstart

Run the some code that contains no errors, you should see the MIR in text form printed:

`cargo run -p calamars_cli -- build --mir docs/examples/minimal.cm`

If you want to check out the error reporting capabilities, run the following:

`cargo run -p calamars_cli -- build --mir docs/examples/error_reporting.cm`


# Pretty Error Reporting

Currently supports pretty error reporting using [`ariadne`](https://github.com/zesterer/ariadne).

<img src="./docs/error_reporting.png" alt="Error reporting example">

# Branch Structure

Branches are named as `<prefix>/<name>`. When creating a new banch, choose a name that describes the changes being made, to choose the prefix, 
reference the follwing table:

| Prefix   | Usage                         |
|----------|-------------------------------|
| feat     | A new feature                 |
| fix      | Fix a bug or something else   |
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

Things that I want to work on soon, but are not yet implemented:
- Codegen / backend (This is top-priority, of course! Likely, cranelift will be used for this)
- Imports / modules (Currently, we just support one file, need to think about how to handle many files)
- Some expressions (For example, blocks are not yet supported, this should also come really soon, top priority!)

# Syntax Highlighting

To play around with Calamars syntax, you can use the following code in you vim config.
This is just a temporary solution, since the language is changing very fast, and the syntax is
not fully stable yet.

```vim
" Keywords
syntax keyword calamarsKeyword def val var mut struct enum match import module or and xor

" Types (after colon, like : String)
syntax match calamarsType /\v:\s*\zs[A-Z][a-zA-Z0-9_]*/

" Type parameters like Option[A]
syntax match calamarsType /\v\[[A-Z][a-zA-Z0-9_, ]*\]/

" Strings
syntax region calamarsString start=/"/ skip=/\\"/ end=/"/

" Comments (single-line and doc blocks)
syntax match calamarsComment /^--.*/ contains=calamarsDoc
syntax region calamarsDoc start=/--\*/ end=/\*--/

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
highlight link calamarsDoc Comment
highlight link calamarsFunction Function
highlight link calamarsFuncDecl Function
highlight link calamarsFuncCall Identifier
highlight link calamarsNumber Number
highlight link calamarsFloat Float
```

