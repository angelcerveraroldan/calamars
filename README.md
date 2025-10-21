# 🐙 Calamars 🌕

A fast, functional general purpose programming language with Typst documentation!

<img src="./docs/calamars.png" alt="Calamars Mascot" height="160">

# Syntax Highlighting

To play around with Calamars syntax, you can use the following code in you vim config.
This is just a temporary solution, since the language is changing very fast, and the syntax is
not fully stable yet.

```
" Keywords
syntax keyword calamarsKeyword def val var mut struct enum match import module

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

