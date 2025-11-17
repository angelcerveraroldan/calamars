//! Tokenizer for Calamars language using [Logos](https://github.com/maciejhirsz/logos)

use std::{
    convert::identity,
    fmt::Debug,
    fs::{self, File},
    process::Output,
    str::FromStr,
};

use crate::syntax::span::Span;
use std::fmt;

use logos::{Lexer, Logos};

/// Helper function to parse numbers
///
/// This ignores underscores
fn parse_num<T>(lex: &mut Lexer<Token>) -> Option<T>
where
    T: FromStr,
    <T as FromStr>::Err: Debug,
{
    lex.slice()
        .chars()
        .filter(|c| *c != '_')
        .collect::<String>()
        .parse::<T>()
        .ok()
}

fn parse_binary(lex: &mut Lexer<Token>) -> Option<i64> {
    i64::from_str_radix(&lex.slice()[2..], 2).ok()
}

fn parse_doc_comment(lex: &mut Lexer<Token>) -> Option<String> {
    let slice = lex.slice();
    Some(slice[3..slice.len() - 3].to_string())
}

#[allow(dead_code)]
#[rustfmt::skip]
#[derive(Debug, PartialEq, Clone, Logos)]
#[logos(skip r"[ \t\n\f]+")]
#[logos(subpattern decimal = r"[0-9][_0-9]*")]
#[logos(subpattern binary  = r"b_[0-1][_0-1]*")]
pub enum Token {
    #[token("def")]    Def,
    #[token("mut")]    Mut,
    #[token("given")]  Given,
    #[token("match")]  Match,
    #[token("if")]     If,
    #[token("then")]   Then,
    #[token("else")]   Else,
    #[token("let")]    Let,
    #[token("return")] Return,
    #[token("module")] Module,
    #[token("import")] Import,
    #[token("struct")] Struct,
    #[token("trait")]  Trait,
    #[token("and")]    And,
    #[token("or")]     Or,
    #[token("xor")]    Xor,
    #[token("not")]    Not,
    #[token("enum")]   Enum,
    #[token("val")]    Val,
    #[token("var")]    Var,
    #[token("true")]   True,
    #[token("false")]  False,

    #[token("(")] LParen,
    #[token(")")] RParen,
    #[token("{")] LBrace,
    #[token("}")] RBrace,
    #[token("[")] LBracket,
    #[token("]")] RBracket,
    #[token(".")] Dot,
    #[token(",")] Comma,
    #[token(":")] Colon,
    #[token(";")] Semicolon,

    #[token("->")] Arrow,
    #[token("=>")] FatArrow,
    #[token("|>")] PipeOp,
    #[token("==")] EqualEqual,
    #[token("!=")] NotEqual,
    #[token("<=")] LessEqual,
    #[token(">=")] GreaterEqual,
    #[token("++")] Concat,

    #[token("+")] Plus,
    #[token("-")] Minus,
    #[token("*")] Star,
    #[token("^")] Pow,
    #[token("/")] Slash,
    #[token("=")] Equal,
    #[token("<")] Less,
    #[token(">")] Greater,
    #[token("%")] Mod,

    #[regex(r"--[^\n]*", priority = 2)]
    LineComment,

    #[regex(r"--\*([^*]|\*[^-])*\*--", parse_doc_comment)]
    DocComment(String),

    #[regex(r"[A-Za-z_][A-Za-z0-9_]*", callback = |lex| lex.slice().to_string())]
    Ident(String),

    #[regex(r"[+-]?(?&decimal)", parse_num::<i64>)]
    #[regex(r"(?&binary)", parse_binary)] 
    Int(i64),

    #[regex(r"[+-]?(?&decimal)\.(?&decimal)", parse_num::<f64>)]
    Float(f64),

    #[regex(r#"'(\\.|[^\\'])'"#, callback = |lex| {
        let s = lex.slice();
        let inner = &s[1..s.len()-1]; // strip quotes
        inner.chars().next().unwrap()
    })]
    Char(char),

    #[regex(r#""([^"\\]|\\.)*""#, callback = |lex| {
        let s = lex.slice();
        s[1..s.len()-1].to_string() // naive unescape
    })]
    String(String),

    Error,
    EOF,
}

impl Token {
    pub fn tokenize_line<'a>(s: &'a str) -> Vec<Token> {
        let mut lex = Token::lexer(&s);
        let mut tokens = vec![];
        for token in lex.into_iter() {
            if let Ok(t) = token {
                tokens.push(t);
            }
        }
        tokens
    }

    pub fn tokens_spanned_stream<'a>(source: &'a str) -> Vec<(Token, logos::Span)> {
        Token::lexer(source)
            .spanned()
            .map(|(token, span)| match token {
                Ok(tok) => (tok, span),
                Err(()) => (Token::Error, span),
            })
            .filter(|(token, _)| *token != Token::LineComment)
            .collect()
    }
}

impl fmt::Display for Token {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fn escape_str(s: &str) -> String {
            let mut out = String::with_capacity(s.len() + 2);
            for ch in s.chars() {
                match ch {
                    '\\' => out.push_str(r"\\"),
                    '"' => out.push_str(r#"\""#),
                    '\n' => out.push_str(r"\n"),
                    '\r' => out.push_str(r"\r"),
                    '\t' => out.push_str(r"\t"),
                    c => out.push(c),
                }
            }
            out
        }

        fn escape_char(c: char) -> String {
            match c {
                '\\' => r"\\".to_string(),
                '\'' => r"\'".to_string(),
                '\n' => r"\n".to_string(),
                '\r' => r"\r".to_string(),
                '\t' => r"\t".to_string(),
                c => c.to_string(),
            }
        }

        match self {
            // keywords
            Token::Def => write!(f, "def"),
            Token::Mut => write!(f, "mut"),
            Token::Given => write!(f, "given"),
            Token::Match => write!(f, "match"),
            Token::If => write!(f, "if"),
            Token::Then => write!(f, "then"),
            Token::Else => write!(f, "else"),
            Token::Let => write!(f, "let"),
            Token::Return => write!(f, "return"),
            Token::Module => write!(f, "module"),
            Token::Import => write!(f, "import"),
            Token::Struct => write!(f, "struct"),
            Token::Trait => write!(f, "trait"),
            Token::And => write!(f, "and"),
            Token::Or => write!(f, "or"),
            Token::Xor => write!(f, "xor"),
            Token::Not => write!(f, "not"),
            Token::Enum => write!(f, "enum"),
            Token::Val => write!(f, "val"),
            Token::Var => write!(f, "var"),
            Token::True => write!(f, "true"),
            Token::False => write!(f, "false"),

            // punctuation
            Token::LParen => write!(f, "("),
            Token::RParen => write!(f, ")"),
            Token::LBrace => write!(f, "{{"),
            Token::RBrace => write!(f, "}}"),
            Token::LBracket => write!(f, "["),
            Token::RBracket => write!(f, "]"),
            Token::Dot => write!(f, "."),
            Token::Comma => write!(f, ","),
            Token::Colon => write!(f, ":"),
            Token::Semicolon => write!(f, ";"),

            // operators
            Token::Arrow => write!(f, "->"),
            Token::FatArrow => write!(f, "=>"),
            Token::PipeOp => write!(f, "|>"),
            Token::EqualEqual => write!(f, "=="),
            Token::NotEqual => write!(f, "!="),
            Token::LessEqual => write!(f, "<="),
            Token::GreaterEqual => write!(f, ">="),
            Token::Concat => write!(f, "++"),
            Token::Plus => write!(f, "+"),
            Token::Minus => write!(f, "-"),
            Token::Star => write!(f, "*"),
            Token::Slash => write!(f, "/"),
            Token::Equal => write!(f, "="),
            Token::Less => write!(f, "<"),
            Token::Greater => write!(f, ">"),
            Token::Pow => write!(f, "^"),
            Token::Mod => write!(f, "%"),

            // comments
            Token::LineComment => write!(f, "--…"),
            Token::DocComment(s) => write!(f, "--*{}*--", s),

            // identifiers & literals
            Token::Ident(s) => write!(f, "{s}"),
            Token::Int(i) => write!(f, "{i}"),
            Token::Float(x) => write!(f, "{x}"),
            Token::Char(c) => write!(f, "'{}'", escape_char(*c)),
            Token::String(s) => write!(f, "\"{}\"", escape_str(s)),

            Token::Error => write!(f, "<ERROR>"),
            Token::EOF => write!(f, "<eof>"),
        }
    }
}

#[cfg(test)]
mod test_tokens {
    use super::*;

    #[test]
    fn test_number_parsing() {
        let mut lex = Token::lexer("-5_3_0 -34.0 -34 51.34444 b_11");

        assert_eq!(lex.next(), Some(Ok(Token::Int(-530))));
        assert_eq!(lex.next(), Some(Ok(Token::Float(-34.0))));
        assert_eq!(lex.next(), Some(Ok(Token::Int(-34))));
        assert_eq!(lex.next(), Some(Ok(Token::Float(51.34444))));
        assert_eq!(lex.next(), Some(Ok(Token::Int(3))))
    }

    #[test]
    fn test_string_parsing() {
        let mut lex = Token::lexer("\"Hello\"");
        assert_eq!(lex.next().unwrap(), Ok(Token::String("Hello".to_string())));
    }

    #[test]
    fn test_comment_parsing() {
        let mut lex = Token::lexer("-- And then this and that \n--*This is now a doc comment!*--");

        assert_eq!(lex.next().unwrap(), Ok(Token::LineComment));
        assert_eq!(
            lex.next().unwrap(),
            Ok(Token::DocComment("This is now a doc comment!".to_string()))
        );
    }
}
