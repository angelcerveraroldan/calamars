use std::{
    fmt::Debug,
    fs::{self, File},
    process::Output,
    str::FromStr,
};

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
    #[token("else")]   Else,
    #[token("let")]    Let,
    #[token("return")] Return,
    #[token("module")] Module,
    #[token("import")] Import,
    #[token("struct")] Struct,
    #[token("trait")]  Trait,
    #[token("and")]    And,
    #[token("or")]     Or,
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

    #[token("+")] Plus,
    #[token("-")] Minus,
    #[token("*")] Star,
    #[token("/")] Slash,
    #[token("=")] Equal,
    #[token("<")] Less,
    #[token(">")] Greater,

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
    EOF,
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

