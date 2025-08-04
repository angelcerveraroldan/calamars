use logos::Logos;

#[derive(Debug, PartialEq, Logos)]
/// Calamars tokens
pub enum TokenType {
    // Keywords
    Def,
    Mut,
    Given,
    Match,
    If,
    Else,
    Let,
    Return,
    Module,
    Import,
    Struct,
    Trait,
    And,
    Or,
    Not,
    Enum,

    // Single chars
    LParen,
    RParen,
    LBrace,
    RBrace,
    LBracket,
    RBracket,
    Dot,
    Comma,
    Colon,
    Semicolon,

    // -> and =>
    Arrow,
    FatArrow,

    // Symbols
    Plus,
    Minus,
    Star,
    Slash,
    Equal,
    EqualEqual,
    NotEqual,
    Less,
    LessEqual,
    Greater,
    GreaterEqual,

    // Comments
    LineComment,
    BlockComment,
    DocComment,

    // Primitives
    Ident(String),
    Int(i64),
    Float(f64),
    Char(char),
    String(String),
    True,
    False,

    // End of the file!
    EOF,
}
