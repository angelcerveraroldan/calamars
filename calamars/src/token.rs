use std::{
    iter::Peekable,
    str::{CharIndices, Chars},
};

use crate::errors::LexErrorKind;

#[allow(dead_code)]
#[rustfmt::skip]
#[derive(Debug, PartialEq)]
/// Calamars Toke types
pub enum TokenType {
    // Keywords
    Def, Mut, Given, Match, If, Else,
    Let, Return, Module, Import, Struct, 
    Trait, And, Or, Not, Enum,

    // Single chars
    LParen, RParen, LBrace, RBrace,
    LBracket, RBracket, Dot, Comma,
    Colon, Semicolon,

    // -> and =>
    Arrow, FatArrow,

    // Symbols
    Plus, Minus, Star, Slash, Equal, 
    EqualEqual, NotEqual, Less, LessEqual,
    Greater, GreaterEqual,

    // Comments
    LineComment, BlockComment, DocComment,

    // Primitives
    Ident, Int, Float, Char, String, True, False,

    // End of the file!
    EOF,
}

/// A token
#[derive(Debug)]
pub struct Token {
    /// The type which is tokenized
    ttype: TokenType,
    /// The span of the source code which is taken up by this token
    source: Span,
}

/// A positon in the source code
#[derive(Debug, Clone)]
pub struct Position {
    /// Number of bytes offset
    offset: usize,

    /// Line number
    line: usize,
    /// Column
    col: usize,
}

impl Default for Position {
    fn default() -> Self {
        Self {
            offset: 0,
            line: 1,
            col: 1,
        }
    }
}

impl Position {
    pub fn next_line(&self) -> Self {
        Self {
            offset: self.offset + 1,
            line: self.line + 1,
            col: 1,
        }
    }

    pub fn process(&self, char: u8) -> Self {
        match char {
            b'\n' => self.next_line(),
            _ => {
                let mut n = self.clone();
                n.col += 1;
                n.offset += 1;
                n
            }
        }
    }
}

/// A Span of the source code
#[derive(Default, Debug)]
pub struct Span {
    /// Fist bit in the span
    from: Position,
    /// Number of bits
    bytes: usize,
}

/// Hold the source code, and generates tokens
///
/// The source code must be ascii
#[derive(Debug)]
pub struct Scanner<'a> {
    source: &'a [u8],
    tokens: Vec<Token>,
    current_pos: Position,
}

impl<'a> Scanner<'a> {
    /// Generate a new scanner from the source code as a string
    fn new(source_code: &'a str) -> Self {
        Self {
            source: source_code.as_bytes(),
            tokens: vec![],
            current_pos: Position::default(),
        }
    }

    fn peek(&self) -> Option<u8> {
        self.source.get(self.current_pos.offset).copied()
    }

    fn peek_n(&self, n: usize) -> Option<&'a [u8]> {
        let start = self.current_pos.offset;
        let ending = start + n;
        if self.source.len() < ending {
            return None;
        }
        Some(&self.source[start..ending])
    }

    fn advance(&mut self) {
        if let Some(c) = self.peek() {
            self.current_pos = self.current_pos.process(c);
        }
    }

    // ################# //
    // HANDLE WHITESPACE //
    // ################# //

    /// Is the next character whitespace
    fn is_whitespace(&self) -> bool {
        let ws = [' ', '\t'];
        match self.peek() {
            None => false,
            Some(c) => ws.contains(&(c as char)),
        }
    }

    /// Skip all leading whitespace
    fn skip_whitespace(&mut self) {
        while self.is_whitespace() {
            self.advance();
        }
    }

    fn tokenize_keywords(&mut self) {}
}

#[cfg(test)]
mod test_scanner {
    use crate::token::Scanner;

    fn sample_scanner() -> Scanner<'static> {
        Scanner::new("     this is a test!")
    }

    #[test]
    fn test_peekn() {
        let mut s = sample_scanner();

        assert_eq!(s.peek_n(1), Some(" ".as_bytes()));
        assert_eq!(s.peek_n(2), Some("  ".as_bytes()));
        assert_eq!(s.peek_n(10), Some("     this ".as_bytes()));
        assert_eq!(s.peek_n(100), None);
    }
}
