use std::collections::HashMap;

#[rustfmt::skip]
#[derive(Debug, PartialEq)]
/// Calamars tokens
pub enum TokenType {
    // Keywords
    Def, Mut, Given, Match,
    If, Else, Let, Return,
    Module, Import, Struct,
    Trait, And, Or, Not, Enum,

    // Single chars
    LParen, RParen, LBrace,
    RBrace, LBracket, RBracket,
    Dot, Comma, Colon, Semicolon,

    // -> and =>
    Arrow, EqArrow,

    // Symbols
    Plus, Minus, Star,
    Slash, Equal, EqualEqual,
    BangEqual, Less, LessEqual,
    Greater, GreaterEqual,

    // Comments
    LineComment(String),
    BlockComment(String),
    DocComment(String),

    // Primitives
    Ident(String),
    Int(i64), Float(f64),
    Char(char), String(String),
    Bool(bool),

    // End of the file!
    EOF,
}

#[derive(Debug, PartialEq)]
struct Position {
    /// Line in the source code
    line: usize,
    /// Column in the source code
    column: usize,
    /// Byte offset -- The number of bytes before this position
    ///
    /// This is different from the number of characters
    offset: usize,
}

impl Default for Position {
    fn default() -> Self {
        Self {
            line: 1,
            column: 1,
            offset: 0,
        }
    }
}

impl Position {
    fn new(line: usize, column: usize, offset: usize) -> Self {
        Self {
            line,
            column,
            offset,
        }
    }

    fn next_line(&mut self) {
        self.line += 1;
        self.column = 1;
    }

    fn record_offset(&mut self, offset: char) {
        self.offset += offset.len_utf8();
    }
}

struct Token {
    token_type: TokenType,
    lexeme: String,
    position: Position,
}

impl Token {
    fn new(token_type: TokenType, lexeme: String, position: Position) -> Self {
        Self {
            token_type,
            lexeme,
            position,
        }
    }
}

pub struct Lexer {
    /// All the source code
    source: String,
    /// Here we will store the Tokens
    tokens: Vec<Token>,
    /// Current position in the source
    current: Position,
}

impl Lexer {
    /// Make a new lexer from some source code
    pub fn new(source: String) -> Self {
        Self {
            source,
            tokens: vec![],
            current: Default::default(),
        }
    }

    /// Look at the next character without updating the current position
    pub fn peek(&self) -> Option<char> {
        self.source.get(self.current.offset..)?.chars().next()
    }

    /// Look at the next character and update the current position
    pub fn pop(&mut self) -> Option<char> {
        let c: char = self.peek()?;
        self.current.record_offset(c);
        if c != '\n' {
            self.current.next_line();
        }
        Some(c)
    }

    pub fn lex_whitespace(&mut self) {
        while let Some(c) = self.peek() {
            if c.is_whitespace() {
                self.pop();
            } else {
                break;
            }
        }
    }

    /// Able to lex integers and floating point numbers
    pub fn lex_number(&mut self) -> Option<TokenType> {
        let mut number = String::new();
        let mut found_decimal = false;

        // Handle negative numbers
        if self.peek() == Some('-') {
            self.pop();
            number.push('-');
        }
        loop {
            match self.pop() {
                // We reached the end of the file
                None => break,
                // Collect all digits
                Some(c) if c.is_numeric() => {
                    number.push(c);
                }
                // Allow for floats
                Some('.') if !found_decimal => {
                    number.push('.');
                    found_decimal = true;
                }
                // Ignore _, this makes it easier to enter large numbers
                Some('_') => (),
                // If we find anything else, we are done
                _ => break,
            }
        }

        if found_decimal {
            Some(TokenType::Float(number.parse().unwrap()))
        } else {
            Some(TokenType::Int(number.parse().unwrap()))
        }
    }

    /// Able to lex an identifier
    pub fn lex_identifier(&mut self) -> Option<TokenType> {
        let mut identifier = String::new();
        while let Some(c) = self.peek() {
            if c.is_alphanumeric() || c == '_' {
                identifier.push(c);
                self.pop();
            } else {
                break;
            }
        }

        if identifier.is_empty() {
            None
        } else {
            Some(TokenType::Ident(identifier))
        }
    }

    pub fn lex_sign(&mut self) -> Option<TokenType> {
        // FIXME: This is not complete
        match self.pop() {
            Some('+') => Some(TokenType::Plus),
            Some('-') => Some(TokenType::Minus),
            Some('=') if self.peek() == Some('>') => Some(TokenType::EqArrow),
            Some('=') if self.peek() == Some('=') => Some(TokenType::EqualEqual),
            Some('=') => Some(TokenType::Equal),
            Some('!') if self.peek() == Some('=') => Some(TokenType::BangEqual),
            Some('>') if self.peek() == Some('=') => Some(TokenType::GreaterEqual),
            Some('>') => Some(TokenType::Greater),
            Some('<') if self.peek() == Some('=') => Some(TokenType::LessEqual),
            Some('<') => Some(TokenType::Less),
            Some('*') => Some(TokenType::Star),
            Some('/') => Some(TokenType::Slash),
            _ => None,
        }
    }

    pub fn lex_next(&mut self) {}
}

#[cfg(test)]
mod test_lexer {
    use super::*;
    use proptest::arbitrary::any;
    use proptest::strategy::{Just, Strategy};
    use proptest::string::string_regex;
    use proptest::{prop_oneof, proptest};

    // Generate "normal" floating points (for example, no scientific notation)
    fn float_string_strategy() -> impl Strategy<Value = String> {
        let sign = prop_oneof![Just("".to_string()), Just("-".to_string())];
        let int_part = string_regex("[1-9][0-9]{0,30}").unwrap();
        let frac_part = string_regex("\\.[0-9]{0,30}").unwrap();
        (sign, int_part, frac_part)
            .prop_map(|(sign, int_part, frac_part)| format!("{}{}{}", sign, int_part, frac_part))
    }

    proptest! {
        #[test]
        fn test_lex_float(s in float_string_strategy()) {
            let mut lexer = Lexer::new(String::from(s.clone()));
            assert_eq!(lexer.lex_number(), Some(TokenType::Float(
                s.parse::<f64>().unwrap()
            )));
        }

        #[test]
        fn test_lex_int(s in any::<i64>()) {
            let mut lexer = Lexer::new(String::from(format!("{}", s)));
            assert_eq!(lexer.lex_number(), Some(TokenType::Int(s)));
        }
    }
}
