use crate::errors::LexErrorKind;

#[allow(dead_code)]
#[rustfmt::skip]
#[derive(Debug, PartialEq, Clone)]
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
#[derive(Debug, Clone, Copy)]
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

    pub fn process_many(&self, chars: &[u8]) -> Self {
        chars
            .iter()
            .fold(self.clone(), |acc, next| acc.process(*next))
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

    /// # Safety
    /// This assumes that the span is inbounds
    unsafe fn get_span(&self, Span { from, bytes }: &Span) -> &[u8] {
        let start = from.offset;
        &self.source[start..start + bytes]
    }

    fn peek(&self) -> Result<u8, LexErrorKind> {
        match self.source.get(self.current_pos.offset) {
            Some(c) => Ok(*c),
            None => Err(LexErrorKind::NoCharacterFound),
        }
    }

    /// Return error if peek is none
    fn consume(&mut self) -> Result<(), LexErrorKind> {
        if let Ok(c) = self.peek() {
            self.current_pos = self.current_pos.process(c);
            return Ok(());
        }

        Err(LexErrorKind::NoCharacterFound)
    }

    fn consume_n(&mut self, n: usize) -> Result<(), LexErrorKind> {
        if let Some(cs) = self.peek_n(n) {
            for c in cs {
                self.current_pos = self.current_pos.process(*c);
            }
            return Ok(());
        }

        Err(LexErrorKind::NoCharacterFound)
    }

    fn peek_n(&self, n: usize) -> Option<&'a [u8]> {
        let start = self.current_pos.offset;
        let ending = start + n;
        if self.source.len() < ending {
            return None;
        }
        Some(&self.source[start..ending])
    }

    fn next_word(&self) -> Option<&'a [u8]> {
        let start = self.current_pos.offset;
        let mut end = start;
        while end < self.source.len() && self.source[end].is_ascii_alphanumeric() {
            end += 1;
        }

        (start != end).then_some(&self.source[start..end])
    }

    fn advance(&mut self) {
        if let Ok(c) = self.peek() {
            self.current_pos = self.current_pos.process(c);
        }
    }

    // ################# //
    // HANDLE WHITESPACE //
    // ################# //

    /// Is the next character whitespace
    fn is_whitespace(&self) -> bool {
        match self.peek() {
            Err(_) => false,
            Ok(c) => c.is_ascii_whitespace(),
        }
    }

    /// Skip all leading whitespace
    fn skip_whitespace(&mut self) {
        while self.is_whitespace() {
            self.advance();
        }
    }

    fn tokenize_keywords(&self) -> Result<Token, LexErrorKind> {
        let next_word = self.next_word().ok_or(LexErrorKind::TokenizingError(
            "No next word found".to_string(),
        ))?;

        #[rustfmt::skip]
        let token_type = match next_word {
            b"def"    => TokenType::Def,
            b"mut"    => TokenType::Mut,
            b"given"  => TokenType::Given,
            b"match"  => TokenType::Match,
            b"if"     => TokenType::If,
            b"else"   => TokenType::Else,
            b"let"    => TokenType::Let,
            b"return" => TokenType::Return,
            b"module" => TokenType::Module,
            b"import" => TokenType::Import,
            b"struct" => TokenType::Struct,
            b"trait"  => TokenType::Trait,
            b"and"    => TokenType::And,
            b"or"     => TokenType::Or,
            b"not"    => TokenType::Not,
            b"enum"   => TokenType::Enum,
            b"true"   => TokenType::True,
            b"false"  => TokenType::False,
            _         => TokenType::Ident,
        };

        Ok(Token {
            ttype: token_type,
            source: Span {
                from: self.current_pos,
                bytes: next_word.len(),
            },
        })
    }

    fn tokenize_punctuation(&self) -> Result<Token, LexErrorKind> {
        let span = |len: usize| Span {
            from: self.current_pos,
            bytes: len,
        };

        if let Some(two) = self.peek_n(2) {
            #[rustfmt::skip]
            let tt = match two {
                b"->" => Some(TokenType::Arrow),
                b"=>" => Some(TokenType::FatArrow),
                b"==" => Some(TokenType::EqualEqual),
                b"!=" => Some(TokenType::NotEqual),
                b"<=" => Some(TokenType::LessEqual),
                b">=" => Some(TokenType::GreaterEqual),
                _ => None,
            };

            if let Some(token_type) = tt {
                return Ok(Token {
                    ttype: token_type,
                    source: span(2),
                });
            }
        }

        let one = self.peek()?;
        let token_type = match one {
            b'(' => TokenType::LParen,
            b')' => TokenType::RParen,
            b'{' => TokenType::LBrace,
            b'}' => TokenType::RBrace,
            b'[' => TokenType::LBracket,
            b']' => TokenType::RBracket,
            b'.' => TokenType::Dot,
            b',' => TokenType::Comma,
            b':' => TokenType::Colon,
            b';' => TokenType::Semicolon,
            b'+' => TokenType::Plus,
            b'-' => TokenType::Minus,
            b'*' => TokenType::Star,
            b'/' => TokenType::Slash,
            b'=' => TokenType::Equal,
            b'<' => TokenType::Less,
            b'>' => TokenType::Greater,
            _ => return Err(LexErrorKind::StartingMismatch),
        };

        Ok(Token {
            ttype: token_type,
            source: span(1),
        })
    }

    fn tokenize_number(&self) -> Result<Token, LexErrorKind> {
        let p = self.peek()?;
        if !p.is_ascii_alphanumeric() {
            return Err(LexErrorKind::StartingMismatch);
        }

        let start = self.current_pos.offset;
        let mut end = start;

        let mut found_dot = false;
        while end < self.source.len()
            && ((self.source[end] <= b'9' && self.source[end] >= b'0')
                || self.source[end] == b'_'
                || self.source[end] == b'.')
        {
            // Only allow for puncuation once
            if self.source[end] == b'.' {
                if found_dot {
                    break;
                }
                found_dot = true;
            }

            end += 1;
        }

        (start != end)
            .then_some({
                let ttype = found_dot
                    .then_some(TokenType::Float)
                    .unwrap_or(TokenType::Int);
                Token {
                    ttype,
                    source: Span {
                        from: self.current_pos,
                        bytes: end - start,
                    },
                }
            })
            .ok_or(LexErrorKind::TokenizingError(
                "No number was found".to_string(),
            ))
    }

    fn tokenize_string(&self) -> Result<Token, LexErrorKind> {
        if self.peek()? != b'"' {
            return Err(LexErrorKind::CouldNotTokenize);
        }

        let mut esc = false;
        let mut end = self.current_pos.offset + 1;
        while end < self.source.len() {
            let c = self.source[end];

            // We are ignoring the next char
            if esc {
                esc = false;
                end += 1;
                continue;
            }

            if c == b'\\' {
                esc = true;
            }

            // We found the closing quote
            if c == b'"' {
                let span = Span {
                    from: self.current_pos,
                    bytes: end + 1 - self.current_pos.offset,
                };
                return Ok(Token {
                    ttype: TokenType::String,
                    source: span,
                });
            }

            end += 1;
        }

        Err(LexErrorKind::TokenizingError(
            "Did not find closing quote".to_string(),
        ))
    }

    fn tokenize_comments(&self) -> Result<Token, LexErrorKind> {
        if self.peek_n(2).ok_or(LexErrorKind::TokenizingError(
            "Could not get 2 characters".to_string(),
        ))? != b"--"
        {
            return Err(LexErrorKind::StartingMismatch);
        }

        let start = self.current_pos.offset;
        let mut end = start + 2;

        while end < self.source.len() && self.source[end] != b'\n' {
            end += 1;
        }

        Ok(Token {
            ttype: TokenType::LineComment,
            source: Span {
                from: self.current_pos,
                bytes: end - start,
            },
        })
    }

    fn next_token(&mut self) -> Result<Token, LexErrorKind> {
        self.skip_whitespace();

        self.tokenize_comments()
            .or_else(|_| self.tokenize_number())
            .or_else(|_| self.tokenize_keywords())
            .or_else(|_| self.tokenize_string())
            .or_else(|_| self.tokenize_punctuation())
    }

    /// Turn the source code into a vector of tokens
    fn tokenize_source(&mut self) -> Result<(), LexErrorKind> {
        // If we find a token, we will process it, and push it
        while let Ok(token) = self.next_token() {
            let source = unsafe { self.get_span(&token.source) };
            self.current_pos = self.current_pos.process_many(source);
            self.tokens.push(token);
        }

        // Check that we finished the entire code
        if self.current_pos.offset != self.source.len() {
            Err(LexErrorKind::TokenizingError(
                "Did not finish tokeinizing all file".to_string(),
            ))
        } else {
            Ok(())
        }
    }

    pub fn tokens(&self) -> &[Token] {
        &self.tokens
    }
}

#[cfg(test)]
mod test_scanner {
    use crate::token::{Scanner, TokenType};

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

    #[test]
    fn test_token_keyw() {
        let mut scanner = Scanner::new("def let return");
        let _ = scanner.tokenize_source();
        let toks = scanner.tokens;

        assert_eq!(toks.len(), 3);
        assert_eq!(toks[0].ttype, TokenType::Def);
        assert_eq!(toks[1].ttype, TokenType::Let);
        assert_eq!(toks[2].ttype, TokenType::Return);
    }

    #[test]
    fn test_token_numbers() {
        let mut scanner = Scanner::new("def 12 12.2");
        let _ = scanner.tokenize_source();
        let toks = scanner.tokens;

        assert_eq!(toks.len(), 3);
        assert_eq!(toks[0].ttype, TokenType::Def);
        assert_eq!(toks[1].ttype, TokenType::Int);
        assert_eq!(toks[2].ttype, TokenType::Float);
    }

    #[test]
    fn test_tokenize_string() {
        let mut s = Scanner::new("def x \"Hye\"");
        s.tokenize_source().unwrap();
        assert_eq!(s.tokens.len(), 3);
        assert_eq!(s.tokens[2].ttype, TokenType::String);
    }

    #[test]
    fn test_string_with_inner_quotes() {
        use crate::token::{Scanner, TokenType};

        let input = r#""say \"hi\" to me""#;
        let mut scanner = Scanner::new(input);
        scanner.tokenize_source();
        let tokens = scanner.tokens();
        assert_eq!(tokens.len(), 1);
        assert_eq!(tokens[0].ttype, TokenType::String);
        let span = unsafe { scanner.get_span(&tokens[0].source) };
        assert_eq!(span, input.as_bytes());
    }

    #[test]
    fn test_tokenize_punct() {
        use crate::token::TokenType::*;
        let input = "-> => == != <= >= ( ) { } [ ] . , : ; + - * / = < >";
        let mut scanner = Scanner::new(input);
        scanner.tokenize_source().unwrap();

        let expected = vec![
            Arrow,
            FatArrow,
            EqualEqual,
            NotEqual,
            LessEqual,
            GreaterEqual,
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
            Plus,
            Minus,
            Star,
            Slash,
            Equal,
            Less,
            Greater,
        ];

        let actual: Vec<_> = scanner.tokens().iter().map(|t| t.ttype.clone()).collect();
        assert_eq!(actual, expected);
    }
}
