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
    Arrow, FatArrow,

    // Symbols
    Plus, Minus, Star,
    Slash, Equal, EqualEqual,
    NotEqual, Less, LessEqual,
    Greater, GreaterEqual,

    // Comments
    LineComment,
    BlockComment,
    DocComment,

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

    pub fn peek_n(&self, n: usize) -> Option<&str> {
        self.source
            .get(self.current.offset..self.current.offset + n)
    }

    /// Look at the next character and update the current position
    ///
    /// This will also keep track of the offset
    pub fn pop(&mut self) -> Option<char> {
        let c: char = self.peek()?;
        self.current.record_offset(c);
        if c == '\n' {
            self.current.next_line();
        } else {
            self.current.column += 1;
        }
        Some(c)
    }

    pub fn pop_n(&mut self, n: usize) -> Option<String> {
        let mut ans = String::new();
        for _ in 0..n {
            ans.push(self.pop()?);
        }
        Some(ans)
    }

    pub fn matches(&self, value: &str) -> bool {
        let len = value.len();
        self.source
            .get(self.current.offset..)
            .map(|s| s.starts_with(value))
            .unwrap_or(false)
    }

    pub fn take_while(&mut self, predicate: impl Fn(char) -> bool) -> Option<String> {
        let mut result = String::new();
        while self.peek().map_or(false, &predicate) {
            result.push(self.pop()?);
        }
        (!result.is_empty()).then_some(result)
    }

    /// Return None if we need to stop taking
    /// Return Some(A) to keep updating the accumulator
    pub fn take_while_acc<A>(
        &mut self,
        start: A,
        predicate: impl Fn(&A, char) -> Option<A>,
    ) -> Option<String> {
        let mut result = String::new();
        let mut accumulator = start;
        while let Some(acc) = self.peek().map_or(None, |c| predicate(&accumulator, c)) {
            result.push(self.pop()?);
            accumulator = acc;
        }
        (!result.is_empty()).then_some(result)
    }

    /// Consume all the whitespace -- returns nothing, but updates offset
    pub fn lex_whitespace(&mut self) {
        self.take_while(|c| c.is_whitespace());
    }

    /// Able to lex integers and floating point numbers
    pub fn lex_number(&mut self) -> Option<TokenType> {
        let is_neg = self.peek() == Some('-');
        if is_neg {
            self.pop();
        }

        self.take_while_acc::<bool>(false, |acc, c| match c {
            '.' if !acc => Some(true),
            '.' => None,
            d if d.is_digit(10) || d == '_' => Some(*acc),
            _ => None,
        })
        .map(|s| {
            let s = if is_neg { format!("-{}", s) } else { s };
            if s.contains('.') {
                TokenType::Float(s.parse().unwrap())
            } else {
                TokenType::Int(s.parse().unwrap())
            }
        })
    }

    pub fn lex_keyword_or_ident(&mut self) -> Option<TokenType> {
        let start = self.peek()?;
        if !start.is_alphabetic() && start != '_' {
            return None;
        }

        let ident = self.take_while(|c| c.is_alphanumeric() || c == '_')?;
        match ident.as_str() {
            "def" => Some(TokenType::Def),
            "mut" => Some(TokenType::Mut),
            "given" => Some(TokenType::Given),
            "match" => Some(TokenType::Match),
            "if" => Some(TokenType::If),
            "else" => Some(TokenType::Else),
            "let" => Some(TokenType::Let),
            "return" => Some(TokenType::Return),
            "module" => Some(TokenType::Module),
            "import" => Some(TokenType::Import),
            "struct" => Some(TokenType::Struct),
            "trait" => Some(TokenType::Trait),
            "and" => Some(TokenType::And),
            "or" => Some(TokenType::Or),
            "not" => Some(TokenType::Not),
            "enum" => Some(TokenType::Enum),
            _ => Some(TokenType::Ident(ident)),
        }
    }

    /// Single character tokens
    pub fn lex_single_char(&mut self) -> Option<TokenType> {
        let c = self.peek()?;
        let token = match c {
            '(' => TokenType::LParen,
            ')' => TokenType::RParen,
            '{' => TokenType::LBrace,
            '}' => TokenType::RBrace,
            '[' => TokenType::LBracket,
            ']' => TokenType::RBracket,
            '.' => TokenType::Dot,
            ',' => TokenType::Comma,
            ':' => TokenType::Colon,
            ';' => TokenType::Semicolon,
            _ => return None,
        };
        self.pop(); // consume the matched char
        Some(token)
    }

    /// Lex double character operators
    ///
    /// Todo:
    /// - Block comments
    #[rustfmt::skip]
    pub fn lex_operator(&mut self) -> Option<TokenType> {
        let two = self.peek_n(2);
        let tok = match two.as_deref() {
            Some("==") => { self.pop_n(2); TokenType::EqualEqual }
            Some("!=") => { self.pop_n(2); TokenType::NotEqual }
            Some("<=") => { self.pop_n(2); TokenType::LessEqual }
            Some(">=") => { self.pop_n(2); TokenType::GreaterEqual }
            Some("=>") => { self.pop_n(2); TokenType::FatArrow }
            Some("->") => { self.pop_n(2); TokenType::Arrow }
            _ => {
                let one = self.peek()?;
                match one {
                    '+' => { self.pop(); TokenType::Plus }
                    '-' => { self.pop(); TokenType::Minus }
                    '*' => { self.pop(); TokenType::Star }
                    '/' => { self.pop(); TokenType::Slash }
                    '=' => { self.pop(); TokenType::Equal }
                    '<' => { self.pop(); TokenType::Less }
                    '>' => { self.pop(); TokenType::Greater }
                    _   => return None,
                }
            }
        };
        Some(tok)
    }

    pub fn lex_line_comment(&mut self) -> Option<TokenType> {
        let two = self.peek_n(2);
        if two != Some("--") {
            return None;
        }

        self.pop_n(2); // Remove the --
        self.take_while(|c| c != '\n'); // Remove everything in this line
        self.pop(); // Go on to the new line
        Some(TokenType::LineComment)
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
