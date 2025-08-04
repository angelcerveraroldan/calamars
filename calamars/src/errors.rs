use crate::token::Position;
use std::io::Error;

#[derive(Debug)]
pub enum LexErrorKind {
    UnexpectedEndOfFile,
    CouldNotTokenize,
    TokenizingError(String),
    NoCharacterFound,
    StartingMismatch,
    FileNotFound(Error),
}

pub struct LexError {
    pub kind: LexErrorKind,
    pub position: Position,
}

impl LexError {
    pub fn new(kind: LexErrorKind, position: Position) -> Self {
        Self { kind, position }
    }
}
