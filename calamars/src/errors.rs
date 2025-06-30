use crate::lexer::Position;
use std::io::Error;

#[derive(Debug)]
pub enum LexError {
    CouldNotTokenize(Position),
    FileNotFound(Error),
}
