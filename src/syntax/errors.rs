use crate::syntax::span::Span;

pub enum ParsingError {
    Expected { expected: String, span: Span },
}
