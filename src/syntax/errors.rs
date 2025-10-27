use crate::syntax::span::Span;

pub enum ParsingError {
    Expected {
        expected: String,
        span: Span,
    },
    DelimeterNotClosed {
        expected: &'static str,
        at: Span,
        opening_loc: Span,
    },
}
