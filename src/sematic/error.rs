use crate::syntax::span::Span;

pub enum SemanticError {
    Redeclaration {
        original_span: Span,
        redec_span: Span,
    },
    IdentNotFound {
        /// Name of the identifier which was not found
        name: String,
        /// Where in the source code we tried to call this ident
        span: Span,
    },
}
