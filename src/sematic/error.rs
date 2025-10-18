use crate::syntax::span::Span;

#[derive(Debug)]
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
    TypeNotFound {
        type_name: String,
        span: Span,
    },
    /// Something is missing it's type!
    TypeMissing,
    TypeMissingCtx {
        for_identifier: Span,
    },
    QualifiedTypeNotSupported {
        span: Span,
    },

    WrongType {
        span: Span,
    },
}
