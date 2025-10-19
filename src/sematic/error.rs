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
    /// Return types on if and else are not the same, so error
    MismatchedIfBranches {
        /// Span of the expressoin being returned in the `then`
        then_span: Span,
        /// Span of the expression being returned in the `else`
        else_span: Span,
    },
    WrongType {
        /// Accepted type or types
        expected: String,
        /// Type that was found
        actual: String,
        span: Span,
    },
    NotSupported {
        msg: &'static str,
        span: Span,
    },
}
