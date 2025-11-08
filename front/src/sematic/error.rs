use std::ops::Range;

use ariadne::{Color, Fmt, Label, Report, ReportKind, Source};

use crate::{
    errors::{PrettyError, label_from},
    syntax::span::Span,
};

#[derive(Debug, PartialEq, Clone)]
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
    FnWrongReturnType {
        expected: String,
        return_type_span: Option<Span>, // When no span is found, we have inferred Unit
        fn_name_span: Span,
        actual: String,
        return_span: Option<Span>, // If something is being returned
        body_span: Span,
    },
    BindingWrongType {
        expected: String,
        return_type_span: Span,
        actual: String,
        return_span: Option<Span>,
        body_span: Span,
    },
    NotSupported {
        msg: &'static str,
        span: Span,
    },
    /// When looking up the SymbolId in the symbol arena, it was not found.
    ///
    /// This should be an internal error, as it will make no sense to the user, and so maybe it
    /// should be handled differently...
    SymbolIdNotFound {
        id: calamars_core::ids::SymbolId,
    },
    ArityError {
        expected: usize,
        actual: usize,
        span: Span,
    },
    NonCallable {
        msg: &'static str,
        span: Span,
    },
    InternalError {
        msg: &'static str,
        span: Span,
    },
}

impl PrettyError for SemanticError {
    fn message(&self) -> &str {
        match self {
            SemanticError::Redeclaration { .. } => "Redeclaration is not allowed",
            SemanticError::IdentNotFound { .. } => "Identifier not found",
            SemanticError::WrongType { .. } | SemanticError::BindingWrongType { .. } => {
                "Wrong type returned"
            }
            SemanticError::TypeNotFound { .. } => " Type not found",
            SemanticError::TypeMissingCtx { .. } => "Missing type",
            SemanticError::MismatchedIfBranches { .. } => {
                "Both branches in if statement must return the same type"
            }
            SemanticError::TypeMissing => "Type declaration missing",
            SemanticError::QualifiedTypeNotSupported { .. } => {
                "Qualified types are not yet supported"
            }
            SemanticError::NotSupported { .. } => "Use of unsupported feature",
            SemanticError::SymbolIdNotFound { .. } => "Internal error",
            SemanticError::ArityError { .. } => "Wrong number of inputs found",
            SemanticError::NonCallable { .. } => "Calling non-callable",
            SemanticError::InternalError { .. } => "Internal error",
            SemanticError::FnWrongReturnType { .. } => "Wrong type returned by function",
        }
    }

    fn labels<'a>(&'a self, file_name: &'a String) -> Vec<Label<(&'a String, Range<usize>)>> {
        match self {
            SemanticError::Redeclaration {
                original_span,
                redec_span,
            } => {
                vec![
                    label_from(
                        file_name,
                        *original_span,
                        "First declared here",
                        Some(Color::Green),
                    ),
                    label_from(
                        file_name,
                        *redec_span,
                        "Redeclared here",
                        Some(Color::Magenta),
                    ),
                ]
            }
            SemanticError::IdentNotFound { name, span } => vec![label_from(
                file_name,
                *span,
                "Identifier not found",
                Some(Color::Magenta),
            )],
            SemanticError::TypeNotFound { type_name, span } => {
                vec![label_from(file_name, *span, "Type not found", None)]
            }
            SemanticError::WrongType {
                expected,
                actual,
                span,
            } => vec![label_from(
                file_name,
                *span,
                format!(
                    "Expected to find `{}` but found `{}`",
                    expected.fg(Color::Green),
                    actual.fg(Color::Red)
                ),
                Some(Color::Red),
            )],
            SemanticError::BindingWrongType {
                expected,
                return_type_span,
                actual,
                return_span,
                body_span,
            } => {
                let mut v = vec![label_from(
                    file_name,
                    *return_type_span,
                    format!("Binding is of type `{}`", expected.fg(Color::Green)),
                    Some(Color::Green),
                )];

                let l = if let Some(rs) = return_span {
                    label_from(
                        file_name,
                        *rs,
                        format!("But it was assigned `{}`", actual.fg(Color::Red)),
                        Some(Color::Red),
                    )
                } else {
                    label_from(
                        file_name,
                        *body_span,
                        format!(
                            "But it was assigned `{}` as its body had not final expression",
                            actual.fg(Color::Red)
                        ),
                        Some(Color::Red),
                    )
                };
                v.push(l);
                v
            }
            SemanticError::FnWrongReturnType {
                expected,
                return_type_span,
                actual,
                return_span,
                fn_name_span,
                body_span,
            } => {
                let mut v = vec![label_from(
                    file_name,
                    return_type_span.unwrap_or(*fn_name_span),
                    format!(
                        "Function expected to return `{}`",
                        expected.fg(Color::Green)
                    ),
                    Some(Color::Green),
                )];

                let l = if let Some(rs) = return_span {
                    label_from(
                        file_name,
                        *rs,
                        format!("But it returned `{}`", actual.fg(Color::Red)),
                        Some(Color::Red),
                    )
                } else {
                    label_from(
                        file_name,
                        *body_span,
                        format!(
                            "But it returned `{}` as its body had not final expression",
                            actual.fg(Color::Red)
                        ),
                        Some(Color::Red),
                    )
                };

                v.push(l);
                v
            }
            SemanticError::TypeMissingCtx { for_identifier } => vec![label_from(
                file_name,
                *for_identifier,
                "Type annotation is needed",
                Some(Color::Magenta),
            )],
            SemanticError::MismatchedIfBranches {
                then_span,
                else_span,
            } => vec![
                label_from(
                    file_name,
                    *then_span,
                    "First return here",
                    Some(Color::Blue),
                ),
                label_from(
                    file_name,
                    *else_span,
                    "Second return here",
                    Some(Color::Cyan),
                ),
            ],
            SemanticError::QualifiedTypeNotSupported { span } => todo!(),
            SemanticError::NotSupported { msg, span } => {
                vec![label_from(file_name, *span, *msg, Some(Color::Red))]
            }
            SemanticError::SymbolIdNotFound { id } => todo!(),
            SemanticError::ArityError {
                expected,
                actual,
                span,
            } => vec![label_from(
                file_name,
                *span,
                format!(
                    "Expected {} parameters, but found {}",
                    expected.fg(Color::Green),
                    actual.fg(Color::Red)
                ),
                Some(Color::Magenta),
            )],
            SemanticError::NonCallable { msg, span } => {
                vec![label_from(file_name, *span, *msg, Some(Color::Magenta))]
            }
            SemanticError::InternalError { msg, span } => {
                vec![label_from(file_name, *span, *msg, None)]
            }
            _ => todo!(),
        }
    }

    fn notes(&self) -> Option<String> {
        match self {
            SemanticError::FnWrongReturnType {
                return_type_span, ..
            } => Some(
                "No return type was given to this function, so it was inferred that the function should expect no return `()`".to_string(),
            ),
            _ => None,
        }
    }
}
