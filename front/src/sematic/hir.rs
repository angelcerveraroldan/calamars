use calamars_core::global::GlobalContext;
use calamars_core::{MaybeErr, ids, types};

use crate::{
    sematic::{error::SemanticError, hir},
    syntax::span::Span,
};

/// An arena for compile-time known strings.
pub type ConstantStringArena = calamars_core::InternArena<String, ids::StringId>;

/// An arena for the names of functions
pub type IdentArena = calamars_core::InternArena<String, ids::IdentId>;

pub type ExpressionArena = calamars_core::Arena<Expr, ids::ExpressionId>;

pub type SymbolArena = calamars_core::UncheckedArena<Symbol, ids::SymbolId>;

#[derive(Debug, PartialEq, Eq, Clone, Hash)]
pub enum ItemId {
    Expr(ids::ExpressionId),
    Symbol(ids::SymbolId),
}

#[derive(Debug, PartialEq, Clone)]
pub enum Const {
    I64(i64),
    Bool(bool),
    String(ids::StringId),
}

#[derive(Debug, PartialEq, Clone)]
pub enum BinOp {
    Add,
    Sub,
    Mult,
    Div,
    Mod,

    EqEq,
    NotEqual,
    Greater,
    Geq,
    Less,
    Leq,

    And,
    Or,
    Xor,
}

#[derive(Debug, PartialEq, Clone)]
pub enum Expr {
    Err,
    Literal {
        constant: Const,
        span: Span,
    },
    /// A function or a var/val is being used, for example `x`.
    Identifier {
        /// This tells us exactly what "declaration" we are referring to
        id: ids::SymbolId,
        span: Span,
    },
    BinaryOperation {
        operator: BinOp,
        lhs: ids::ExpressionId,
        rhs: ids::ExpressionId,
        span: Span,
    },
    Call {
        f: ids::ExpressionId,
        input: ids::ExpressionId,
        span: Span,
    },
    If {
        predicate: ids::ExpressionId,
        then: ids::ExpressionId,
        otherwise: ids::ExpressionId,

        span: Span,
        pred_span: Span,
        then_span: Span,
        othewise_span: Span,
    },
    Block {
        items: Box<[ItemId]>,
        final_expr: Option<ids::ExpressionId>,
        span: Span,
    },
}

impl Expr {
    pub fn get_span(&self) -> Option<Span> {
        match self {
            Expr::Err => None,
            Expr::Literal { span, .. }
            | Expr::If { span, .. }
            | Expr::Identifier { span, .. }
            | Expr::BinaryOperation { span, .. }
            | Expr::Block { span, .. }
            | Expr::Call { span, .. } => Some(*span),
        }
    }
}

impl MaybeErr for Expr {
    const ERR: Self = Expr::Err;
}

#[derive(Debug, Clone)]
pub struct SymbolDec {
    pub inputs: Box<[ids::SymbolId]>,
    pub body: ids::ExpressionId,
}

/// A builder for calamars declarations
///
/// This should not be used for things like FFI, a body must be had.
pub struct SymbolBuilder {
    pub ty: ids::TypeId,
    pub name: ids::IdentId,
    /// A span to the name of the symbol in its type declaration
    pub span_name_type: Span,
    /// A span to the name of the symbol in its body declaration
    pub span_name_decl: Option<Span>,
    pub symbol_declaration: Option<SymbolDec>,
    pub reserved_spot: ids::SymbolId,
}

impl SymbolBuilder {
    pub fn new(
        ty: ids::TypeId,
        name: ids::IdentId,
        span_name_type: Span,
        span_name_decl: Option<Span>,
        symbol_declaration: Option<SymbolDec>,
        reserved_spot: ids::SymbolId,
    ) -> Self {
        Self {
            ty,
            name,
            span_name_type,
            span_name_decl,
            symbol_declaration,
            reserved_spot,
        }
    }

    pub fn ident_id(&self) -> ids::IdentId {
        self.name
    }
}

/// A symbol such as:
/// ```txt
/// foo :: Int
/// foo = 2
/// ```
#[derive(Debug)]
pub struct Symbol {
    pub ty: ids::TypeId,
    pub name: ids::IdentId,
    pub kind: SymbolKind,
}

#[derive(Debug, Clone)]
pub enum SymbolKind {
    Param {
        span: Span,
    },
    Ffi {
        /// span of the function name on type definition
        span_type: Span,
    },
    Defn {
        /// span of the function name on type definition
        span_type: Span,
        /// span of the function name on the declaration
        span_decl: Span,
        declaration: SymbolDec,
    },
}

impl Symbol {
    pub fn name_id(&self) -> ids::IdentId {
        self.name
    }

    pub fn span(&self) -> Span {
        match self.kind {
            SymbolKind::Param { span: span }
            | SymbolKind::Ffi { span_type: span }
            | SymbolKind::Defn {
                span_type: span, ..
            } => span,
        }
    }
}

/// Given some functions type, find the type of the first n inputs,
/// and the return type.
pub fn take_inputs(
    type_id: ids::TypeId,
    mut n: usize,
    global_ctx: &GlobalContext,
    span: Span,
) -> Result<(Vec<ids::TypeId>, ids::TypeId), SemanticError> {
    let expected = n;
    let mut v = vec![];
    let mut curr_type_id = type_id;

    while n != 0 {
        let curr_type = global_ctx.types.get_unchecked(curr_type_id);
        match curr_type {
            types::Type::Function { input, output } => {
                v.push(*input);
                curr_type_id = *output;
                n -= 1;
            }
            types::Type::Error => {
                return Ok((v, curr_type_id));
            }
            _ => {
                if v.is_empty() {
                    return Err(SemanticError::NotSupported {
                        msg: "Expected a function type",
                        span,
                    });
                }
                return Err(SemanticError::ArityError {
                    expected,
                    actual: expected - n,
                    span,
                });
            }
        }
    }
    Ok((v, curr_type_id))
}

pub struct Module {
    /// For now, ModuleId is the same as FileId, as each file is exactly a module.
    pub id: ids::FileId,
    pub name: String,

    pub idents: hir::IdentArena,
    pub symbols: hir::SymbolArena,
    pub exprs: hir::ExpressionArena,

    /// Top level symbols defined in the module
    pub roots: Box<[ids::SymbolId]>,
    /// Expression types
    pub expression_types: hashbrown::HashMap<ids::ExpressionId, ids::TypeId>,
}
