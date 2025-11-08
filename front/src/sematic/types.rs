//! Type checking on the HIR

use calamars_core::{
    MaybeErr,
    ids::{self, ExpressionId},
};

use crate::{
    sematic::{
        error::SemanticError,
        hir::{self, ExpressionArena, IdentArena, SymbolArena, Type, TypeArena},
    },
    syntax::{ast::Expression, span::Span},
};

pub struct TypeHandler {
    symbols: SymbolArena,
    expressions: ExpressionArena,
    types: TypeArena,

    errors: Vec<SemanticError>,
}

impl TypeHandler {
    fn type_expression(&mut self, e_id: ids::ExpressionId) -> ids::TypeId {
        let expression = self.expressions.get_unchecked(e_id);
        match expression {
            super::hir::Expr::Err => self.types.err_id(),
            super::hir::Expr::Literal { constant, .. } => {
                let ty = match constant {
                    super::hir::Const::I64(_) => Type::Integer,
                    super::hir::Const::Bool(_) => Type::Boolean,
                    super::hir::Const::String(_) => Type::String,
                };
                *self.types.resolve_unchecked(&ty)
            }
            super::hir::Expr::Identifier { id, .. } => self
                .symbols
                .get(*id)
                .map(|s| s.ty_id())
                .unwrap_or(self.types.err_id()),
            super::hir::Expr::BinaryOperation {
                operator,
                lhs,
                rhs,
                span,
            } => self.type_check_binary_ops(operator, lhs, rhs, *span),
            super::hir::Expr::Call { f, inputs, span } => {
                let expression_ty = self.type_expression(*f);
                // TODO: Handle error type separately here
                match self.types.get_unchecked(expression_ty) {
                    Type::Function { output, .. } => *output,
                    otherwise => {
                        self.errors.push(SemanticError::NonCallable {
                            msg: "Expected callable",
                            span: *span,
                        });
                        self.types.err_id()
                    }
                }
            }
        }
    }

    fn type_check_binary_ops(
        &mut self,
        op: &hir::BinOp,
        lhs: &ExpressionId,
        rhs: &ExpressionId,
        span: Span,
    ) -> ids::TypeId {
        todo!()
    }
}
