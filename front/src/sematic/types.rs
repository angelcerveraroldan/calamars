//! Type checking on the HIR

use calamars_core::{
    MaybeErr,
    ids::{self, ExpressionId},
};

use crate::{
    sematic::{
        error::SemanticError,
        hir::{self, ExpressionArena, IdentArena, SymbolArena, Type, TypeArena, type_id_stringify},
    },
    syntax::{ast::Expression, span::Span},
};

/// Immutable context containig arenas from Stage 1
pub struct Context {
    symbols: SymbolArena,
    expressions: ExpressionArena,
}

pub struct TypeHandler {
    types: TypeArena,
    errors: Vec<SemanticError>,
}

impl TypeHandler {
    fn match_type(&mut self, id: ids::TypeId, ty: &Type) -> bool {
        let ty_id = self.types.intern(ty);
        id == ty_id
    }

    #[inline]
    fn push_wrong_type(&mut self, actual: ids::TypeId, expected: &str, span: Span) {
        self.errors.push(SemanticError::WrongType {
            actual: type_id_stringify(&self.types, actual),
            expected: expected.into(),
            span,
        });
    }

    fn ensure_numeric(&mut self, cx: &Context, expr: ids::ExpressionId, t: ids::TypeId) {
        if self.match_type(t, &Type::Integer) || self.match_type(t, &Type::Float) {
            return;
        }
        let sp = cx.expressions.get_unchecked(expr).get_span().unwrap();
        self.push_wrong_type(t, "Numerical", sp);
    }

    fn ensure_type(
        &mut self,
        ctx: &Context,
        expr: ids::ExpressionId,
        ty_id: ids::TypeId,
        expected: ids::TypeId,
    ) {
        if ty_id == expected {
            return;
        }
        let sp = ctx.expressions.get_unchecked(expr).get_span().unwrap();
        self.push_wrong_type(ty_id, &type_id_stringify(&self.types, expected), sp);
    }

    /// Given some expression, this will return it's type id. If there are any semantic typing
    /// errors, for example: `2 + "hello"` they will be added to the `errors` vector, and will
    /// return the typeid of the `Error` type.
    fn type_expression(&mut self, ctx: &Context, e_id: &ids::ExpressionId) -> ids::TypeId {
        let expression = ctx.expressions.get_unchecked(*e_id);

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
            super::hir::Expr::Identifier { id, .. } => ctx
                .symbols
                .get(*id)
                .map(|s| s.ty_id())
                .unwrap_or(self.types.err_id()),
            super::hir::Expr::BinaryOperation {
                operator,
                lhs,
                rhs,
                span,
            } => self.type_check_binary_ops(ctx, operator, lhs, rhs, *span),
            super::hir::Expr::Call { f, inputs, span } => {
                let expression_ty = self.type_expression(ctx, f);
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
        ctx: &Context,
        op: &hir::BinOp,
        lhs: &ExpressionId,
        rhs: &ExpressionId,
        span: Span,
    ) -> ids::TypeId {
        let lhs_type_id = self.type_expression(ctx, lhs);
        let rhs_type_id = self.type_expression(ctx, rhs);

        let int_type_id = *self.types.resolve_unchecked(&Type::Integer);
        let float_type_id = *self.types.resolve_unchecked(&Type::Float);

        match op {
            // Both numerical
            hir::BinOp::Add | hir::BinOp::Sub | hir::BinOp::Mult | hir::BinOp::Div => {
                self.ensure_numeric(ctx, *lhs, lhs_type_id);
                self.ensure_numeric(ctx, *rhs, rhs_type_id);

                let lhs_numerical = (lhs_type_id == float_type_id) || (lhs_type_id == int_type_id);
                let rhs_numerical = (rhs_type_id == float_type_id) || (rhs_type_id == int_type_id);

                // If they are not both numerica, then this is an error
                if !(lhs_numerical && rhs_numerical) {
                    return self.types.err_id();
                }

                // If they are both integers, then we will return integer
                if (lhs_type_id == int_type_id && rhs_type_id == int_type_id) {
                    return int_type_id;
                }

                // Both floats, or one float, then we cast to float
                float_type_id
            }
            // Both integers
            hir::BinOp::Mod => {
                self.ensure_type(ctx, *lhs, lhs_type_id, int_type_id);
                self.ensure_type(ctx, *rhs, rhs_type_id, int_type_id);
                self.types.err_id()
            }
        }
    }
}
