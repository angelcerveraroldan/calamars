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
            hir::Expr::Err => self.types.err_id(),
            hir::Expr::Literal { constant, .. } => {
                let ty = match constant {
                    hir::Const::I64(_) => Type::Integer,
                    hir::Const::Bool(_) => Type::Boolean,
                    hir::Const::String(_) => Type::String,
                };
                *self.types.resolve_unchecked(&ty)
            }
            hir::Expr::Identifier { id, .. } => ctx
                .symbols
                .get(*id)
                .map(|s| s.ty_id())
                .unwrap_or(self.types.err_id()),
            hir::Expr::BinaryOperation {
                operator,
                lhs,
                rhs,
                span,
            } => self.type_check_binary_ops(ctx, operator, lhs, rhs, *span),
            hir::Expr::Call { f, inputs, span } => {
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
            hir::Expr::If {
                predicate,
                then,
                otherwise,
                span,
                pred_span,
                then_span,
                othewise_span,
            } => {
                // Make sure that the predicate is a boolean
                let p_ty = self.type_expression(ctx, predicate);
                if (p_ty != self.types.err_id()) {
                    let bool = self.types.intern(&Type::Boolean);
                    self.ensure_type(ctx, *predicate, p_ty, bool);
                }

                // Make sure that if and else branches return the same
                let t_ty = self.type_expression(ctx, then);
                let o_ty = self.type_expression(ctx, otherwise);

                if (t_ty == self.types.err_id() || o_ty == self.types.err_id()) {
                    return self.types.err_id();
                }

                println!("lhs: {:?}, rhs: {:?}", t_ty, o_ty);
                if (t_ty != o_ty) {
                    self.errors.push(SemanticError::MismatchedIfBranches {
                        then_span: *then_span,
                        else_span: *othewise_span,
                    });
                    return self.types.err_id();
                }

                // If both branches retur the same, then return that type
                t_ty
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
            hir::BinOp::EqEq => {
                if (lhs_type_id == self.types.err_id() || rhs_type_id == self.types.err_id()) {
                    return self.types.err_id();
                }

                if (lhs_type_id != rhs_type_id) {
                    let rhs_expr = ctx.expressions.get_unchecked(*rhs);
                    self.errors.push(SemanticError::WrongType {
                        expected: type_id_stringify(&self.types, lhs_type_id),
                        actual: type_id_stringify(&self.types, rhs_type_id),
                        // We can unwrap since we made sure its not error type
                        span: rhs_expr.get_span().unwrap(),
                    });
                }
                self.types.err_id()
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

#[cfg(test)]
mod tests {
    use crate::sematic::hir::ConstantStringArena;

    use super::hir::{BinOp, Const, Expr, Symbol, SymbolKind};
    use super::ids;
    use super::*;
    fn mk_handler() -> TypeHandler {
        let mut types = TypeArena::new_checked();
        types.intern(&Type::Integer);
        types.intern(&Type::Float);
        types.intern(&Type::Boolean);
        types.intern(&Type::String);
        TypeHandler {
            types,
            errors: vec![],
        }
    }

    fn ty_int(handler: &TypeHandler) -> ids::TypeId {
        *handler.types.resolve_unchecked(&Type::Integer)
    }

    fn ty_float(handler: &TypeHandler) -> ids::TypeId {
        *handler.types.resolve_unchecked(&Type::Float)
    }

    fn ty_bool(handler: &TypeHandler) -> ids::TypeId {
        *handler.types.resolve_unchecked(&Type::Boolean)
    }

    fn ty_str(handler: &TypeHandler) -> ids::TypeId {
        *handler.types.resolve_unchecked(&Type::String)
    }

    fn ty_err(handler: &TypeHandler) -> ids::TypeId {
        handler.types.err_id()
    }

    fn lit_i64(arena: &mut ExpressionArena, v: i64) -> ids::ExpressionId {
        arena.push(Expr::Literal {
            constant: Const::I64(v),
            span: Span::dummy(),
        })
    }

    fn lit_bool(arena: &mut ExpressionArena, v: bool) -> ids::ExpressionId {
        arena.push(Expr::Literal {
            constant: Const::Bool(v),
            span: Span::dummy(),
        })
    }

    fn lit_str(
        arena: &mut ExpressionArena,
        s: &str,
        consts: &mut ConstantStringArena,
    ) -> ids::ExpressionId {
        let sid = consts.intern(&s.to_string());
        arena.push(Expr::Literal {
            constant: Const::String(sid),
            span: Span::dummy(),
        })
    }

    fn if_expr(
        arena: &mut ExpressionArena,
        predicate: ids::ExpressionId,
        then_e: ids::ExpressionId,
        otherwise_e: ids::ExpressionId,
    ) -> ids::ExpressionId {
        let pred_span = Span::dummy();
        let then_span = Span::dummy();
        let otherwise_span = Span::dummy();
        let node = Expr::If {
            predicate,
            then: then_e,
            otherwise: otherwise_e,
            span: Span::dummy(),
            pred_span,
            then_span,
            othewise_span: otherwise_span, // note: field name per your struct
        };
        arena.push(node)
    }

    fn ident(arena: &mut ExpressionArena, sym: ids::SymbolId) -> ids::ExpressionId {
        arena.push(Expr::Identifier {
            id: sym,
            span: Span::dummy(),
        })
    }

    fn bin(
        arena: &mut ExpressionArena,
        op: BinOp,
        l: ids::ExpressionId,
        r: ids::ExpressionId,
    ) -> ids::ExpressionId {
        arena.push(Expr::BinaryOperation {
            operator: op,
            lhs: l,
            rhs: r,
            span: Span::dummy(),
        })
    }

    fn eq(
        arena: &mut ExpressionArena,
        l: ids::ExpressionId,
        r: ids::ExpressionId,
    ) -> ids::ExpressionId {
        bin(arena, BinOp::EqEq, l, r)
    }

    fn var_symbol(ty: ids::TypeId, body: ids::ExpressionId) -> Symbol {
        Symbol::new(
            SymbolKind::Variable {
                body,
                mutable: false,
            },
            ty,
            // ahhhhh this is not great, but it will do for now
            ids::IdentId::from(1),
            Span::dummy(),
            Span::dummy(),
        )
    }

    #[test]
    fn add_int_and_string_reports_error() {
        // 1 + "hello" -> error
        let mut consts = ConstantStringArena::new_unchecked();
        let mut exprs = ExpressionArena::new_checked();

        let one = lit_i64(&mut exprs, 1);
        let hello = lit_str(&mut exprs, "hello", &mut consts);
        let add = bin(&mut exprs, BinOp::Add, one, hello);

        let ctx = Context {
            symbols: SymbolArena::new_unchecked(),
            expressions: exprs,
        };

        let mut handler = mk_handler();
        let before = handler.errors.len();
        let ty = handler.type_expression(&ctx, &add);

        assert_eq!(ty, ty_err(&handler));
        assert_eq!(
            handler.errors.len(),
            before + 1,
            "one type error should be recorded"
        );
    }

    #[test]
    /// 1 + 1.0 and 1.0 + 1 should both return float
    fn add_int_and_float_is_float_and_commutative() {
        let mut exprs = ExpressionArena::new_checked();
        let mut syms = SymbolArena::new_unchecked();
        let mut handler = mk_handler();

        let one = lit_i64(&mut exprs, 1);

        let float_ty = ty_float(&handler);
        let dummy_body = one;
        let y_sym_id = syms.push(var_symbol(float_ty, dummy_body));
        let y_ident = ident(&mut exprs, y_sym_id);

        let add1 = bin(&mut exprs, BinOp::Add, one, y_ident);
        let add2 = bin(&mut exprs, BinOp::Add, y_ident, one);

        let ctx = Context {
            symbols: syms,
            expressions: exprs,
        };

        let t1 = handler.type_expression(&ctx, &add1);
        let t2 = handler.type_expression(&ctx, &add2);

        assert_eq!(t1, ty_float(&handler));
        assert_eq!(t2, ty_float(&handler));
    }

    #[test]
    fn mult_int_int_is_int() {
        let mut exprs = ExpressionArena::new_checked();

        let a = lit_i64(&mut exprs, 1);
        let b = lit_i64(&mut exprs, 2);
        let mul = bin(&mut exprs, BinOp::Mult, a, b);

        let ctx = Context {
            symbols: SymbolArena::new_unchecked(),
            expressions: exprs,
        };

        let mut handler = mk_handler();
        let ty = handler.type_expression(&ctx, &mul);

        assert_eq!(ty, ty_int(&handler));
    }

    #[test]
    fn literals_have_expected_types() {
        let mut consts = ConstantStringArena::new_unchecked();
        let mut exprs = ExpressionArena::new_checked();

        let i = lit_i64(&mut exprs, 42);
        let b = lit_bool(&mut exprs, true);
        let s = lit_str(&mut exprs, "x", &mut consts);

        let ctx = Context {
            symbols: SymbolArena::new_unchecked(),
            expressions: exprs,
        };
        let mut handler = mk_handler();

        assert_eq!(handler.type_expression(&ctx, &i), ty_int(&handler));
        assert_eq!(handler.type_expression(&ctx, &b), ty_bool(&handler));
        assert_eq!(handler.type_expression(&ctx, &s), ty_str(&handler));
    }

    #[test]
    fn identifier_uses_symbol_type() {
        let mut exprs = ExpressionArena::new_checked();
        let mut syms = SymbolArena::new_unchecked();
        let mut handler = mk_handler();

        let val_expr = lit_i64(&mut exprs, 2);
        let x_sym = syms.push(var_symbol(ty_int(&handler), val_expr));
        let x_id = ident(&mut exprs, x_sym);

        let ctx = Context {
            symbols: syms,
            expressions: exprs,
        };
        let ty = handler.type_expression(&ctx, &x_id);

        assert_eq!(ty, ty_int(&handler));
    }

    #[test]
    fn if_predicate_non_bool_literal_reports_error() {
        // if (2) { 10 } else { 20 }  -> error (predicate is int)
        let mut exprs = ExpressionArena::new_checked();
        let two = lit_i64(&mut exprs, 2);
        let then_e = lit_i64(&mut exprs, 10);
        let else_e = lit_i64(&mut exprs, 20);
        let ife = if_expr(&mut exprs, two, then_e, else_e);

        let ctx = Context {
            symbols: SymbolArena::new_unchecked(),
            expressions: exprs,
        };

        let mut handler = mk_handler();
        let before = handler.errors.len();
        let ty = handler.type_expression(&ctx, &ife);
        let integer_type = handler.types.intern(&Type::Integer);

        assert_eq!(
            ty, integer_type,
            "return type is integer regardless of bad predicate"
        );
        assert_eq!(handler.errors.len(), before + 1, "one type error expected");
    }

    #[test]
    fn if_predicate_non_bool_arithmetic_reports_error() {
        // if (2 + 2) { 1 } else { 0 } -> error (predicate is int)
        let mut exprs = ExpressionArena::new_checked();
        let a = lit_i64(&mut exprs, 2);
        let b = lit_i64(&mut exprs, 2);

        let c = lit_i64(&mut exprs, 1);
        let d = lit_i64(&mut exprs, 0);
        let sum = bin(&mut exprs, BinOp::Add, a, b);

        let ife = if_expr(&mut exprs, sum, c, d);

        let ctx = Context {
            symbols: SymbolArena::new_unchecked(),
            expressions: exprs,
        };

        let mut handler = mk_handler();
        let before = handler.errors.len();
        let ty = handler.type_expression(&ctx, &ife);
        let integer_type = handler.types.intern(&Type::Integer);

        assert_eq!(
            ty, integer_type,
            "return type is integer regardless of bad predicate"
        );
        assert_eq!(handler.errors.len(), before + 1, "one type error expected");
    }

    #[test]
    fn if_branch_type_mismatch_reports_error() {
        // if true { 3 } else { false } -> error (branches int vs bool)
        let mut exprs = ExpressionArena::new_checked();
        let pred = lit_bool(&mut exprs, true);
        let then_e = lit_i64(&mut exprs, 3);
        let else_e = lit_bool(&mut exprs, false);
        let ife = if_expr(&mut exprs, pred, then_e, else_e);

        let ctx = Context {
            symbols: SymbolArena::new_unchecked(),
            expressions: exprs,
        };

        let mut handler = mk_handler();
        let before = handler.errors.len();
        let ty = handler.type_expression(&ctx, &ife);

        assert_eq!(ty, ty_err(&handler));
        assert_eq!(handler.errors.len(), before + 1, "one type error expected");
    }

    #[test]
    fn if_simple_happy_path() {
        // if true { 2 } else { 3 } -> int
        let mut exprs = ExpressionArena::new_checked();
        let t = lit_bool(&mut exprs, true);
        let (a, b) = (lit_i64(&mut exprs, 2), lit_i64(&mut exprs, 3));

        let ife = if_expr(&mut exprs, t, a, b);

        let ctx = Context {
            symbols: SymbolArena::new_unchecked(),
            expressions: exprs,
        };

        let mut handler = mk_handler();
        let before = handler.errors.len();
        let ty = handler.type_expression(&ctx, &ife);

        assert_eq!(handler.errors.len(), before, "no type errors expected");
        assert_eq!(ty, ty_int(&handler));
    }

    #[test]
    fn if_nested_happy_path() {
        // if (2+2)==5 { 3 } else { if (2+2)==4 { 3 } else { 4 } } -> int
        let mut exprs = ExpressionArena::new_checked();

        let two = lit_i64(&mut exprs, 2);
        let two2 = lit_i64(&mut exprs, 2);
        let sum1 = bin(&mut exprs, BinOp::Add, two, two2);

        let (five, four, three) = (
            lit_i64(&mut exprs, 5),
            lit_i64(&mut exprs, 4),
            lit_i64(&mut exprs, 3),
        );

        let pred_outer = eq(&mut exprs, sum1, five); // bool

        let two3 = lit_i64(&mut exprs, 2);
        let two4 = lit_i64(&mut exprs, 2);
        let sum2 = bin(&mut exprs, BinOp::Add, two3, two4);
        let pred_inner = eq(&mut exprs, sum2, four); // bool

        let inner_if = if_expr(&mut exprs, pred_inner, three, four);
        let outer_if = if_expr(&mut exprs, pred_outer, three, inner_if);

        let ctx = Context {
            symbols: SymbolArena::new_unchecked(),
            expressions: exprs,
        };

        let mut handler = mk_handler();
        let before = handler.errors.len();
        let ty = handler.type_expression(&ctx, &outer_if);

        assert_eq!(handler.errors.len(), before, "no type errors expected");
        assert_eq!(ty, ty_int(&handler));
    }
}
