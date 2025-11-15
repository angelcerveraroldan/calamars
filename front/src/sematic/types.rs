//! Type checking on the HIR

use std::any::Any;

use calamars_core::{
    MaybeErr,
    ids::{self, ExpressionId, SymbolId},
};

use crate::{
    sematic::{
        error::SemanticError,
        hir::{
            self, ExpressionArena, IdentArena, Symbol, SymbolArena, Type, TypeArena,
            type_id_stringify,
        },
    },
    syntax::{
        ast::{Declaration, Expression},
        span::Span,
    },
};

/// `TypeHandler` is responsible for checking that the HIR's type semantics are correct.
///
/// It should identify errors such as `val x: Int = "hello"`, and return pretty diagnostics.
pub struct TypeHandler<'a> {
    pub module: &'a mut hir::Module,
    pub errors: Vec<SemanticError>,
}

impl<'a> TypeHandler<'a> {
    fn match_type(&mut self, id: ids::TypeId, ty: &Type) -> bool {
        let ty_id = self.module.types.intern(ty);
        id == ty_id
    }

    #[inline]
    fn push_wrong_type(&mut self, actual: ids::TypeId, expected: &str, span: Span) {
        self.errors.push(SemanticError::WrongType {
            actual: type_id_stringify(&self.module.types, actual),
            expected: expected.into(),
            span,
        });
    }

    fn ensure_numeric(&mut self, expr: ids::ExpressionId, t: ids::TypeId) {
        if self.match_type(t, &Type::Integer) || self.match_type(t, &Type::Float) {
            return;
        }
        let sp = self.module.exprs.get_unchecked(expr).get_span().unwrap();
        self.push_wrong_type(t, "Numerical", sp);
    }

    fn ensure_type(&mut self, expr: ids::ExpressionId, ty_id: ids::TypeId, expected: ids::TypeId) {
        if ty_id == expected {
            return;
        }
        let sp = self.module.exprs.get_unchecked(expr).get_span().unwrap();
        self.push_wrong_type(ty_id, &type_id_stringify(&self.module.types, expected), sp);
    }

    /// Given some expression, this will return it's type id. If there are any semantic typing
    /// errors, for example: `2 + "hello"` they will be added to the `errors` vector, and will
    /// return the typeid of the `Error` type.
    ///
    /// TODO: Insert the typeid to a map, so that later we can check the type of an expression id
    fn type_expression(&mut self, e_id: &ids::ExpressionId) -> ids::TypeId {
        if let Some(ty) = self.module.expression_types.get(e_id) {
            return *ty;
        }

        let expression = self.module.exprs.get_unchecked(*e_id).clone();
        let type_id = match &expression {
            hir::Expr::Err => self.module.types.err_id(),
            hir::Expr::Literal { constant, .. } => {
                let ty = match constant {
                    hir::Const::I64(_) => Type::Integer,
                    hir::Const::Bool(_) => Type::Boolean,
                    hir::Const::String(_) => Type::String,
                };
                *self.module.types.resolve_unchecked(&ty)
            }
            hir::Expr::Identifier { id, .. } => self
                .module
                .symbols
                .get(*id)
                .map(|s| s.ty_id())
                .unwrap_or(self.module.types.err_id()),
            hir::Expr::BinaryOperation {
                operator,
                lhs,
                rhs,
                span,
            } => self.type_check_binary_ops(operator, lhs, rhs, *span),
            hir::Expr::Call { f, inputs, span } => self.type_check_fncall(f, inputs, *span),
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
                let p_ty = self.type_expression(predicate);
                if (p_ty != self.module.types.err_id()) {
                    let bool = self.module.types.intern(&Type::Boolean);
                    self.ensure_type(*predicate, p_ty, bool);
                }

                // Make sure that if and else branches return the same
                let t_ty = self.type_expression(then);
                let o_ty = self.type_expression(otherwise);

                if (t_ty == self.module.types.err_id() || o_ty == self.module.types.err_id()) {
                    return self.module.types.err_id();
                }

                if (t_ty != o_ty) {
                    self.errors.push(SemanticError::MismatchedIfBranches {
                        then_span: *then_span,
                        then_return: type_id_stringify(&self.module.types, t_ty),
                        else_span: *othewise_span,
                        else_return: type_id_stringify(&self.module.types, o_ty),
                    });
                    return self.module.types.err_id();
                }

                // If both branches retur the same, then return that type
                t_ty
            }
        };

        self.module.expression_types.insert(*e_id, type_id);
        type_id
    }

    fn type_check_fncall(
        &mut self,
        f: &ExpressionId,
        inputs: &Box<[ExpressionId]>,
        span: Span,
    ) -> ids::TypeId {
        let expression_ty = self.type_expression(f);
        // TODO: Handle error type separately here
        let (out_ty, in_tys) = match self.module.types.get_unchecked(expression_ty) {
            Type::Function { output, input } => (*output, input.clone()),
            otherwise => {
                self.errors.push(SemanticError::NonCallable {
                    msg: "Expected callable",
                    span: span,
                });
                return self.module.types.err_id();
            }
        };

        // Airity check
        if inputs.len() != in_tys.len() {
            self.errors.push(SemanticError::ArityError {
                expected: in_tys.len(),
                actual: inputs.len(),
                span: span,
            });

            return out_ty;
        }

        // Check that there are no input errors when calling the function
        for (actual, exp) in inputs.into_iter().zip(in_tys) {
            let acc_ty = self.type_expression(&actual);
            let acc_expr = self.module.exprs.get_unchecked(*actual);
            if acc_ty != exp && acc_ty != self.module.types.err_id() {
                self.errors.push(SemanticError::WrongType {
                    expected: type_id_stringify(&self.module.types, exp),
                    actual: type_id_stringify(&self.module.types, acc_ty),
                    span: acc_expr.get_span().unwrap(),
                });
            }
        }

        out_ty
    }

    fn type_check_binary_ops(
        &mut self,
        op: &hir::BinOp,
        lhs: &ExpressionId,
        rhs: &ExpressionId,
        span: Span,
    ) -> ids::TypeId {
        let lhs_type_id = self.type_expression(lhs);
        let rhs_type_id = self.type_expression(rhs);

        let error_id = self.module.types.err_id();
        if (lhs_type_id == error_id || rhs_type_id == error_id) {
            return error_id;
        }

        let int_type_id = *self.module.types.resolve_unchecked(&Type::Integer);
        let float_type_id = *self.module.types.resolve_unchecked(&Type::Float);

        match op {
            // Both numerical
            hir::BinOp::Add | hir::BinOp::Sub | hir::BinOp::Mult | hir::BinOp::Div => {
                self.ensure_numeric(*lhs, lhs_type_id);
                self.ensure_numeric(*rhs, rhs_type_id);

                let lhs_numerical = (lhs_type_id == float_type_id) || (lhs_type_id == int_type_id);
                let rhs_numerical = (rhs_type_id == float_type_id) || (rhs_type_id == int_type_id);

                // If they are not both numerica, then this is an error
                if !(lhs_numerical && rhs_numerical) {
                    return self.module.types.err_id();
                }

                // If they are both integers, then we will return integer
                if (lhs_type_id == int_type_id && rhs_type_id == int_type_id) {
                    return int_type_id;
                }

                // Both floats, or one float, then we cast to float
                float_type_id
            }
            hir::BinOp::EqEq => {
                if (lhs_type_id == self.module.types.err_id()
                    || rhs_type_id == self.module.types.err_id())
                {
                    return self.module.types.err_id();
                }

                if (lhs_type_id != rhs_type_id) {
                    let rhs_expr = self.module.exprs.get_unchecked(*rhs);
                    self.errors.push(SemanticError::WrongType {
                        expected: type_id_stringify(&self.module.types, lhs_type_id),
                        actual: type_id_stringify(&self.module.types, rhs_type_id),
                        // We can unwrap since we made sure its not error type
                        span: rhs_expr.get_span().unwrap(),
                    });
                }
                self.module.types.err_id()
            }
            // Both integers
            hir::BinOp::Mod => {
                self.ensure_type(*lhs, lhs_type_id, int_type_id);
                self.ensure_type(*rhs, rhs_type_id, int_type_id);
                self.module.types.err_id()
            }
        }
    }

    pub fn type_check_function_declaration(
        &mut self,
        name_span: Span,
        body: ids::ExpressionId,
        expected_type: ids::TypeId,
    ) {
        let body_ty = self.type_expression(&body);
        if body_ty != expected_type && body_ty != self.module.types.err_id() {
            let body = self.module.exprs.get_unchecked(body);
            self.errors.push(SemanticError::FnWrongReturnType {
                expected: type_id_stringify(&self.module.types, expected_type),
                // none for now, but it really shuold not be none ... We need to improve spans
                return_type_span: None,
                fn_name_span: name_span,
                actual: type_id_stringify(&self.module.types, body_ty),
                return_span: body.get_span(),
                body_span: body.get_span().unwrap(),
            });
        }
    }

    pub fn type_check_variable_declaration(
        &mut self,
        name_span: Span,
        body: ids::ExpressionId,
        expected_type: ids::TypeId,
    ) {
        let body_ty = self.type_expression(&body);
        if body_ty != expected_type && body_ty != self.module.types.err_id() {
            let body = self.module.exprs.get_unchecked(body);
            self.errors.push(SemanticError::BindingWrongType {
                expected: type_id_stringify(&self.module.types, expected_type),
                return_type_span: name_span,
                actual: type_id_stringify(&self.module.types, body_ty),
                return_span: body.get_span(),
                body_span: body.get_span().unwrap(),
            })
        }
    }

    pub fn type_check_declaration(&mut self, dec: ids::SymbolId) {
        let sym = self.module.symbols.get_unchecked(dec);
        match &sym.kind {
            hir::SymbolKind::Function { body, .. } => {
                let expected_ty = sym.ty_id();
                let ty = self.module.types.get_unchecked(expected_ty);
                self.type_check_function_declaration(sym.name_span(), *body, ty.function_output())
            }
            hir::SymbolKind::Variable { body, .. } => {
                let expected_ty = sym.ty_id();
                self.type_check_variable_declaration(sym.name_span(), *body, expected_ty);
            }
            hir::SymbolKind::FunctionUndeclared { .. }
            | hir::SymbolKind::VariableUndeclared { .. } => {
                panic!("Undeclared should not exist after lowering")
            }
            _ => {}
        }
    }

    pub fn type_check_module(&mut self) {
        self.module
            .roots
            .clone()
            .into_iter()
            .for_each(|symbol| self.type_check_declaration(symbol));
    }
}

#[cfg(test)]
mod tests {
    use crate::sematic::hir::ConstantStringArena;

    use super::hir::{BinOp, Const, Expr, Symbol, SymbolKind};
    use super::ids;
    use super::*;

    fn mk_module(
        exprs: ExpressionArena,
        syms: SymbolArena,
        consts: ConstantStringArena,
    ) -> hir::Module {
        let mut types = TypeArena::new_checked();
        types.intern(&Type::Integer);
        types.intern(&Type::Float);
        types.intern(&Type::Boolean);
        types.intern(&Type::String);

        hir::Module {
            id: ids::FileId::from(0),
            name: "TestFile".to_owned(),
            types,
            const_str: consts,
            idents: IdentArena::new_unchecked(),
            symbols: syms,
            exprs,
            roots: Box::new([]),
            expression_types: hashbrown::HashMap::new(),
        }
    }

    fn ty_int(handler: &TypeHandler) -> ids::TypeId {
        *handler.module.types.resolve_unchecked(&Type::Integer)
    }

    fn ty_float(handler: &TypeHandler) -> ids::TypeId {
        *handler.module.types.resolve_unchecked(&Type::Float)
    }

    fn ty_bool(handler: &TypeHandler) -> ids::TypeId {
        *handler.module.types.resolve_unchecked(&Type::Boolean)
    }

    fn ty_str(handler: &TypeHandler) -> ids::TypeId {
        *handler.module.types.resolve_unchecked(&Type::String)
    }

    fn ty_err(handler: &TypeHandler) -> ids::TypeId {
        handler.module.types.err_id()
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

        let syms = SymbolArena::new_unchecked();
        let mut module = mk_module(exprs, syms, consts);

        let mut handler = TypeHandler {
            module: &mut module,
            errors: vec![],
        };

        let before = handler.errors.len();
        let ty = handler.type_expression(&add);

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
        let mut handler = mk_module(
            ExpressionArena::new_checked(),
            SymbolArena::new_unchecked(),
            ConstantStringArena::new_unchecked(),
        );

        let mut handler = TypeHandler {
            module: &mut handler,
            errors: vec![],
        };

        let float_ty = ty_float(&handler);
        let exprs = &mut handler.module.exprs;

        let one = lit_i64(exprs, 1);

        let dummy_body = one;
        let y_sym_id = handler
            .module
            .symbols
            .push(var_symbol(float_ty, dummy_body));

        let y_ident = ident(exprs, y_sym_id);

        let add1 = bin(exprs, BinOp::Add, one, y_ident);
        let add2 = bin(exprs, BinOp::Add, y_ident, one);

        let t1 = handler.type_expression(&add1);
        let t2 = handler.type_expression(&add2);

        assert_eq!(t1, ty_float(&handler));
        assert_eq!(t2, ty_float(&handler));
    }

    #[test]
    fn mult_int_int_is_int() {
        let mut exprs = ExpressionArena::new_checked();

        let a = lit_i64(&mut exprs, 1);
        let b = lit_i64(&mut exprs, 2);
        let mul = bin(&mut exprs, BinOp::Mult, a, b);

        let mut module = mk_module(
            exprs,
            SymbolArena::new_unchecked(),
            ConstantStringArena::new_unchecked(),
        );

        let mut handler = TypeHandler {
            module: &mut module,
            errors: vec![],
        };

        let ty = handler.type_expression(&mul);
        assert_eq!(ty, ty_int(&handler));
    }

    #[test]
    fn literals_have_expected_types() {
        let mut consts = ConstantStringArena::new_unchecked();
        let mut exprs = ExpressionArena::new_checked();

        let i = lit_i64(&mut exprs, 42);
        let b = lit_bool(&mut exprs, true);
        let s = lit_str(&mut exprs, "x", &mut consts);

        let mut module = mk_module(exprs, SymbolArena::new_unchecked(), consts);
        let mut handler = TypeHandler {
            module: &mut module,
            errors: vec![],
        };

        assert_eq!(handler.type_expression(&i), ty_int(&handler));
        assert_eq!(handler.type_expression(&b), ty_bool(&handler));
        assert_eq!(handler.type_expression(&s), ty_str(&handler));
    }

    #[test]
    fn identifier_uses_symbol_type() {
        let mut exprs = ExpressionArena::new_checked();
        let mut syms = SymbolArena::new_unchecked();

        let mut module = mk_module(exprs, syms, ConstantStringArena::new_unchecked());
        let mut handler = TypeHandler {
            module: &mut module,
            errors: vec![],
        };

        let val_expr = lit_i64(&mut handler.module.exprs, 2);
        let x_sym = handler
            .module
            .symbols
            .push(var_symbol(ty_int(&handler), val_expr));
        let x_id = ident(&mut handler.module.exprs, x_sym);

        let ty = handler.type_expression(&x_id);
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

        let mut module = mk_module(
            exprs,
            SymbolArena::new_unchecked(),
            ConstantStringArena::new_unchecked(),
        );
        let mut handler = TypeHandler {
            module: &mut module,
            errors: vec![],
        };

        let before = handler.errors.len();
        let ty = handler.type_expression(&ife);
        let integer_type = handler.module.types.intern(&Type::Integer);

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

        let mut module = mk_module(
            exprs,
            SymbolArena::new_unchecked(),
            ConstantStringArena::new_unchecked(),
        );
        let mut handler = TypeHandler {
            module: &mut module,
            errors: vec![],
        };

        let before = handler.errors.len();
        let ty = handler.type_expression(&ife);
        let integer_type = handler.module.types.intern(&Type::Integer);

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

        let mut module = mk_module(
            exprs,
            SymbolArena::new_unchecked(),
            ConstantStringArena::new_unchecked(),
        );
        let mut handler = TypeHandler {
            module: &mut module,
            errors: vec![],
        };

        let before = handler.errors.len();
        let ty = handler.type_expression(&ife);

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

        let mut module = mk_module(
            exprs,
            SymbolArena::new_unchecked(),
            ConstantStringArena::new_unchecked(),
        );
        let mut handler = TypeHandler {
            module: &mut module,
            errors: vec![],
        };
        let before = handler.errors.len();
        let ty = handler.type_expression(&ife);

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

        let mut module = mk_module(
            exprs,
            SymbolArena::new_unchecked(),
            ConstantStringArena::new_unchecked(),
        );
        let mut handler = TypeHandler {
            module: &mut module,
            errors: vec![],
        };
        let before = handler.errors.len();
        let ty = handler.type_expression(&outer_if);

        assert_eq!(handler.errors.len(), before, "no type errors expected");
        assert_eq!(ty, ty_int(&handler));
    }
}
