//! Type checking on the HIR

use calamars_core::ids::{self, ExpressionId};

use crate::{
    sematic::{
        error::SemanticError,
        hir::{self, take_inputs, type_id_stringify, GlobalContext, ItemId, Type},
    },
    syntax::span::Span,
};

/// `TypeHandler` is responsible for checking that the HIR's type semantics are correct.
///
/// It should identify errors such as `x :: Int` followed by `x = "hello"`, and return pretty diagnostics.
pub struct TypeHandler<'a> {
    pub module: &'a mut hir::Module,
    /// Collection of semantic errors encountered while type checking
    pub errors: Vec<SemanticError>,
}

impl<'a> TypeHandler<'a> {
    fn match_type(&mut self, id: ids::TypeId, ty: &Type, global_ctx: &mut GlobalContext) -> bool {
        let ty_id = global_ctx.types.intern(ty);
        id == ty_id
    }

    #[inline]
    fn push_wrong_type(
        &mut self,
        actual: ids::TypeId,
        expected: &str,
        span: Span,
        global_ctx: &GlobalContext,
    ) {
        self.errors.push(SemanticError::WrongType {
            actual: type_id_stringify(&global_ctx.types, actual),
            expected: expected.into(),
            span,
        });
    }

    #[inline]
    fn err_id(&self, global_ctx: &GlobalContext) -> ids::TypeId {
        global_ctx.types.err_id()
    }

    #[inline]
    fn intern_ty(&mut self, ty: &Type, global_ctx: &mut GlobalContext) -> ids::TypeId {
        global_ctx.types.intern(ty)
    }

    #[inline]
    /// Get an expression from the modules type arena without checking if the ID is valid. You must
    /// guarantee that the ID is valid when calling this function.
    fn get_expr_unchecked(&self, expr_id: ids::ExpressionId) -> &hir::Expr {
        self.module.exprs.get_unchecked(expr_id)
    }

    /// Check that some expression has a numerical type. Otherwise, log an error.
    fn ensure_numeric(
        &mut self,
        expr: ids::ExpressionId,
        t: ids::TypeId,
        global_ctx: &mut GlobalContext,
    ) {
        if self.match_type(t, &Type::Integer, global_ctx)
            || self.match_type(t, &Type::Float, global_ctx)
        {
            return;
        }
        let sp = self.get_expr_unchecked(expr).get_span().unwrap();
        self.push_wrong_type(t, "Numerical", sp, global_ctx);
    }

    /// Check that some expression has a correct type. Otherwise, log an error.
    fn ensure_type(
        &mut self,
        expr: ids::ExpressionId,
        ty_id: ids::TypeId,
        expected: ids::TypeId,
        global_ctx: &mut GlobalContext,
    ) {
        if ty_id == expected {
            return;
        }
        let sp = self.get_expr_unchecked(expr).get_span().unwrap();
        self.push_wrong_type(
            ty_id,
            &type_id_stringify(&global_ctx.types, expected),
            sp,
            global_ctx,
        );
    }

    /// Given some expression, this will return it's type id. If there are any semantic typing
    /// errors, for example: `2 + "hello"` they will be added to the `errors` vector, and will
    /// return the typeid of the `Error` type.
    ///
    /// This function will also memoize each expressions type id to the `expression_types` map in
    /// the module.
    fn type_expression(
        &mut self,
        e_id: &ids::ExpressionId,
        global_ctx: &mut GlobalContext,
    ) -> ids::TypeId {
        if let Some(ty) = self.module.expression_types.get(e_id) {
            return *ty;
        }

        let expression = self.get_expr_unchecked(*e_id).clone();
        let type_id = match &expression {
            hir::Expr::Err => self.err_id(global_ctx),
            hir::Expr::Literal { constant, .. } => {
                let ty = match constant {
                    hir::Const::I64(_) => Type::Integer,
                    hir::Const::Bool(_) => Type::Boolean,
                    hir::Const::String(_) => Type::String,
                };
                *global_ctx.types.resolve_unchecked(&ty)
            }
            hir::Expr::Identifier { id, .. } => self
                .module
                .symbols
                .get(*id)
                .map(|s| s.ty)
                .unwrap_or(self.err_id(global_ctx)),
            hir::Expr::BinaryOperation {
                operator,
                lhs,
                rhs,
                span,
            } => self.type_check_binary_ops(operator, lhs, rhs, *span, global_ctx),
            hir::Expr::Call { f, input, span } => {
                let function_ty = self.type_expression(f, global_ctx);
                let input_ty = self.type_expression(input, global_ctx);
                let Ok((in_ty, out_ty)) = take_inputs(function_ty, 1, global_ctx, *span) else {
                    let err = SemanticError::NonCallable {
                        msg: "non-callable being called",
                        span: *span,
                    };
                    self.errors.push(err);
                    return self.err_id(global_ctx);
                };

                if in_ty[0] != input_ty {
                    let expected_type_str = type_id_stringify(&global_ctx.types, in_ty[0]);
                    // TODO: We should really have f span and input span be separate ...
                    self.push_wrong_type(input_ty, expected_type_str.as_str(), *span, global_ctx);
                }
                out_ty
            }
            hir::Expr::Block {
                items, final_expr, ..
            } => self.type_check_block(&items, final_expr, global_ctx),
            hir::Expr::If {
                predicate,
                then,
                otherwise,
                then_span,
                othewise_span,
                ..
            } => {
                // Make sure that the predicate is a boolean
                let p_ty = self.type_expression(predicate, global_ctx);
                if p_ty != self.err_id(global_ctx) {
                    let bool = self.intern_ty(&Type::Boolean, global_ctx);
                    self.ensure_type(*predicate, p_ty, bool, global_ctx);
                }

                // Make sure that if and else branches return the same
                let t_ty = self.type_expression(then, global_ctx);
                let o_ty = self.type_expression(otherwise, global_ctx);

                if t_ty == self.err_id(global_ctx) || o_ty == self.err_id(global_ctx) {
                    return self.err_id(global_ctx);
                }

                if t_ty != o_ty {
                    self.errors.push(SemanticError::MismatchedIfBranches {
                        then_span: *then_span,
                        then_return: type_id_stringify(&global_ctx.types, t_ty),
                        else_span: *othewise_span,
                        else_return: type_id_stringify(&global_ctx.types, o_ty),
                    });
                    return self.err_id(global_ctx);
                }

                // If both branches return the same, then return that type
                t_ty
            }
        };

        self.module.expression_types.insert(*e_id, type_id);
        type_id
    }

    fn type_check_binary_ops(
        &mut self,
        op: &hir::BinOp,
        lhs: &ExpressionId,
        rhs: &ExpressionId,
        span: Span,
        global_ctx: &mut GlobalContext,
    ) -> ids::TypeId {
        let lhs_type_id = self.type_expression(lhs, global_ctx);
        let rhs_type_id = self.type_expression(rhs, global_ctx);

        let error_id = self.err_id(global_ctx);
        if lhs_type_id == error_id || rhs_type_id == error_id {
            return error_id;
        }

        let int_type_id = *global_ctx.types.resolve_unchecked(&Type::Integer);
        let float_type_id = *global_ctx.types.resolve_unchecked(&Type::Float);

        match op {
            hir::BinOp::Add | hir::BinOp::Sub | hir::BinOp::Mult | hir::BinOp::Div => {
                self.ensure_numeric(*lhs, lhs_type_id, global_ctx);
                self.ensure_numeric(*rhs, rhs_type_id, global_ctx);

                let lhs_numerical = (lhs_type_id == float_type_id) || (lhs_type_id == int_type_id);
                let rhs_numerical = (rhs_type_id == float_type_id) || (rhs_type_id == int_type_id);

                // If they are not both numerical, then this is an error
                if !(lhs_numerical && rhs_numerical) {
                    return self.err_id(global_ctx);
                }

                // If they are both integers, then we will return integer
                if lhs_type_id == int_type_id && rhs_type_id == int_type_id {
                    return int_type_id;
                }

                // Both floats, or one float, then we cast to float
                float_type_id
            }
            hir::BinOp::EqEq | hir::BinOp::NotEqual => {
                if lhs_type_id == self.err_id(global_ctx) || rhs_type_id == self.err_id(global_ctx)
                {
                    return self.err_id(global_ctx);
                }

                if lhs_type_id != rhs_type_id {
                    let rhs_expr = self.get_expr_unchecked(*rhs);
                    self.errors.push(SemanticError::WrongType {
                        expected: type_id_stringify(&global_ctx.types, lhs_type_id),
                        actual: type_id_stringify(&global_ctx.types, rhs_type_id),
                        // We can unwrap since we made sure its not error type
                        span: rhs_expr.get_span().unwrap(),
                    });
                }

                self.intern_ty(&Type::Boolean, global_ctx)
            }
            hir::BinOp::Mod => {
                self.ensure_type(*lhs, lhs_type_id, int_type_id, global_ctx);
                self.ensure_type(*rhs, rhs_type_id, int_type_id, global_ctx);

                if lhs_type_id != int_type_id || rhs_type_id != int_type_id {
                    error_id
                } else {
                    self.intern_ty(&Type::Integer, global_ctx)
                }
            }
            hir::BinOp::Greater | hir::BinOp::Geq | hir::BinOp::Less | hir::BinOp::Leq => {
                self.ensure_numeric(*lhs, lhs_type_id, global_ctx);
                self.ensure_numeric(*rhs, rhs_type_id, global_ctx);
                self.intern_ty(&Type::Boolean, global_ctx)
            }
            hir::BinOp::And | hir::BinOp::Or | hir::BinOp::Xor => {
                if self.match_type(lhs_type_id, &Type::Integer, global_ctx)
                    && self.match_type(rhs_type_id, &Type::Integer, global_ctx)
                {
                    self.intern_ty(&Type::Integer, global_ctx)
                } else if self.match_type(lhs_type_id, &Type::Boolean, global_ctx)
                    && self.match_type(rhs_type_id, &Type::Boolean, global_ctx)
                {
                    self.intern_ty(&Type::Boolean, global_ctx)
                } else {
                    // FIXME: Show an error here
                    error_id
                }
            }
        }
    }

    fn type_check_block(
        &mut self,
        items: &[ItemId],
        final_expr: &Option<ids::ExpressionId>,
        global_ctx: &mut GlobalContext,
    ) -> ids::TypeId {
        // Start by analysing each of the items
        for item in items {
            let _ = match item {
                ItemId::Expr(expression_id) => {
                    self.type_expression(&expression_id, global_ctx);
                }
                ItemId::Symbol(symbol_id) => {
                    self.type_check_declaration(*symbol_id, global_ctx);
                }
            };
        }

        // If there is no final expression, then we will return the unit type
        let unit = self.intern_ty(&Type::Unit, global_ctx);
        final_expr
            .map(|e_id| self.type_expression(&e_id, global_ctx))
            .unwrap_or(unit)
    }

    /// When declaring a function, check that the body of the function returns the type expected in
    /// the function signature.
    pub fn type_check_function_declaration(
        &mut self,
        name_span: Span,
        body: ids::ExpressionId,
        expected_type: ids::TypeId,
        global_ctx: &mut GlobalContext,
    ) {
        let body_ty = self.type_expression(&body, global_ctx);
        if body_ty != expected_type && body_ty != self.err_id(global_ctx) {
            let body = self.get_expr_unchecked(body);
            self.errors.push(SemanticError::FnWrongReturnType {
                expected: type_id_stringify(&global_ctx.types, expected_type),
                // none for now, but it really shuold not be none ... We need to improve spans
                return_type_span: None,
                fn_name_span: name_span,
                actual: type_id_stringify(&global_ctx.types, body_ty),
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
        global_ctx: &mut GlobalContext,
    ) {
        let body_ty = self.type_expression(&body, global_ctx);
        if body_ty != expected_type && body_ty != self.err_id(global_ctx) {
            let body = self.get_expr_unchecked(body);
            self.errors.push(SemanticError::BindingWrongType {
                expected: type_id_stringify(&global_ctx.types, expected_type),
                return_type_span: name_span,
                actual: type_id_stringify(&global_ctx.types, body_ty),
                return_span: body.get_span(),
                body_span: body.get_span().unwrap(),
            })
        }
    }

    /// Make sure that a declarations types make sense semantically.
    pub fn type_check_declaration(&mut self, dec: ids::SymbolId, global_ctx: &mut GlobalContext) {
        let hir::Symbol { ty, name, kind } = self.module.symbols.get(dec).unwrap();
        let hir::SymbolKind::Defn {
            span_decl,
            declaration,
            ..
        } = kind
        else {
            return;
        };

        let span_decl = span_decl.clone();
        let body_id = declaration.body;
        let arity = declaration.inputs.len();
        let output_type = match take_inputs(*ty, arity, global_ctx, span_decl) {
            Ok((_, output_type)) => output_type,
            Err(err) => {
                self.errors.push(err);
                return;
            }
        };
        let body_actual_type = self.type_expression(&body_id, global_ctx);
        if output_type != body_actual_type {
            let expected_str = type_id_stringify(&global_ctx.types, output_type);
            self.push_wrong_type(body_actual_type, &expected_str, span_decl, global_ctx);
        }
    }

    /// Type check all declarations in the module.
    pub fn type_check_module(&mut self, global_ctx: &mut GlobalContext) {
        let roots_len = self.module.roots.len();
        for idx in 0..roots_len {
            let symbol = self.module.roots[idx];
            self.type_check_declaration(symbol, global_ctx);
        }
    }
}
