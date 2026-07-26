//! Type checking on the HIR

use calamars_core::{
    data_structs::StructDef,
    global::TypeCtx,
    ids::{self, ExpressionId},
    types,
};

use crate::{
    sematic::{
        error::SemanticError,
        hir::{self, ItemId, take_inputs},
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
    fn match_type(
        &mut self,
        id: ids::TypeId,
        ty: &types::Type,
        type_arena: &mut calamars_core::types::TypeArena,
    ) -> bool {
        let ty_id = type_arena.intern(ty);
        id == ty_id
    }

    #[inline]
    fn push_wrong_type(
        &mut self,
        actual: ids::TypeId,
        expected: &str,
        span: Span,
        type_arena: &calamars_core::types::TypeArena,
    ) {
        self.errors.push(SemanticError::WrongType {
            actual: types::type_id_stringify(type_arena, actual),
            expected: expected.into(),
            span,
        });
    }

    #[inline]
    fn err_id(&self, type_arena: &calamars_core::types::TypeArena) -> ids::TypeId {
        type_arena.err_id()
    }

    #[inline]
    fn intern_ty(
        &mut self,
        ty: &types::Type,
        type_arena: &mut calamars_core::types::TypeArena,
    ) -> ids::TypeId {
        type_arena.intern(ty)
    }

    #[inline]
    /// Get an expression from the modules type arena without checking if the ID is valid. You must
    /// guarantee that the ID is valid when calling this function.
    fn get_expr_unchecked(&self, expr_id: ids::ExpressionId) -> &hir::Expr {
        self.module.exprs.get_unchecked(expr_id)
    }

    /// Check that some expression has a numerical type. Otherwise, log an error.
    fn ensure_numeric(&mut self, expr: ids::ExpressionId, t: ids::TypeId, ctx: &mut TypeCtx) {
        if self.match_type(t, &types::Type::Integer, ctx.types)
            || self.match_type(t, &types::Type::Float, ctx.types)
        {
            return;
        }
        let sp = self.get_expr_unchecked(expr).get_span().unwrap();
        self.push_wrong_type(t, "Numerical", sp, ctx.types);
    }

    /// Check that some expression has a correct type. Otherwise, log an error.
    fn ensure_type(
        &mut self,
        expr: ids::ExpressionId,
        ty_id: ids::TypeId,
        expected: ids::TypeId,
        type_arena: &calamars_core::types::TypeArena,
    ) {
        if ty_id == expected {
            return;
        }
        let sp = self.get_expr_unchecked(expr).get_span().unwrap();
        self.push_wrong_type(
            ty_id,
            &types::type_id_stringify(type_arena, expected),
            sp,
            type_arena,
        );
    }

    /// Given some expression, this will return it's type id. If there are any semantic typing
    /// errors, for example: `2 + "hello"` they will be added to the `errors` vector, and will
    /// return the typeid of the `Error` type.
    ///
    /// This function will also memoize each expressions type id to the `expression_types` map in
    /// the module.
    fn type_expression(&mut self, e_id: &ids::ExpressionId, ctx: &mut TypeCtx) -> ids::TypeId {
        if let Some(ty) = self.module.expression_types.get(e_id) {
            return *ty;
        }

        let expression = self.get_expr_unchecked(*e_id).clone();
        let type_id = match &expression {
            hir::Expr::Err => self.err_id(ctx.types),
            hir::Expr::Literal { constant, .. } => {
                let ty = match constant {
                    hir::Const::I64(_) => types::Type::Integer,
                    hir::Const::Bool(_) => types::Type::Boolean,
                    hir::Const::String(_) => types::Type::String,
                };
                *ctx.types.resolve_unchecked(&ty)
            }
            hir::Expr::Identifier { id, .. } => self
                .module
                .symbols
                .get(*id)
                .map(|s| s.ty)
                .unwrap_or(self.err_id(ctx.types)),
            hir::Expr::BinaryOperation {
                operator,
                lhs,
                rhs,
                span,
            } => self.type_check_binary_ops(operator, lhs, rhs, *span, ctx),
            hir::Expr::Call { f, input, span } => {
                let function_ty = self.type_expression(f, ctx);
                let input_ty = self.type_expression(input, ctx);
                let Ok((in_ty, out_ty)) = take_inputs(function_ty, 1, ctx.types, *span) else {
                    let err = SemanticError::NonCallable {
                        msg: "non-callable being called",
                        span: *span,
                    };
                    self.errors.push(err);
                    return self.err_id(ctx.types);
                };

                if in_ty[0] != input_ty {
                    let expected_type_str = types::type_id_stringify(ctx.types, in_ty[0]);
                    // TODO: We should really have f span and input span be separate ...
                    self.push_wrong_type(input_ty, expected_type_str.as_str(), *span, ctx.types);
                }
                out_ty
            }
            hir::Expr::Block {
                items, final_expr, ..
            } => self.type_check_block(&items, final_expr, ctx),
            hir::Expr::If {
                predicate,
                then,
                otherwise,
                then_span,
                othewise_span,
                ..
            } => {
                // Make sure that the predicate is a boolean
                let p_ty = self.type_expression(predicate, ctx);
                if p_ty != self.err_id(ctx.types) {
                    let bool = self.intern_ty(&types::Type::Boolean, ctx.types);
                    self.ensure_type(*predicate, p_ty, bool, ctx.types);
                }

                // Make sure that if and else branches return the same
                let t_ty = self.type_expression(then, ctx);
                let o_ty = self.type_expression(otherwise, ctx);

                if t_ty == self.err_id(ctx.types) || o_ty == self.err_id(ctx.types) {
                    return self.err_id(ctx.types);
                }

                if t_ty != o_ty {
                    self.errors.push(SemanticError::MismatchedIfBranches {
                        then_span: *then_span,
                        then_return: types::type_id_stringify(ctx.types, t_ty),
                        else_span: *othewise_span,
                        else_return: types::type_id_stringify(ctx.types, o_ty),
                    });
                    return self.err_id(ctx.types);
                }

                // If both branches return the same, then return that type
                t_ty
            }
            hir::Expr::StructInit {
                struct_id,
                fields,
                span,
            } => {
                let fields: Vec<_> = fields
                    .iter()
                    .map(|(name, exprid)| (name, self.type_expression(exprid, ctx)))
                    .collect();

                let struct_def: &StructDef = ctx.struct_defs.get_unchecked(*struct_id);
                let expected_params = &struct_def.fields;

                // check that all the params are the correct type
                for (fname, exp_type) in fields {
                    let Some(expected_field) = expected_params.iter().find(|x| &x.name == fname)
                    else {
                        // TODO: Add span - better error handing needed
                        self.errors.push(SemanticError::StructFieldNotFound {
                            span: Span::from(0..0),
                            name: fname.clone(),
                        });
                        continue;
                    };

                    let expected_type = expected_field.ty;
                    if expected_type != exp_type {
                        let expected_str = types::type_id_stringify(ctx.types, expected_type);
                        self.push_wrong_type(exp_type, &expected_str, *span, ctx.types);
                    }
                }

                let ty = types::Type::Structure(*struct_id);
                *ctx.types.resolve_unchecked(&ty)
            }
            hir::Expr::StructFieldAccess {
                struct_expr,
                struct_span,
                field_name,
                field_span,
            } => {
                // first we make sure that we really have a struct ...
                let struct_tyid = self.type_expression(struct_expr, ctx);
                let calamars_core::types::Type::Structure(struct_id) =
                    ctx.types.get_unchecked(struct_tyid)
                else {
                    let expr_type = types::type_id_stringify(ctx.types, struct_tyid);
                    self.errors.push(SemanticError::CannotGetFieldOfNonStruct {
                        expr_span: *struct_span,
                        field_span: *field_span,
                        expr_type,
                    });
                    return self.err_id(ctx.types);
                };

                let ds = ctx.struct_defs.get_unchecked(*struct_id);
                match ds.fields.iter().find(|field| field.name == *field_name) {
                    Some(field) => field.ty,
                    None => {
                        self.errors.push(SemanticError::StructFieldNotFound {
                            span: *field_span,
                            name: field_name.clone(),
                        });
                        self.err_id(ctx.types)
                    }
                }
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
        ctx: &mut TypeCtx,
    ) -> ids::TypeId {
        let lhs_type_id = self.type_expression(lhs, ctx);
        let rhs_type_id = self.type_expression(rhs, ctx);

        let error_id = self.err_id(ctx.types);
        if lhs_type_id == error_id || rhs_type_id == error_id {
            return error_id;
        }

        let int_type_id = *ctx.types.resolve_unchecked(&types::Type::Integer);
        let float_type_id = *ctx.types.resolve_unchecked(&types::Type::Float);

        match op {
            hir::BinOp::Add | hir::BinOp::Sub | hir::BinOp::Mult | hir::BinOp::Div => {
                self.ensure_numeric(*lhs, lhs_type_id, ctx);
                self.ensure_numeric(*rhs, rhs_type_id, ctx);

                let lhs_numerical = (lhs_type_id == float_type_id) || (lhs_type_id == int_type_id);
                let rhs_numerical = (rhs_type_id == float_type_id) || (rhs_type_id == int_type_id);

                // If they are not both numerical, then this is an error
                if !(lhs_numerical && rhs_numerical) {
                    return self.err_id(ctx.types);
                }

                // If they are both integers, then we will return integer
                if lhs_type_id == int_type_id && rhs_type_id == int_type_id {
                    return int_type_id;
                }

                // Both floats, or one float, then we cast to float
                float_type_id
            }
            hir::BinOp::EqEq | hir::BinOp::NotEqual => {
                if lhs_type_id == self.err_id(ctx.types) || rhs_type_id == self.err_id(ctx.types) {
                    return self.err_id(ctx.types);
                }

                if lhs_type_id != rhs_type_id {
                    let rhs_expr = self.get_expr_unchecked(*rhs);
                    self.errors.push(SemanticError::WrongType {
                        expected: types::type_id_stringify(ctx.types, lhs_type_id),
                        actual: types::type_id_stringify(ctx.types, rhs_type_id),
                        // We can unwrap since we made sure its not error type
                        span: rhs_expr.get_span().unwrap(),
                    });
                }

                self.intern_ty(&types::Type::Boolean, ctx.types)
            }
            hir::BinOp::Mod => {
                self.ensure_type(*lhs, lhs_type_id, int_type_id, ctx.types);
                self.ensure_type(*rhs, rhs_type_id, int_type_id, ctx.types);

                if lhs_type_id != int_type_id || rhs_type_id != int_type_id {
                    error_id
                } else {
                    self.intern_ty(&types::Type::Integer, ctx.types)
                }
            }
            hir::BinOp::Greater | hir::BinOp::Geq | hir::BinOp::Less | hir::BinOp::Leq => {
                self.ensure_numeric(*lhs, lhs_type_id, ctx);
                self.ensure_numeric(*rhs, rhs_type_id, ctx);
                self.intern_ty(&types::Type::Boolean, ctx.types)
            }
            hir::BinOp::And | hir::BinOp::Or | hir::BinOp::Xor => {
                if self.match_type(lhs_type_id, &types::Type::Integer, ctx.types)
                    && self.match_type(rhs_type_id, &types::Type::Integer, ctx.types)
                {
                    self.intern_ty(&types::Type::Integer, ctx.types)
                } else if self.match_type(lhs_type_id, &types::Type::Boolean, ctx.types)
                    && self.match_type(rhs_type_id, &types::Type::Boolean, ctx.types)
                {
                    self.intern_ty(&types::Type::Boolean, ctx.types)
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
        ctx: &mut TypeCtx,
    ) -> ids::TypeId {
        // Start by analysing each of the items
        for item in items {
            let _ = match item {
                ItemId::Expr(expression_id) => {
                    self.type_expression(&expression_id, ctx);
                }
                ItemId::Symbol(symbol_id) => {
                    self.type_check_declaration(*symbol_id, ctx);
                }
            };
        }

        // If there is no final expression, then we will return the unit type
        let unit = self.intern_ty(&types::Type::Unit, ctx.types);
        final_expr
            .map(|e_id| self.type_expression(&e_id, ctx))
            .unwrap_or(unit)
    }

    /// When declaring a function, check that the body of the function returns the type expected in
    /// the function signature.
    pub fn type_check_function_declaration(
        &mut self,
        name_span: Span,
        body: ids::ExpressionId,
        expected_type: ids::TypeId,
        ctx: &mut TypeCtx,
    ) {
        let body_ty = self.type_expression(&body, ctx);
        if body_ty != expected_type && body_ty != self.err_id(ctx.types) {
            let body = self.get_expr_unchecked(body);
            self.errors.push(SemanticError::FnWrongReturnType {
                expected: types::type_id_stringify(ctx.types, expected_type),
                // none for now, but it really shuold not be none ... We need to improve spans
                return_type_span: None,
                fn_name_span: name_span,
                actual: types::type_id_stringify(ctx.types, body_ty),
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
        ctx: &mut TypeCtx,
    ) {
        let body_ty = self.type_expression(&body, ctx);
        if body_ty != expected_type && body_ty != self.err_id(ctx.types) {
            let body = self.get_expr_unchecked(body);
            self.errors.push(SemanticError::BindingWrongType {
                expected: types::type_id_stringify(ctx.types, expected_type),
                return_type_span: name_span,
                actual: types::type_id_stringify(ctx.types, body_ty),
                return_span: body.get_span(),
                body_span: body.get_span().unwrap(),
            })
        }
    }

    /// Make sure that a declarations types make sense semantically.
    pub fn type_check_declaration(&mut self, dec: ids::SymbolId, ctx: &mut TypeCtx) {
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
        let output_type = match take_inputs(*ty, arity, ctx.types, span_decl) {
            Ok((_, output_type)) => output_type,
            Err(err) => {
                self.errors.push(err);
                return;
            }
        };
        let body_actual_type = self.type_expression(&body_id, ctx);
        if output_type != body_actual_type {
            let expected_str = types::type_id_stringify(ctx.types, output_type);
            self.push_wrong_type(body_actual_type, &expected_str, span_decl, ctx.types);
        }
    }

    /// Type check all declarations in the module.
    pub fn type_check_module(&mut self, ctx: &mut TypeCtx) {
        let roots_len = self.module.roots.len();
        for idx in 0..roots_len {
            let symbol = self.module.roots[idx];
            self.type_check_declaration(symbol, ctx);
        }
    }
}
