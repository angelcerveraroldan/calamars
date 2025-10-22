use std::{any::Any, fmt::Debug, process::id, string};

use chumsky::{container::Seq, span::SimpleSpan, text::ascii::ident};

use crate::{
    sematic::{
        error::SemanticError,
        symbols::{DefKind, Symbol, SymbolArena, SymbolId, SymbolScope},
        types::{Type, TypeArena, TypeId},
    },
    syntax::{
        ast::{self, ClCompoundExpression, ClExpression, Ident},
        span::Span,
    },
};

pub mod error;
pub mod symbols;
pub mod types;

#[derive(Debug)]
pub enum ResolverOutput<T: Debug> {
    /// Everything went so perfect
    Ok(T),
    /// There was an error when inserting the symbol/type, but parsing can continue
    Recoverable(T),
    /// Nothing could be inserted to the resolver due to a fatal error
    Fatal,
}

impl<T: Debug> ResolverOutput<T> {
    pub fn is_ok(&self) -> bool {
        matches!(self, ResolverOutput::Ok(_))
    }

    pub fn is_recoverable(&self) -> bool {
        matches!(self, ResolverOutput::Recoverable(_))
    }

    pub fn is_err(&self) -> bool {
        matches!(self, ResolverOutput::Fatal)
    }

    pub fn inner(&self) -> &T {
        match self {
            Self::Ok(t) | Self::Recoverable(t) => t,
            _ => panic!("Can only call inner if you know the type is ok or recoverable"),
        }
    }
}

pub type ResolverSymbolOut = ResolverOutput<SymbolId>;
pub type ResolverTypeOut = ResolverOutput<TypeId>;

#[derive(Debug)]
pub struct Resolver {
    types: TypeArena,

    symbols: SymbolArena,
    scopes: Vec<SymbolScope>,

    dignostics_errors: Vec<SemanticError>,
}

impl Default for Resolver {
    fn default() -> Self {
        Self {
            types: TypeArena::default(),
            symbols: SymbolArena::default(),
            scopes: vec![SymbolScope::default()],

            dignostics_errors: vec![],
        }
    }
}

impl Resolver {
    fn push_scope(&mut self) {
        self.scopes.push(SymbolScope::default());
    }

    fn pop_scope(&mut self) {
        self.scopes.pop();
    }

    pub fn errors(&self) -> &Vec<SemanticError> {
        &self.dignostics_errors
    }

    /// Insert symbol into the current scope. The symbols id will be returned if correctly
    /// inserted. Nothing will be returned if the type already existed (error).
    pub fn declare(&mut self, sym: Symbol) -> ResolverSymbolOut {
        let curr_scope = self.scopes.last_mut().unwrap();
        let mut redeclared = false;
        if let Some(original) = curr_scope.map.get(&sym.name) {
            let original = self.symbols.get(original).unwrap();
            self.dignostics_errors.push(SemanticError::Redeclaration {
                original_span: original.name_span,
                redec_span: sym.name_span,
            });
            redeclared = true;
        }

        // Todo: Not sure if here we should insert (end of scope stack) or overwrite (replace the
        // current duplicate in the scope)
        let name = sym.name.clone();
        let id = self.symbols.insert(sym);
        curr_scope.map.insert(name, id);

        if redeclared {
            ResolverOutput::Recoverable(id)
        } else {
            ResolverOutput::Ok(id)
        }
    }

    /// Find the symbol id for a symbol with a given name. This will look though the current scope,
    /// and then though parent scopes and return the first match. If no matches are found, then an
    /// error will be returned (but not pushed to the resolver)
    pub fn resolve_ident(&self, name: &str, usage_loc: Span) -> Result<SymbolId, SemanticError> {
        for scope in self.scopes.iter().rev() {
            if let Some(symid) = scope.map.get(name) {
                return Ok(*symid);
            }
        }

        Err(SemanticError::IdentNotFound {
            name: name.into(),
            span: usage_loc,
        })
    }

    pub fn get_symbol_type(&self, symbol_id: &SymbolId) -> Result<TypeId, SemanticError> {
        match self.symbols.get(symbol_id) {
            Some(sy) => Ok(sy.ty),
            None => Err(SemanticError::SymbolIdNotFound { id: *symbol_id }),
        }
    }
}

// HANDLE AST DECLARATIONS
impl Resolver {
    /// Given some function declaration in the ast, add it to the symbol table in the current
    /// scope.
    fn push_ast_function(&mut self, node: &ast::ClFuncDec) -> ResolverSymbolOut {
        let ty = node.fntype().clone();
        let ty = match self.types.intern_cltype(&ty) {
            Ok(ty) => ty,
            Err(sem_err) => {
                self.dignostics_errors.push(sem_err);
                println!("There was an error getting the typeid");
                self.types.intern(types::Type::Error)
            }
        };
        let sym = Symbol {
            name: node.name().clone(),
            kind: symbols::DefKind::Func,
            arity: Some(node.airity()),
            ty: ty,
            name_span: node.name_span(),
            decl_span: node.span(),
        };
        self.declare(sym)
    }

    /// Given some binding in the ast, add it to the symbol table in the current scope.
    fn push_ast_binding(&mut self, node: &ast::ClBinding) -> ResolverSymbolOut {
        #[rustfmt::skip]
        let kind = if node.mutable { DefKind::Var } else { DefKind::Val };

        // If the binding had no type, then we will keep the type error attached to it and proceed
        // with the analysis, and push an error to the diagnostics.
        let ty = match &node.vtype {
            Some(ty) => self.types.intern_cltype(ty).unwrap_or_else(|e| {
                self.dignostics_errors.push(e);
                self.types.intern(types::Type::Error)
            }),
            None => {
                self.dignostics_errors.push(SemanticError::TypeMissingCtx {
                    for_identifier: node.vname.span(),
                });
                self.types.intern(types::Type::Error)
            }
        };

        let sym = Symbol {
            name: node.vname.ident().into(),
            kind,
            arity: None,
            ty: ty,
            name_span: node.name_span(),
            decl_span: node.span(),
        };
        self.declare(sym)
    }

    pub fn push_ast_declaration(&mut self, node: &ast::ClDeclaration) -> ResolverSymbolOut {
        match &node {
            ast::ClDeclaration::Binding(node_bind) => self.push_ast_binding(node_bind),
            ast::ClDeclaration::Function(node_fn) => self.push_ast_function(node_fn),
        }
    }
}

// HANDLE AST TYPE CHECKS
impl Resolver {
    fn ast_literal_type(&mut self, lit: &ast::ClLiteral) -> ResolverTypeOut {
        let ty = match lit.kind() {
            ast::ClLiteralKind::Integer(_) => Type::Integer,
            ast::ClLiteralKind::Real(_) => Type::Float,
            ast::ClLiteralKind::String(_) => Type::String,
            ast::ClLiteralKind::Boolean(_) => Type::Boolean,
            ast::ClLiteralKind::Char(_) => Type::Char,
            ast::ClLiteralKind::Array(_cl_literals) => {
                self.dignostics_errors.push(SemanticError::NotSupported {
                    msg: "Arrays not yet supported",
                    span: lit.span(),
                });
                return ResolverTypeOut::Recoverable(self.types.intern(Type::Error));
            }
        };

        ResolverTypeOut::Ok(self.types.intern(ty))
    }

    fn ast_identifier_type(&mut self, ident: &Ident) -> ResolverTypeOut {
        match self.resolve_ident(ident.ident(), ident.span()) {
            Ok(symbol_id) => {
                // We know that this will not fail as the id is the index
                let symbol = self.symbols.get(&symbol_id).unwrap();
                let ty = symbol.ty;
                ResolverTypeOut::Ok(ty)
            }
            // The symbol was not found, nothing we can do
            Err(e) => {
                self.dignostics_errors.push(e);
                ResolverTypeOut::Fatal
            }
        }
    }

    fn ast_function_call_return_type(&mut self, func_call: &ast::FuncCall) -> ResolverTypeOut {
        match self.resolve_ident(func_call.name(), func_call.span()) {
            Ok(func_id) => {
                let symbol = self.symbols.get(&func_id).unwrap();
                let out_ty = match self.types.get(symbol.ty).unwrap() {
                    Type::Function { output, .. } => output,
                    _ => unreachable!("Functions can only have function type"),
                };
                ResolverTypeOut::Ok(*out_ty)
            }
            Err(e) => {
                self.dignostics_errors.push(e);
                ResolverTypeOut::Fatal
            }
        }
    }

    /// Get the return type of an if expression.
    ///
    /// This will also make sure that both the if then and the else expressions have the same type.
    fn ast_if_stm_type(&mut self, if_stm: &ast::IfStm) -> ResolverTypeOut {
        let then_expr = if_stm.then_expr();
        let else_expr = if_stm.else_expr();

        let then_type = self.ast_expression_type(then_expr.as_ref());
        let else_type = self.ast_expression_type(else_expr.as_ref());

        if (then_type.is_err() || else_type.is_err()) {
            return ResolverTypeOut::Recoverable(self.types.intern(Type::Error));
        }

        let then_type = then_type.inner();
        let else_type = else_type.inner();
        if (then_type == else_type) {
            ResolverTypeOut::Ok(*then_type)
        } else {
            self.dignostics_errors
                .push(SemanticError::MismatchedIfBranches {
                    then_span: then_expr.span(),
                    else_span: else_expr.span(),
                });
            ResolverTypeOut::Recoverable(self.types.intern(Type::Error))
        }
    }

    fn ast_unary_type(&mut self, unary: &ast::ClUnaryOp) -> ResolverTypeOut {
        if *unary.operator() != ast::UnaryOperator::Neg {
            self.dignostics_errors.push(SemanticError::NotSupported {
                msg: "unary operator not supported yet",
                span: unary.span(),
            });
            return ResolverTypeOut::Recoverable(self.types.err());
        }
        let inner_type = self.ast_expression_type(unary.inner_exp());
        if inner_type.is_err() {
            return ResolverTypeOut::Fatal;
        }
        let inner_type = *inner_type.inner();

        let integer_type = self.types.int();
        let boolean_type = self.types.bool();

        // We can only negate booleans an integers
        if inner_type == integer_type || inner_type == boolean_type {
            ResolverTypeOut::Ok(inner_type)
        } else {
            self.dignostics_errors.push(SemanticError::WrongType {
                expected: self
                    .types
                    .many_types_as_str(vec![integer_type, boolean_type]),
                actual: self.types.as_string(inner_type),
                span: unary.span(),
            });
            ResolverTypeOut::Recoverable(self.types.err())
        }
    }

    fn ast_binary_op_type(&mut self, binary: &ast::ClBinaryOp) -> ResolverTypeOut {
        use ast::BinaryOperator::*; // match later is mess, import in this scope

        let (lhs, rhs) = (binary.lhs(), binary.rhs());

        let lhs_type = self.ast_expression_type(&lhs);
        let rhs_type = self.ast_expression_type(&rhs);
        if lhs_type.is_err() || rhs_type.is_err() {
            return ResolverTypeOut::Fatal;
        }

        let int_type = self.types.int();
        let float_type = self.types.float();
        let bool_type = self.types.bool();
        let string_type = self.types.string();

        let lhs_type = *lhs_type.inner();
        let rhs_type = *rhs_type.inner();

        match binary.operator() {
            // Basic mathematics ops are supported if both types are numerical
            Add | Sub | Times | Pow | Div => {
                // No errors
                if lhs_type == int_type && rhs_type == int_type {
                    return ResolverTypeOut::Ok(int_type);
                }
                if lhs_type == float_type && rhs_type == float_type {
                    return ResolverTypeOut::Ok(float_type);
                }

                // Handle case where the left and right are type mismatches, but numerical
                if lhs_type == int_type || lhs_type == float_type {
                    self.dignostics_errors.push(SemanticError::WrongType {
                        expected: self.types.as_string(lhs_type),
                        actual: self.types.as_string(rhs_type),
                        span: rhs.span(),
                    });
                    return ResolverTypeOut::Recoverable(lhs_type);
                }

                if rhs_type == int_type || rhs_type == float_type {
                    self.dignostics_errors.push(SemanticError::WrongType {
                        expected: self.types.as_string(rhs_type),
                        actual: self.types.as_string(lhs_type),
                        span: lhs.span(),
                    });
                    return ResolverTypeOut::Recoverable(rhs_type);
                }

                self.dignostics_errors.push(SemanticError::WrongType {
                    expected: self.types.many_types_as_str(vec![int_type, float_type]),
                    actual: self.types.as_string(lhs_type),
                    span: lhs.span(),
                });
                self.dignostics_errors.push(SemanticError::WrongType {
                    expected: self.types.many_types_as_str(vec![int_type, float_type]),
                    actual: self.types.as_string(rhs_type),
                    span: rhs.span(),
                });

                // Both wrong. We could assume int/float, but I dont think that is a good idea
                // for now.
                ResolverTypeOut::Recoverable(self.types.err())
            }
            Geq | Leq => {
                let lint = lhs_type == int_type;
                let rint = rhs_type == int_type;
                let lflt = lhs_type == float_type;
                let rflt = rhs_type == float_type;

                if (lint && rint) || (lflt && rflt) {
                    return ResolverTypeOut::Ok(bool_type);
                }

                if lint || lflt {
                    self.dignostics_errors.push(SemanticError::WrongType {
                        expected: self.types.as_string(lhs_type),
                        actual: self.types.as_string(rhs_type),
                        span: rhs.span(),
                    });
                    return ResolverTypeOut::Recoverable(bool_type);
                }

                if rint || rflt {
                    self.dignostics_errors.push(SemanticError::WrongType {
                        expected: self.types.as_string(rhs_type),
                        actual: self.types.as_string(lhs_type),
                        span: lhs.span(),
                    });
                    return ResolverTypeOut::Recoverable(bool_type);
                }

                self.dignostics_errors.push(SemanticError::WrongType {
                    expected: self.types.many_types_as_str(vec![int_type, float_type]),
                    actual: self.types.as_string(lhs_type),
                    span: lhs.span(),
                });
                self.dignostics_errors.push(SemanticError::WrongType {
                    expected: self.types.many_types_as_str(vec![int_type, float_type]),
                    actual: self.types.as_string(rhs_type),
                    span: rhs.span(),
                });

                ResolverTypeOut::Recoverable(self.types.err())
            }
            EqEq | NotEqual => {
                if lhs_type == rhs_type {
                    ResolverTypeOut::Ok(bool_type)
                } else {
                    self.dignostics_errors.push(SemanticError::WrongType {
                        expected: self.types.as_string(lhs_type),
                        actual: self.types.as_string(rhs_type),
                        span: lhs.span(),
                    });
                    ResolverTypeOut::Recoverable(bool_type)
                }
            }
            Or | Xor | And => {
                // Both integers or both booleans ONLY
                if (lhs_type == int_type && rhs_type == int_type)
                    || (lhs_type == bool_type && rhs_type == bool_type)
                {
                    return ResolverTypeOut::Ok(lhs_type);
                }

                // The left makes sense, but the right does not match / doesnt make sense.
                if lhs_type == int_type || lhs_type == bool_type {
                    self.dignostics_errors.push(SemanticError::WrongType {
                        expected: self.types.as_string(lhs_type),
                        actual: self.types.as_string(rhs_type),
                        span: rhs.span(),
                    });
                    return ResolverTypeOut::Recoverable(lhs_type);
                }

                // The left didnt make sense, and the right does. Assume rhs type was intended
                if rhs_type == int_type || rhs_type == bool_type {
                    self.dignostics_errors.push(SemanticError::WrongType {
                        expected: self.types.as_string(rhs_type),
                        actual: self.types.as_string(lhs_type),
                        span: lhs.span(),
                    });
                    return ResolverTypeOut::Recoverable(rhs_type);
                }

                // Neither of the types made sense.

                self.dignostics_errors.push(SemanticError::WrongType {
                    expected: self.types.many_types_as_str(vec![int_type, bool_type]),
                    actual: self.types.as_string(lhs_type),
                    span: lhs.span(),
                });
                self.dignostics_errors.push(SemanticError::WrongType {
                    expected: self.types.many_types_as_str(vec![int_type, bool_type]),
                    actual: self.types.as_string(rhs_type),
                    span: rhs.span(),
                });

                ResolverTypeOut::Recoverable(self.types.err())
            }
            Concat => {
                if lhs_type == string_type && rhs_type == string_type {
                    return ResolverTypeOut::Ok(string_type);
                }

                if lhs_type != string_type {
                    self.dignostics_errors.push(SemanticError::WrongType {
                        expected: self.types.as_string(string_type),
                        actual: self.types.as_string(lhs_type),
                        span: lhs.span(),
                    })
                }

                if rhs_type != string_type {
                    self.dignostics_errors.push(SemanticError::WrongType {
                        expected: self.types.as_string(string_type),
                        actual: self.types.as_string(rhs_type),
                        span: rhs.span(),
                    })
                }

                // Concat always returns a string, so we know the output
                ResolverTypeOut::Recoverable(string_type)
            }
        }
    }

    /// Given some expression, return the TypeId of the expressions return type
    pub fn ast_expression_type(&mut self, node: &ast::ClExpression) -> ResolverTypeOut {
        match node {
            ast::ClExpression::Literal(cl_literal) => self.ast_literal_type(cl_literal),
            ast::ClExpression::Identifier(ident) => self.ast_identifier_type(ident),
            ast::ClExpression::UnaryOp(cl_unary_op) => self.ast_unary_type(cl_unary_op),
            ast::ClExpression::BinaryOp(cl_binary_op) => self.ast_binary_op_type(cl_binary_op),
            ast::ClExpression::IfStm(if_stm) => self.ast_if_stm_type(if_stm),
            ast::ClExpression::FunctionCall(func_call) => {
                self.ast_function_call_return_type(func_call)
            }
            ast::ClExpression::Block(ClCompoundExpression { final_expr, .. }) => match final_expr {
                Some(exp) => self.ast_expression_type(exp),
                None => ResolverTypeOut::Ok(self.types.intern(Type::Unit)),
            },
        }
    }

    fn _check_type_eq_and_log_error(&mut self, exp: TypeId, acc: TypeId, span: Span) {
        // If they are the same, the check passes
        //
        // If either of them had an error type, then we will ignore this check, as this means that
        // the expression was malformatted, and that error is already logged.
        if exp == acc || exp == self.types.err() || acc == self.types.err() {
            return;
        }

        self.dignostics_errors.push(SemanticError::WrongType {
            expected: self.types.as_string(exp),
            actual: self.types.as_string(acc),
            span,
        });
    }

    fn type_check_binding(&mut self, binding: &ast::ClBinding) {
        let acc_ty = self.ast_expression_type(&binding.assigned);
        if acc_ty.is_err() {
            return;
        }
        let acc_ty = *acc_ty.inner();
        let sym_id = match self.resolve_ident(binding.vname.ident(), binding.name_span()) {
            Ok(sym_id) => sym_id,
            Err(e) => {
                self.dignostics_errors.push(e);
                return;
            }
        };

        let sym_ty = self
            .get_symbol_type(&sym_id)
            .expect("SymbolId not in range...");

        self._check_type_eq_and_log_error(sym_ty, acc_ty, binding.assigned.span());
    }

    fn type_check_function(&mut self, binding: &ast::ClFuncDec) {
        let acc_ty = self.ast_expression_type(&binding.body());
        if acc_ty.is_err() {
            return;
        }
        let acc_ty = *acc_ty.inner();
        let sym_id = match self.resolve_ident(binding.name(), binding.name_span()) {
            Ok(sym_id) => sym_id,
            Err(e) => {
                self.dignostics_errors.push(e);
                return;
            }
        };

        let sym_ty = self
            .get_symbol_type(&sym_id)
            .expect("SymbolId not in range...");

        let output_ty = self
            .types
            .get(sym_ty)
            .map(|ty| match ty {
                Type::Function { output, .. } => output,
                _ => &sym_ty,
            })
            .unwrap();

        self._check_type_eq_and_log_error(*output_ty, acc_ty, binding.body().span());
    }

    fn type_check_declaration(&mut self, node: &ast::ClDeclaration) {
        match node {
            ast::ClDeclaration::Binding(node) => self.type_check_binding(node),
            ast::ClDeclaration::Function(node) => self.type_check_function(node),
        }
    }

    fn type_check_if_condition(&mut self, if_stm: &ast::IfStm) {
        let cond = if_stm.pred();
        let pred_type = self.ast_expression_type(cond);
        if pred_type.is_err() {
            return;
        }
        let pred_type = pred_type.inner();
        self._check_type_eq_and_log_error(self.types.bool(), *pred_type, if_stm.pred_span());
    }

    fn type_check_func_call_input(&mut self, func_call: &ast::FuncCall) {
        let f = self.resolve_ident(func_call.name(), func_call.span());
        let id = match f
            .map(|symbol_id| self.get_symbol_type(&symbol_id))
            .flatten()
        {
            Ok(type_id) => type_id,
            Err(e) => {
                // We didnt find the function we are calling, so error
                self.dignostics_errors.push(e);
                return;
            }
        };

        let (inpt, out) = match self.types.unchecked_get(id) {
            // TODO: Dont clone so much
            Type::Function { input, output } => (input.clone(), output.clone()),
            // FIXME: Dont panic!
            _ => unreachable!("Function should have function type"),
        };

        if func_call.params().len() != inpt.len() {
            self.dignostics_errors.push(SemanticError::ArityError {
                expected: inpt.len(),
                actual: func_call.params().len(),
                span: func_call.span(),
            });
            return;
        }

        // For each wrong type we will throw one error
        for (expected, expression) in inpt.iter().zip(func_call.params()) {
            let acc_type = self.ast_expression_type(&expression);
            if acc_type.is_err() {
                continue;
            }

            let acc_type = *acc_type.inner();
            if acc_type != *expected {
                self.dignostics_errors.push(SemanticError::WrongType {
                    expected: self.types.as_string(*expected),
                    actual: self.types.as_string(acc_type),
                    span: expression.span(),
                })
            }
        }
    }
}

#[cfg(test)]
mod test_helpers_resolver {
    use chumsky::{Parser, span::SimpleSpan};

    use crate::syntax::ast::{
        self, BinaryOperator, ClBinaryOp, ClBinding, ClExpression, ClLiteral, ClLiteralKind,
        ClType, Ident,
    };

    pub fn fake_span() -> SimpleSpan {
        SimpleSpan {
            start: 0,
            end: 0,
            context: (),
        }
    }

    pub fn cltype_int() -> ClType {
        ClType::Path {
            segments: vec![Ident::new("Int".to_string(), fake_span())],
            span: fake_span(),
        }
    }

    pub fn cltype_bool() -> ClType {
        ClType::Path {
            segments: vec![Ident::new("Bool".to_string(), fake_span())],
            span: fake_span(),
        }
    }

    pub fn cltype_float() -> ClType {
        ClType::Path {
            segments: vec![Ident::new("Float".to_string(), fake_span())],
            span: fake_span(),
        }
    }

    pub fn cltype_string() -> ClType {
        ClType::Path {
            segments: vec![Ident::new("String".to_string(), fake_span())],
            span: fake_span(),
        }
    }

    pub fn expr_int(n: i64) -> ClExpression {
        ClExpression::Literal(ClLiteral::new(ClLiteralKind::Integer(n), fake_span()))
    }

    pub fn expr_real(f: f64) -> ClExpression {
        ClExpression::Literal(ClLiteral::new(ClLiteralKind::Real(f), fake_span()))
    }

    pub fn expr_bool(b: bool) -> ClExpression {
        ClExpression::Literal(ClLiteral::new(ClLiteralKind::Boolean(b), fake_span()))
    }

    pub fn expr_string(s: &str) -> ClExpression {
        ClExpression::Literal(ClLiteral::new(
            ClLiteralKind::String(s.to_string()),
            fake_span(),
        ))
    }

    pub fn bin(lhs: ClExpression, op: BinaryOperator, rhs: ClExpression) -> ClExpression {
        ClExpression::BinaryOp(ClBinaryOp::new(op, lhs.into(), rhs.into(), fake_span()))
    }

    pub fn integer_literal() -> ClLiteral {
        ClLiteral::new(ast::ClLiteralKind::Integer(10), fake_span())
    }

    pub fn make_bad_ast_func(name: &str) -> ast::ClFuncDec {
        let fake = fake_span();

        ast::ClFuncDec::new(
            Ident::new(name.to_string(), fake),
            vec![(Ident::new("x".to_string(), fake), cltype_int().into())],
            cltype_string().into(),                        // Returns string
            ast::ClExpression::Literal(integer_literal()), // Body actually returns integer
            fake_span(),
            None,
        )
    }

    pub fn make_ast_func(name: &str) -> ast::ClFuncDec {
        let fake = fake_span();

        ast::ClFuncDec::new(
            Ident::new(name.to_string(), fake),
            vec![(Ident::new("x".to_string(), fake), cltype_int().into())],
            cltype_int().into(),
            ast::ClExpression::Literal(integer_literal()),
            fake_span(),
            None,
        )
    }

    pub fn make_bad_var(name: &str) -> ast::ClBinding {
        ClBinding::new(
            Ident::new(name.to_string(), fake_span()),
            cltype_string().into(),
            Box::new(ast::ClExpression::Literal(integer_literal())),
            false,
            fake_span(),
        )
    }

    pub fn make_var(name: &str) -> ast::ClBinding {
        ClBinding::new(
            Ident::new(name.to_string(), fake_span()),
            cltype_int().into(),
            Box::new(ast::ClExpression::Literal(integer_literal())),
            false,
            fake_span(),
        )
    }
}

#[cfg(test)]
mod test_insert_node_to_resolver {
    use super::test_helpers_resolver::*;
    use crate::sematic::Resolver;

    #[test]
    fn happy_resolver() {
        let mut resolver = Resolver::default();
        let f = make_ast_func("f");
        let y = make_var("y");
        let g = make_ast_func("g");

        let a = resolver.push_ast_function(&f);
        let b = resolver.push_ast_binding(&y);
        let c = resolver.push_ast_function(&g);

        assert!(a.is_ok() & b.is_ok() & c.is_ok());
        assert_eq!(resolver.symbols.len(), 3);
    }

    #[test]
    fn phase_a_duplicate_var_in_same_scope_is_error() {
        let mut resolver = Resolver::default();
        let a1 = make_var("a");
        let a2 = make_var("a");

        assert!(resolver.push_ast_binding(&a1).is_ok());
        assert!(
            resolver.push_ast_binding(&a2).is_recoverable(),
            "redecalring a is a recoverable error"
        );
        assert_eq!(resolver.symbols.len(), 2);
    }

    #[test]
    fn phase_a_duplicate_func_in_same_scope_is_error() {
        let mut resolver = Resolver::default();
        let f1 = make_ast_func("f");
        let f2 = make_ast_func("f");

        assert!(resolver.push_ast_function(&f1).is_ok());
        assert!(
            resolver.push_ast_function(&f2).is_recoverable(),
            "duplicate function is a recoverable error"
        );
        assert_eq!(resolver.symbols.len(), 2);
    }

    #[test]
    fn phase_a_function_and_var_same_name() {
        let mut resolver = Resolver::default();
        let f = make_ast_func("f");
        let v = make_var("f");
        assert!(resolver.push_ast_function(&f).is_ok());
        assert!(
            resolver.push_ast_binding(&v).is_recoverable(),
            "function and var having the same name is a recovarble error"
        );
        assert_eq!(resolver.symbols.len(), 2);
    }
}

#[cfg(test)]
mod test_get_expr_type {
    use super::test_helpers_resolver::*;
    use crate::{
        sematic::{Resolver, types},
        syntax::ast::{
            self, BinaryOperator, ClCompoundExpression, ClExpression, ClLiteral, ClLiteralKind,
            ClType, FuncCall, Ident,
        },
    };

    #[test]
    fn expr_literal_int_has_int_type() {
        let mut resolver = Resolver::default();

        let expr = ClExpression::Literal(integer_literal());
        let out = resolver.ast_expression_type(&expr);

        assert!(out.is_ok(), "typing an int literal should succeed");

        let got = *out.inner();
        let expect = resolver.types.intern_cltype(&cltype_int()).unwrap();
        assert_eq!(got, expect, "int literal should type to Int");
    }

    #[test]
    // Get the return type of a function even if the input types are wrong.
    fn get_function_return_type() {
        let mut resolver = Resolver::default();
        let f = make_ast_func("f");
        resolver.push_ast_function(&f);

        let fcall = FuncCall::new(
            Ident::new("f".to_string(), fake_span()),
            vec![],
            fake_span(),
        );
        let out = resolver.ast_function_call_return_type(&fcall);
        assert!(out.is_ok());
        let exp = resolver.types.intern_cltype(&cltype_int()).unwrap();
        let acc = *out.inner();
        assert_eq!(exp, acc);
    }

    #[test]
    fn expr_identifier_uses_declared_binding_type() {
        let mut resolver = Resolver::default();

        // val x: Int = 10
        let x = make_var("x");
        assert!(resolver.push_ast_binding(&x).is_ok());

        // use x
        let expr = ClExpression::Identifier(Ident::new("x".into(), fake_span()));
        let out = resolver.ast_expression_type(&expr);

        assert!(out.is_ok(), "typing an existing identifier should succeed");

        let got = *out.inner();
        let expect = resolver.types.intern_cltype(&cltype_int()).unwrap();
        assert_eq!(got, expect, "identifier x should have type Int");
    }

    #[test]
    fn expr_block_without_final_expr_is_unit() {
        let mut resolver = Resolver::default();

        let block = ClExpression::Block(ClCompoundExpression::new(vec![], None, fake_span()));

        let out = resolver.ast_expression_type(&block);
        assert!(out.is_ok(), "empty block should type check");
        let got = *out.inner();
        let expect = resolver
            .types
            .intern_cltype(&ClType::Path {
                segments: vec![Ident::new("()".into(), fake_span())],
                span: fake_span(),
            })
            .unwrap();
        assert_eq!(got, expect, "empty block should have type Unit");
    }

    #[test]
    fn if_with_matching_branch_types_yields_that_type() {
        let mut resolver = Resolver::default();

        let cond = ClExpression::Literal(ClLiteral::new(ClLiteralKind::Boolean(true), fake_span()));
        let then_e = ClExpression::Literal(integer_literal());
        let else_e = ClExpression::Literal(integer_literal());
        let if_e = ClExpression::IfStm(ast::IfStm::new(
            cond.into(),
            then_e.into(),
            else_e.into(),
            fake_span(),
        ));

        let out = resolver.ast_expression_type(&if_e);
        assert!(out.is_ok(), "if with equal branch types should be ok");

        let got = *out.inner();
        let expect = resolver.types.intern_cltype(&cltype_int()).unwrap();
        assert_eq!(got, expect);
    }

    #[test]
    fn if_with_mismatched_branch_types_is_recoverable_error() {
        let mut resolver = Resolver::default();
        let before = resolver.dignostics_errors.len();

        let cond = ClExpression::Literal(ClLiteral::new(ClLiteralKind::Boolean(true), fake_span()));
        let then_e = ClExpression::Literal(integer_literal());
        let else_e =
            ClExpression::Literal(ClLiteral::new(ClLiteralKind::Boolean(false), fake_span()));
        let if_e = ClExpression::IfStm(ast::IfStm::new(
            cond.into(),
            then_e.into(),
            else_e.into(),
            fake_span(),
        ));

        let out = resolver.ast_expression_type(&if_e);
        assert!(
            out.is_recoverable(),
            "Different branches have diff return types, so there should be an error type, but it is recoverable."
        );

        let got = *out.inner();
        let expect_err = resolver.types.intern(types::Type::Error);
        assert_eq!(got, expect_err);
        assert!(resolver.dignostics_errors.len() > before,);
    }

    #[test]
    fn add_int_int_ok() {
        let mut r = Resolver::default();
        let out = r.ast_expression_type(&bin(expr_int(1), BinaryOperator::Add, expr_int(2)));
        assert!(out.is_ok());
        let got = *out.inner();
        let expect = r.types.intern_cltype(&cltype_int()).unwrap();
        assert_eq!(got, expect);
    }

    #[test]
    fn add_float_float_ok() {
        let mut r = Resolver::default();
        let out = r.ast_expression_type(&bin(expr_real(1.0), BinaryOperator::Add, expr_real(2.0)));
        assert!(out.is_ok());
        let got = *out.inner();
        let expect = r.types.intern_cltype(&cltype_float()).unwrap();
        assert_eq!(got, expect);
    }

    #[test]
    fn add_int_bool_recover_left_type() {
        let mut r = Resolver::default();
        let out = r.ast_expression_type(&bin(expr_int(1), BinaryOperator::Add, expr_bool(true)));
        assert!(
            out.is_recoverable(),
            "int + bool should be a recoverable error"
        );
        let got = *out.inner();
        let expect = r.types.intern_cltype(&cltype_int()).unwrap();
        assert_eq!(got, expect, "recover with lhs numeric type");
    }

    #[test]
    fn leq_int_bool_recover_bool() {
        let mut r = Resolver::default();
        let out = r.ast_expression_type(&bin(expr_int(1), BinaryOperator::Leq, expr_bool(true)));
        assert!(
            out.is_recoverable(),
            "int <= bool should be recoverable with Bool result"
        );
        let got = *out.inner();
        let expect = r.types.intern_cltype(&cltype_bool()).unwrap();
        assert_eq!(got, expect);
    }

    #[test]
    fn eqeq_int_int_ok_bool() {
        let mut r = Resolver::default();
        let out = r.ast_expression_type(&bin(expr_int(1), BinaryOperator::EqEq, expr_int(1)));
        assert!(out.is_ok());
        let got = *out.inner();
        let expect = r.types.intern_cltype(&cltype_bool()).unwrap();
        assert_eq!(got, expect);
    }

    #[test]
    fn eqeq_int_bool_recover_bool() {
        let mut r = Resolver::default();
        let out = r.ast_expression_type(&bin(expr_int(1), BinaryOperator::EqEq, expr_bool(true)));
        assert!(
            out.is_recoverable(),
            "int == bool should be recoverable with Bool result"
        );
        let got = *out.inner();
        let expect = r.types.intern_cltype(&cltype_bool()).unwrap();
        assert_eq!(got, expect);
    }

    #[test]
    fn or_ints_ok_int() {
        let mut r = Resolver::default();
        let out = r.ast_expression_type(&bin(expr_int(1), BinaryOperator::Or, expr_int(2)));
        assert!(out.is_ok());
        let got = *out.inner();
        let expect = r.types.intern_cltype(&cltype_int()).unwrap();
        assert_eq!(got, expect);
    }

    #[test]
    fn and_bools_ok_bool() {
        let mut r = Resolver::default();
        let out =
            r.ast_expression_type(&bin(expr_bool(true), BinaryOperator::And, expr_bool(false)));
        assert!(out.is_ok());
        let got = *out.inner();
        let expect = r.types.intern_cltype(&cltype_bool()).unwrap();
        assert_eq!(got, expect);
    }

    #[test]
    fn or_int_bool_mismatch_recover_operand_type() {
        let mut r = Resolver::default();
        let out = r.ast_expression_type(&bin(expr_int(1), BinaryOperator::Or, expr_bool(true)));
        assert!(out.is_recoverable(), "int or bool should be recoverable");
        let got = *out.inner();
        let expect = r.types.intern_cltype(&cltype_int()).unwrap(); // recover with lhs type per your policy
        assert_eq!(got, expect);
    }
}

#[cfg(test)]
mod test_type_matching {
    use crate::{
        sematic::{Resolver, test_helpers_resolver::*},
        syntax::ast::{
            self, BinaryOperator, ClExpression, ClLiteral, ClLiteralKind, FuncCall, Ident,
        },
    };

    #[test]
    fn test_declaration_type_matches_no_error() {
        let mut resolver = Resolver::default();
        let f = make_ast_func("f"); // Type Int -> Int
        resolver.push_ast_function(&f);
        resolver.type_check_function(&f);
        assert!(resolver.dignostics_errors.is_empty()); // No errors emitted
    }

    #[test]
    fn test_declaration_type_missmatch_err() {
        let mut resolver = Resolver::default();
        let f = make_bad_ast_func("f"); // Type Int -> String (acc returns int)
        resolver.push_ast_function(&f);
        resolver.type_check_function(&f);
        assert!(resolver.dignostics_errors.len() == 1); // Just one error
    }

    #[test]
    fn test_function_call_wrong_arity() {
        let mut resolver = Resolver::default();
        let f = make_ast_func("f");
        resolver.push_ast_function(&f);
        let func_call =
            &ast::FuncCall::new(Ident::new("f".into(), fake_span()), vec![], fake_span());
        resolver.type_check_func_call_input(func_call);
        assert_eq!(resolver.dignostics_errors.len(), 1);
    }

    #[test]
    fn test_function_call_right_arity() {
        let mut resolver = Resolver::default();
        let f = make_ast_func("f");
        resolver.push_ast_function(&f);
        let func_call = &ast::FuncCall::new(
            Ident::new("f".into(), fake_span()),
            vec![expr_int(1)],
            fake_span(),
        );
        resolver.type_check_func_call_input(func_call);
        assert_eq!(resolver.dignostics_errors.len(), 0);
    }

    #[test]
    fn test_function_call_right_arity_wrong_type() {
        let mut resolver = Resolver::default();
        let f = make_ast_func("f");
        resolver.push_ast_function(&f);
        let func_call = &ast::FuncCall::new(
            Ident::new("f".into(), fake_span()),
            vec![expr_string("aaa")],
            fake_span(),
        );
        resolver.type_check_func_call_input(func_call);
        assert_eq!(resolver.dignostics_errors.len(), 1);
    }
    #[test]
    fn test_binding_type_matches_no_error() {
        let mut resolver = Resolver::default();
        let v = make_var("v");
        resolver.push_ast_binding(&v);
        resolver.type_check_binding(&v);
        assert!(resolver.dignostics_errors.is_empty()); // No errors emitted
    }

    #[test]
    fn test_binding_type_missmatch_err() {
        let mut resolver = Resolver::default();
        let v = make_bad_var("v");
        resolver.push_ast_binding(&v);
        resolver.type_check_binding(&v);
        assert!(resolver.dignostics_errors.len() == 1);
    }

    fn generate_if(cond: ast::ClExpression) -> ast::IfStm {
        let then_e = ClExpression::Literal(integer_literal());
        let else_e = ClExpression::Literal(integer_literal());
        ast::IfStm::new(cond.into(), then_e.into(), else_e.into(), fake_span())
    }

    #[test]
    fn test_if_with_bool() {
        let mut resolver = Resolver::default();
        let ifstm = generate_if(ClExpression::Literal(ClLiteral::new(
            ClLiteralKind::Boolean(true),
            fake_span(),
        )));
        resolver.type_check_if_condition(&ifstm);
        assert!(resolver.dignostics_errors.is_empty());
    }

    #[test]
    fn test_if_with_eq() {
        let mut resolver = Resolver::default();
        let cond = bin(expr_int(1), BinaryOperator::EqEq, expr_int(2));
        let ifstm = generate_if(cond);
        resolver.type_check_if_condition(&ifstm);
        assert_eq!(resolver.dignostics_errors.len(), 0);
    }

    #[test]
    fn test_if_with_add() {
        let mut resolver = Resolver::default();
        let cond = bin(expr_int(1), BinaryOperator::Add, expr_int(2));
        let ifstm = generate_if(cond);
        resolver.type_check_if_condition(&ifstm);
        assert_eq!(resolver.dignostics_errors.len(), 1);
    }

    #[test]
    fn test_if_with_integer() {
        let mut resolver = Resolver::default();
        let ifstm = generate_if(expr_int(2));
        resolver.type_check_if_condition(&ifstm);
        assert_eq!(resolver.dignostics_errors.len(), 1);
    }
}
