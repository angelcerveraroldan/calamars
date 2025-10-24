use std::{any::Any, fmt::Debug, process::id, string};

use chumsky::{container::Seq, span::SimpleSpan, text::ascii::ident};

use crate::{
    parser::declaration,
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

    pub fn is_fatal(&self) -> bool {
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

        // We will add the redeclaration to the end of the same scope.
        //
        // We will keep the old symbol in the table too, it should not be accessible by name, as we
        // are overwitting it, but if we have the symbolid, and want to check something about it,
        // we still can do so.
        let name = sym.name.clone();
        let id = self.symbols.insert(sym);
        curr_scope.map.insert(name, id);

        redeclared
            .then(|| ResolverSymbolOut::Recoverable(id))
            .unwrap_or_else(|| ResolverSymbolOut::Ok(id))
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

// Verifiers
impl Resolver {
    pub fn verify_module(&mut self, module: &ast::Module) {
        #[rustfmt::skip]
        let symbol_ids = module.items.iter()
            .map(|declaration| self.push_ast_declaration(declaration))
            .collect::<Vec<_>>();

        for (symout, dec) in symbol_ids.iter().zip(&module.items) {
            if symout.is_fatal() {
                continue;
            }
            self.verify_declaration(&dec, *symout.inner());
        }
    }

    fn verify_compound_expression(&mut self, expr: &ast::ClCompoundExpression) -> ResolverTypeOut {
        self.push_scope();

        // Insert all declarations in the table
        for i in &expr.items {
            match i {
                ast::ClItem::Declaration(cl_declaration) => {
                    let symbol_id = self.push_ast_declaration(cl_declaration);
                    if symbol_id.is_fatal() {
                        return ResolverTypeOut::Fatal;
                    }
                    self.verify_declaration(cl_declaration, *symbol_id.inner());
                }
                ast::ClItem::Expression(cl_expression) => {
                    self.verify_expression_validity_and_return_typeid(cl_expression);
                }
                ast::ClItem::Import(cl_import) => {}
            }
        }

        let ret = match &expr.final_expr {
            Some(e) => self.verify_expression_validity_and_return_typeid(e),
            None => ResolverTypeOut::Ok(self.types.unit()),
        };

        self.pop_scope();
        ret
    }

    /// Verify that a declaration is semantically sound
    ///
    /// 1. Check that the declared return type and the actual return type match
    /// 2. Check that any sub-expressions are valid
    fn verify_declaration(&mut self, declaration: &ast::ClDeclaration, symbol_id: SymbolId) {
        match declaration {
            ast::ClDeclaration::Binding(cl_binding) => {
                self.type_check_binding(cl_binding, symbol_id)
            }
            ast::ClDeclaration::Function(cl_func_dec) => {
                self.type_check_function(cl_func_dec, symbol_id)
            }
        }
    }

    /// Given some expression, we will verify that its body is semantically sound
    ///
    /// This will also return its return typeid
    fn verify_expression_validity_and_return_typeid(
        &mut self,
        expr: &ast::ClExpression,
    ) -> ResolverTypeOut {
        // Here we will check that all sub-expressions make sense
        match expr {
            ClExpression::Identifier(cl_ident) => self.ast_identifier_type(cl_ident),
            ClExpression::Literal(lit) => self.ast_literal_type(lit),
            ClExpression::UnaryOp(cl_unary_op) => self.ast_unary_type(cl_unary_op),
            ClExpression::BinaryOp(cl_binary_op) => self.ast_binary_op_type(cl_binary_op),
            ClExpression::IfStm(if_stm) => {
                self.type_check_if_condition(if_stm);
                self.ast_if_stm_type(if_stm)
            }
            ClExpression::FunctionCall(func_call) => {
                self.type_check_func_call_and_get_return_type(func_call)
            }
            ClExpression::Block(cl_compound_expression) => {
                self.verify_compound_expression(cl_compound_expression)
            }
        }
    }
}

// HANDLE AST DECLARATIONS
impl Resolver {
    /// Given some function declaration in the ast, add it to the symbol table in the current
    /// scope.
    fn push_ast_function(&mut self, node: &ast::ClFuncDec) -> ResolverSymbolOut {
        let ty = match self.types.intern_cltype(&node.fntype()) {
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

    /// Get the return type of an if expression.
    ///
    /// This will also make sure that both the if then and the else expressions have the same type.
    fn ast_if_stm_type(&mut self, if_stm: &ast::IfStm) -> ResolverTypeOut {
        let then_expr = if_stm.then_expr();
        let else_expr = if_stm.else_expr();

        let then_type = self.verify_expression_validity_and_return_typeid(then_expr.as_ref());
        let else_type = self.verify_expression_validity_and_return_typeid(else_expr.as_ref());

        if (then_type.is_fatal() || else_type.is_fatal()) {
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
        let inner_type = self.verify_expression_validity_and_return_typeid(unary.inner_exp());
        if inner_type.is_fatal() {
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

        let lhs_type = self.verify_expression_validity_and_return_typeid(&lhs);
        let rhs_type = self.verify_expression_validity_and_return_typeid(&rhs);
        if lhs_type.is_fatal() || rhs_type.is_fatal() {
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

    fn type_check_binding(&mut self, binding: &ast::ClBinding, symbol_id: SymbolId) {
        let acc_ty = self.verify_expression_validity_and_return_typeid(&binding.assigned);
        if acc_ty.is_fatal() {
            return;
        }
        let acc_ty = *acc_ty.inner();
        let sym_ty = self
            .get_symbol_type(&symbol_id)
            .expect("SymbolId not in range...");

        self._check_type_eq_and_log_error(sym_ty, acc_ty, binding.assigned.span());
    }

    fn type_check_function(&mut self, binding: &ast::ClFuncDec, symbol_id: SymbolId) {
        let symbol_type = self.get_symbol_type(&symbol_id).unwrap();
        let input_typeids = match self.types.fn_input_typeids(symbol_type) {
            Some(v) => v.clone(),
            None => {
                self.dignostics_errors.push(SemanticError::InternalError {
                    msg: "This should have a Function type, but didnt! Please report this error.",
                    span: binding.name_span(),
                });
                return;
            }
        };
        let input_idents = binding.input_idents();

        self.push_scope();
        // Add input parameters to the scope
        for (typeid, ident) in input_typeids.iter().zip(input_idents) {
            self.declare(Symbol {
                name: ident.ident().into(),
                kind: DefKind::Val,
                arity: None,
                ty: *typeid,
                name_span: ident.span(),
                decl_span: ident.span(),
            });
        }
        let acc_ty = self.verify_expression_validity_and_return_typeid(&binding.body());
        self.pop_scope();

        if acc_ty.is_fatal() {
            return;
        }
        let acc_ty = *acc_ty.inner();
        let sym_ty = self
            .get_symbol_type(&symbol_id)
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

    fn type_check_if_condition(&mut self, if_stm: &ast::IfStm) {
        let cond = if_stm.pred();
        let pred_type = self.verify_expression_validity_and_return_typeid(cond);
        if pred_type.is_fatal() {
            return;
        }
        let pred_type = pred_type.inner();
        self._check_type_eq_and_log_error(self.types.bool(), *pred_type, if_stm.pred_span());
    }

    fn type_check_func_call_and_get_return_type(
        &mut self,
        func_call: &ast::FuncCall,
    ) -> ResolverTypeOut {
        let f = self.resolve_ident(func_call.name(), func_call.span());
        let id = match f
            .map(|symbol_id| self.get_symbol_type(&symbol_id))
            .flatten()
        {
            Ok(type_id) => type_id,
            Err(e) => {
                // We didnt find the function we are calling, so error
                self.dignostics_errors.push(e);
                return ResolverTypeOut::Recoverable(self.types.err());
            }
        };

        let (inpt, out) = match self.types.unchecked_get(id) {
            // TODO: Dont clone so much
            Type::Function { input, output } => (input.clone(), output.clone()),
            _ => {
                self.dignostics_errors.push(SemanticError::NonCallable {
                    msg: "Cannot call a non-function",
                    name: func_call.name().into(),
                    span: func_call.name_span(),
                });
                return ResolverTypeOut::Recoverable(self.types.err());
            }
        };

        if func_call.params().len() != inpt.len() {
            self.dignostics_errors.push(SemanticError::ArityError {
                expected: inpt.len(),
                actual: func_call.params().len(),
                span: func_call.span(),
            });
            return ResolverTypeOut::Recoverable(out);
        }

        // For each wrong type we will throw one error
        for (expected, expression) in inpt.iter().zip(func_call.params()) {
            let acc_type = self.verify_expression_validity_and_return_typeid(&expression);
            if acc_type.is_fatal() {
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

        ResolverOutput::Ok(out)
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

    pub fn make_bad_func_undefined_literal() -> ast::ClFuncDec {
        let fake = fake_span();
        ast::ClFuncDec::new(
            Ident::new("id".to_string(), fake),
            vec![(Ident::new("x".to_string(), fake), cltype_int().into())],
            cltype_int().into(),
            // Not the correct return literal
            ast::ClExpression::Identifier((Ident::new("y".to_string(), fake))),
            fake_span(),
            None,
        )
    }

    pub fn make_ast_id_func() -> ast::ClFuncDec {
        let fake = fake_span();
        ast::ClFuncDec::new(
            Ident::new("id".to_string(), fake),
            vec![(Ident::new("x".to_string(), fake), cltype_int().into())],
            cltype_int().into(),
            ast::ClExpression::Identifier((Ident::new("x".to_string(), fake))),
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
        sematic::{Resolver, error::SemanticError, types},
        syntax::ast::{
            self, BinaryOperator, ClCompoundExpression, ClExpression, ClItem, ClLiteral,
            ClLiteralKind, ClType, FuncCall, Ident,
        },
    };

    #[test]
    fn bad_expression_with_return_logs_error() {
        let mut resolver = Resolver::default();
        let expr = ClExpression::Block(ClCompoundExpression::new(
            vec![
                // This is a bad expression, should log an error, but the final return type
                // should still be returned
                ClItem::Expression(bin(expr_int(1), BinaryOperator::Add, expr_bool(true))),
            ],
            Some(expr_int(2).into()),
            fake_span(),
        ));
        let out = resolver.verify_expression_validity_and_return_typeid(&expr);
        assert!(out.is_ok(), "typing an int literal should succeed");
        assert_eq!(
            resolver.dignostics_errors.len(),
            1,
            "Error should be logged because of an invalid expression"
        );
        let got = *out.inner();
        let expect = resolver.types.intern_cltype(&cltype_int()).unwrap();
        assert_eq!(got, expect, "int literal should type to Int");
    }

    #[test]
    fn good_expression_no_error() {
        let mut resolver = Resolver::default();
        let expr = ClExpression::Block(ClCompoundExpression::new(
            vec![
                // This is a bad expression, should log an error, but the final return type
                // should still be returned
                ClItem::Expression(bin(expr_int(1), BinaryOperator::Add, expr_int(2))),
            ],
            Some(expr_int(2).into()),
            fake_span(),
        ));
        let out = resolver.verify_expression_validity_and_return_typeid(&expr);
        assert!(out.is_ok(), "typing an int literal should succeed");
        assert!(resolver.dignostics_errors.is_empty());
        let got = *out.inner();
        let expect = resolver.types.intern_cltype(&cltype_int()).unwrap();
        assert_eq!(got, expect, "int literal should type to Int");
    }

    // Check that a compound expression such as:
    //
    // ```
    // {
    //      var y: Int = 2;
    //      y
    // };
    // ```
    //
    // Evaluates to the correct type
    #[test]
    fn compound_expression_logs_inners() {
        let mut resolver = Resolver::default();
        let expr = ClExpression::Block(ClCompoundExpression::new(
            vec![
                // Make y be an integer
                ClItem::Declaration(ast::ClDeclaration::Binding(make_var("y"))),
            ],
            // Return y - We shuold know that y is an integer now
            Some(ClExpression::Identifier(Ident::new("y".to_string(), fake_span())).into()),
            fake_span(),
        ));

        let out = resolver.verify_expression_validity_and_return_typeid(&expr);
        assert!(resolver.dignostics_errors.is_empty());
        let got = *out.inner();
        let expect = resolver.types.intern_cltype(&cltype_int()).unwrap();
        assert_eq!(
            got, expect,
            "y shuold have Int type, that should have been returned"
        );
    }
    // Check that a compound expression such as:
    //
    // ```
    // {
    //      var y: String = 2;
    //      y
    // };
    // ```
    //
    // Has return type String (recovered) but logs the type mismatch error.
    #[test]
    fn compound_bad_expression_error_recovery() {
        let mut resolver = Resolver::default();
        let expr = ClExpression::Block(ClCompoundExpression::new(
            vec![
                // Make y be an integer
                ClItem::Declaration(ast::ClDeclaration::Binding(make_bad_var("y"))),
            ],
            // Return y - We shuold know that y is an integer now
            Some(ClExpression::Identifier(Ident::new("y".to_string(), fake_span())).into()),
            fake_span(),
        ));

        let out = resolver.verify_expression_validity_and_return_typeid(&expr);
        assert_eq!(
            resolver.dignostics_errors.len(),
            1,
            "Type mismatch error should have been logged"
        );

        let err = resolver.dignostics_errors[0].clone();
        let exp_err = SemanticError::WrongType {
            expected: "str".to_string(),
            actual: "int".to_string(),
            span: fake_span(),
        };
        assert_eq!(err, exp_err);

        let got = *out.inner();
        let expect = resolver.types.intern_cltype(&cltype_string()).unwrap();
        assert_eq!(
            got, expect,
            "y shuold have String type, that should have been returned"
        );
    }

    #[test]
    fn expr_literal_int_has_int_type() {
        let mut resolver = Resolver::default();

        let expr = ClExpression::Literal(integer_literal());
        let out = resolver.verify_expression_validity_and_return_typeid(&expr);

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
        let out = resolver.type_check_func_call_and_get_return_type(&fcall);
        assert!(out.is_recoverable());
        let exp = resolver.types.intern_cltype(&cltype_int()).unwrap();
        let acc = *out.inner();
        assert_eq!(exp, acc);
        assert_eq!(resolver.dignostics_errors.len(), 1, "Missing input error");
    }

    #[test]
    fn expr_identifier_uses_declared_binding_type() {
        let mut resolver = Resolver::default();

        // val x: Int = 10
        let x = make_var("x");
        assert!(resolver.push_ast_binding(&x).is_ok());

        // use x
        let expr = ClExpression::Identifier(Ident::new("x".into(), fake_span()));
        let out = resolver.verify_expression_validity_and_return_typeid(&expr);

        assert!(out.is_ok(), "typing an existing identifier should succeed");

        let got = *out.inner();
        let expect = resolver.types.intern_cltype(&cltype_int()).unwrap();
        assert_eq!(got, expect, "identifier x should have type Int");
    }

    #[test]
    fn expr_block_without_final_expr_is_unit() {
        let mut resolver = Resolver::default();

        let block = ClExpression::Block(ClCompoundExpression::new(vec![], None, fake_span()));

        let out = resolver.verify_expression_validity_and_return_typeid(&block);
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

        let out = resolver.verify_expression_validity_and_return_typeid(&if_e);
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

        let out = resolver.verify_expression_validity_and_return_typeid(&if_e);
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
        let out = r.verify_expression_validity_and_return_typeid(&bin(
            expr_int(1),
            BinaryOperator::Add,
            expr_int(2),
        ));
        assert!(out.is_ok());
        let got = *out.inner();
        let expect = r.types.intern_cltype(&cltype_int()).unwrap();
        assert_eq!(got, expect);
    }

    #[test]
    fn add_float_float_ok() {
        let mut r = Resolver::default();
        let out = r.verify_expression_validity_and_return_typeid(&bin(
            expr_real(1.0),
            BinaryOperator::Add,
            expr_real(2.0),
        ));
        assert!(out.is_ok());
        let got = *out.inner();
        let expect = r.types.intern_cltype(&cltype_float()).unwrap();
        assert_eq!(got, expect);
    }

    #[test]
    fn add_int_bool_recover_left_type() {
        let mut r = Resolver::default();
        let out = r.verify_expression_validity_and_return_typeid(&bin(
            expr_int(1),
            BinaryOperator::Add,
            expr_bool(true),
        ));
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
        let out = r.verify_expression_validity_and_return_typeid(&bin(
            expr_int(1),
            BinaryOperator::Leq,
            expr_bool(true),
        ));
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
        let out = r.verify_expression_validity_and_return_typeid(&bin(
            expr_int(1),
            BinaryOperator::EqEq,
            expr_int(1),
        ));
        assert!(out.is_ok());
        let got = *out.inner();
        let expect = r.types.intern_cltype(&cltype_bool()).unwrap();
        assert_eq!(got, expect);
    }

    #[test]
    fn eqeq_int_bool_recover_bool() {
        let mut r = Resolver::default();
        let out = r.verify_expression_validity_and_return_typeid(&bin(
            expr_int(1),
            BinaryOperator::EqEq,
            expr_bool(true),
        ));
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
        let out = r.verify_expression_validity_and_return_typeid(&bin(
            expr_int(1),
            BinaryOperator::Or,
            expr_int(2),
        ));
        assert!(out.is_ok());
        let got = *out.inner();
        let expect = r.types.intern_cltype(&cltype_int()).unwrap();
        assert_eq!(got, expect);
    }

    #[test]
    fn and_bools_ok_bool() {
        let mut r = Resolver::default();
        let out = r.verify_expression_validity_and_return_typeid(&bin(
            expr_bool(true),
            BinaryOperator::And,
            expr_bool(false),
        ));
        assert!(out.is_ok());
        let got = *out.inner();
        let expect = r.types.intern_cltype(&cltype_bool()).unwrap();
        assert_eq!(got, expect);
    }

    #[test]
    fn or_int_bool_mismatch_recover_operand_type() {
        let mut r = Resolver::default();
        let out = r.verify_expression_validity_and_return_typeid(&bin(
            expr_int(1),
            BinaryOperator::Or,
            expr_bool(true),
        ));
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
        let out = resolver.push_ast_function(&f);
        resolver.type_check_function(&f, *out.inner());
        assert!(resolver.dignostics_errors.is_empty()); // No errors emitted
    }

    #[test]
    fn test_declaration_type_missmatch_err() {
        let mut resolver = Resolver::default();
        let f = make_bad_ast_func("f"); // Type Int -> String (acc returns int)
        let out = resolver.push_ast_function(&f);
        resolver.type_check_function(&f, *out.inner());
        assert!(resolver.dignostics_errors.len() == 1); // Just one error
    }

    #[test]
    fn test_function_call_wrong_arity() {
        let mut resolver = Resolver::default();
        let f = make_ast_func("f");
        resolver.push_ast_function(&f);
        let func_call =
            &ast::FuncCall::new(Ident::new("f".into(), fake_span()), vec![], fake_span());
        resolver.type_check_func_call_and_get_return_type(func_call);
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
        resolver.type_check_func_call_and_get_return_type(func_call);
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
        resolver.type_check_func_call_and_get_return_type(func_call);
        assert_eq!(resolver.dignostics_errors.len(), 1);
    }
    #[test]
    fn test_binding_type_matches_no_error() {
        let mut resolver = Resolver::default();
        let v = make_var("v");
        let out = resolver.push_ast_binding(&v);
        resolver.type_check_binding(&v, *out.inner());
        assert!(resolver.dignostics_errors.is_empty()); // No errors emitted
    }

    #[test]
    fn test_binding_type_missmatch_err() {
        let mut resolver = Resolver::default();
        let v = make_bad_var("v");
        let out = resolver.push_ast_binding(&v);
        resolver.type_check_binding(&v, *out.inner());
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

#[cfg(test)]
mod test_declarations {
    use crate::sematic::{
        Resolver,
        test_helpers_resolver::{make_ast_id_func, make_bad_func_undefined_literal},
    };

    // Check that `def foo(x: Int): Int = x` does not give an error. `x` shuold be added to the scope.
    #[test]
    fn test_function_declaration_adds_inputs_to_scope() {
        let mut resolver = Resolver::default();
        let id = make_ast_id_func();
        let out = resolver.push_ast_function(&id);
        resolver.verify_declaration(
            &crate::syntax::ast::ClDeclaration::Function(id),
            *out.inner(),
        );
        assert!(resolver.dignostics_errors.is_empty());
    }

    // Check that `def foo(x: Int): Int = y` does give an error. `y` shuold be added to the scope.
    #[test]
    fn test_function_declaration_bad_ident() {
        let mut resolver = Resolver::default();
        let badfn = make_bad_func_undefined_literal();
        let out = resolver.push_ast_function(&badfn);
        resolver.verify_declaration(
            &crate::syntax::ast::ClDeclaration::Function(badfn),
            *out.inner(),
        );
        assert_eq!(
            resolver.dignostics_errors.len(),
            1,
            "Bad literal when returning, but we should still have Recoverable Int type"
        );
    }
}

#[cfg(test)]
mod test_exprs {
    use crate::{
        sematic::{
            Resolver,
            test_helpers_resolver::{fake_span, integer_literal, make_var},
        },
        syntax::ast::{ClExpression, FuncCall, Ident},
    };

    /// Calling a non function should log an error
    ///
    /// ```
    /// var x: Int = 10;
    /// x(5);               -- Error!
    /// ```
    #[test]
    fn call_non_function() {
        let mut resolver = Resolver::default();
        let var = make_var("x"); // var x: Int = 10;
        // x(5)
        let call = ClExpression::FunctionCall(FuncCall::new(
            Ident::new("x".into(), fake_span()),
            vec![ClExpression::Literal(integer_literal())],
            fake_span(),
        ));
        resolver.push_ast_binding(&var);
        resolver.verify_expression_validity_and_return_typeid(&call);
        assert_eq!(resolver.dignostics_errors.len(), 1)
    }
}
