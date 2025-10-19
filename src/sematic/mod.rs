use std::{any::Any, fmt::Debug, process::id};

use chumsky::{container::Seq, span::SimpleSpan, text::ascii::ident};

use crate::{
    sematic::{
        error::SemanticError,
        symbols::{DefKind, Symbol, SymbolArena, SymbolId, SymbolScope},
        types::{Type, TypeArena, TypeId},
    },
    syntax::{
        ast::{self, ClCompoundExpression, Ident},
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

    /// Given some expression, return the TypeId of the expressions return type
    pub fn ast_expression_type(&mut self, node: &ast::ClExpression) -> ResolverTypeOut {
        match node {
            ast::ClExpression::Literal(cl_literal) => self.ast_literal_type(cl_literal),
            ast::ClExpression::Identifier(ident) => self.ast_identifier_type(ident),
            ast::ClExpression::UnaryOp(cl_unary_op) => todo!(),
            ast::ClExpression::BinaryOp(cl_binary_op) => todo!(),
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
}

#[cfg(test)]
mod test_helpers_resolver {
    use chumsky::{Parser, span::SimpleSpan};

    use crate::{
        parser::parse_cl_item,
        sematic::Resolver,
        syntax::{
            ast::{self, ClBinding, ClLiteral, ClType, Ident},
            token::Token,
        },
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

    pub fn integer_literal() -> ClLiteral {
        ClLiteral::new(ast::ClLiteralKind::Integer(10), fake_span())
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
        sematic::Resolver,
        syntax::ast::{FuncCall, Ident},
    };

    #[test]
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

        let exp_type = resolver.types.get(exp);
        let acc_type = resolver.types.get(acc);
        println!("Acc: {:?} vs Exp: {:?}", acc_type, exp_type);

        assert_eq!(exp, acc);
    }
}
