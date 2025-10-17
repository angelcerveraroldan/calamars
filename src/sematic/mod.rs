use chumsky::container::Seq;

use crate::{
    sematic::{
        error::SemanticError,
        symbols::{DefKind, Symbol, SymbolArena, SymbolId, SymbolScope},
        types::TypeArena,
    },
    syntax::{ast, span::Span},
};

pub mod error;
pub mod symbols;
pub mod types;

#[derive(Debug)]
pub struct Resolver {
    types: TypeArena,

    symbols: SymbolArena,
    scopes: Vec<SymbolScope>,
}

impl Default for Resolver {
    fn default() -> Self {
        Self {
            types: TypeArena::default(),
            symbols: SymbolArena::default(),
            scopes: vec![SymbolScope::default()],
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

    /// Insert symbol into the current scope. The symbols id will be returned if correctly
    /// inserted. Nothing will be returned if the type already existed (error).
    pub fn declare(&mut self, sym: Symbol) -> Result<SymbolId, SemanticError> {
        let curr_scope = self.scopes.last_mut().unwrap();
        if let Some(original) = curr_scope.map.get(&sym.name) {
            let original = self.symbols.get(original).unwrap();
            return Err(SemanticError::Redeclaration {
                original_span: original.name_span,
                redec_span: sym.name_span,
            });
        }

        let name = sym.name.clone();
        let id = self.symbols.insert(sym);
        curr_scope.map.insert(name, id);
        Ok(id)
    }

    /// Find the symbol id for a symbol with a given name. This will look though the current scope,
    /// and then though parent scopes and return the first match. If no matches are found, then an
    /// error will be returned.
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

    /// Given some function declaration in the ast, add it to the symbol table in the current
    /// scope.
    fn push_ast_function(&mut self, node: &ast::ClFuncDec) -> Result<SymbolId, SemanticError> {
        let ty = node.fntype().clone();
        let ty = self.types.intern_cltype(&ty)?;
        let sym = Symbol {
            name: node.name().clone(),
            kind: symbols::DefKind::Func,
            arity: Some(node.airity()),
            ty: Some(ty),
            name_span: node.name_span(),
            decl_span: node.span(),
        };
        self.declare(sym)
    }

    /// Given some binding in the ast, add it to the symbol table in the current scope.
    fn push_ast_binding(&mut self, node: &ast::ClBinding) -> Result<SymbolId, SemanticError> {
        #[rustfmt::skip]
        let kind = if node.mutable { DefKind::Var } else { DefKind::Val };
        let ty = self.types.intern_cltype(&node.vtype)?;
        let sym = Symbol {
            name: node.vname.ident().into(),
            kind,
            arity: None,
            ty: Some(ty),
            name_span: node.name_span(),
            decl_span: node.span(),
        };
        self.declare(sym)
    }

    pub fn push_ast_declaration(
        &mut self,
        node: &ast::ClDeclaration,
    ) -> Result<SymbolId, SemanticError> {
        match &node {
            ast::ClDeclaration::Binding(node_bind) => self.push_ast_binding(node_bind),
            ast::ClDeclaration::Function(node_fn) => self.push_ast_function(node_fn),
        }
    }
}

#[cfg(test)]
mod test_resolver {
    use chumsky::{Parser, span::SimpleSpan};

    use crate::{
        parser::parse_cl_item,
        sematic::Resolver,
        syntax::{
            ast::{self, ClBinding, ClLiteral, ClType, Ident},
            token::Token,
        },
    };

    fn fake_span() -> SimpleSpan {
        SimpleSpan {
            start: 0,
            end: 0,
            context: (),
        }
    }

    fn cltype_int() -> ClType {
        ClType::Path {
            segments: vec![Ident::new("Int".to_string(), fake_span())],
            span: fake_span(),
        }
    }

    fn integer_literal() -> ClLiteral {
        ClLiteral::new(ast::ClLiteralKind::Integer(10), fake_span())
    }

    fn make_ast_func(name: &str) -> ast::ClFuncDec {
        let fake = fake_span();

        ast::ClFuncDec::new(
            Ident::new(name.to_string(), fake),
            vec![(Ident::new("x".to_string(), fake), cltype_int())],
            cltype_int(),
            ast::ClExpression::Literal(integer_literal()),
            fake_span(),
            None,
        )
    }

    fn make_var(name: &str) -> ast::ClBinding {
        ClBinding::new(
            Ident::new(name.to_string(), fake_span()),
            cltype_int(),
            Box::new(ast::ClExpression::Literal(integer_literal())),
            false,
            fake_span(),
        )
    }

    #[test]
    fn happy_resolver() {
        let mut resolver = Resolver::default();
        let f = make_ast_func("f");
        let y = make_var("y");
        let g = make_ast_func("g");

        resolver.push_ast_function(&f).unwrap();
        resolver.push_ast_binding(&y).unwrap();
        resolver.push_ast_function(&g).unwrap();

        assert_eq!(resolver.symbols.len(), 3);
    }

    #[test]
    fn phase_a_duplicate_var_in_same_scope_is_error() {
        let mut resolver = Resolver::default();
        let a1 = make_var("a");
        let a2 = make_var("a");

        assert!(resolver.push_ast_binding(&a1).is_ok());
        let dup = resolver.push_ast_binding(&a2);
        assert!(dup.is_err(), "expected duplicate var error");
        assert_eq!(resolver.symbols.len(), 1);
    }

    #[test]
    fn phase_a_duplicate_func_in_same_scope_is_error() {
        let mut resolver = Resolver::default();
        let f1 = make_ast_func("f");
        let f2 = make_ast_func("f");

        assert!(resolver.push_ast_function(&f1).is_ok());
        let dup = resolver.push_ast_function(&f2);
        assert!(dup.is_err(), "expected duplicate func error");
        assert_eq!(resolver.symbols.len(), 1);
    }

    #[test]
    fn phase_a_function_and_var_same_name() {
        let mut resolver = Resolver::default();
        let f = make_ast_func("f");
        let v = make_var("f");
        assert!(resolver.push_ast_function(&f).is_ok());
        let dup = resolver.push_ast_binding(&v);
        assert!(dup.is_err(), "expected duplicate name error");
        assert_eq!(resolver.symbols.len(), 1);
    }
}
