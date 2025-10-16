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

#[derive(Debug, Default)]
pub struct Resolver {
    types: TypeArena,

    symbols: SymbolArena,
    scopes: Vec<SymbolScope>,
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
    fn push_ast_function(&mut self, node: ast::ClFuncDec) -> Result<SymbolId, SemanticError> {
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
    fn push_ast_binding(&mut self, node: ast::ClBinding) -> Result<SymbolId, SemanticError> {
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
}
