use std::collections::HashMap;

use crate::{
    sematic::types::TypeId,
    syntax::{ast, span::Span},
};

#[derive(Debug, Copy, Clone)]
pub struct SymbolId(usize);

#[derive(Debug)]
pub enum DefKind {
    Var, // The only mutable def
    Val,
    Param,
    Func,
}

#[derive(Debug)]
pub struct Symbol {
    /// Name of the symbol
    pub name: String,
    pub kind: DefKind,
    /// Number of params this symbol accepts as its input
    pub arity: Option<u16>,
    /// Type of the symbol
    pub ty: Option<TypeId>,
    /// Span to the identifier for this symbol
    pub name_span: Span,
    /// Span of the entire declaration for this symbol
    pub decl_span: Span,
}

/// An arena containing all of the symbols
#[derive(Debug, Default)]
pub struct SymbolArena {
    arena: Vec<Symbol>,
}

impl SymbolArena {
    pub fn insert(&mut self, sym: Symbol) -> SymbolId {
        self.arena.push(sym);
        SymbolId(self.arena.len() - 1)
    }

    pub fn get(&self, id: &SymbolId) -> Option<&Symbol> {
        self.arena.get(id.0)
    }

    pub fn get_mut(&mut self, id: &SymbolId) -> Option<&mut Symbol> {
        self.arena.get_mut(id.0)
    }

    pub fn len(&self) -> usize {
        self.arena.len()
    }
}

#[derive(Debug, Default)]
pub struct SymbolScope {
    pub map: HashMap<String, SymbolId>,
}
