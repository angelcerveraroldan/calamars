//! Lower top new top-level type definitions, for now this is just
//! structs, but will extend to enums and type aliases later
//!
//! The general idea is:
//! - gather rich information of one module
//! - combine all the data from the modules into a CTX
//!
//! A ctx is front-end global context, we will build them up until we
//! have all checks done

use calamars_core::{
    data_structs::{DStructArena, DataStructureKey},
    ids::{self, DStructId},
    types::{self, TypeArena},
};

use crate::{
    semantic::error::SemanticError,
    syntax::{ast, span::Span},
};

/// The top level types in a module
#[derive(Clone, Debug)]
pub struct TopLvlTypes {
    module_id: ids::FileId,
    typedefns: Vec<TypeDefn>,
}

#[derive(Clone, Debug)]
pub struct TypeDefn {
    module_id: ids::FileId,
    name: String,
    span: Span,
}

impl TypeDefn {
    pub fn new(module_id: ids::FileId, name: String, span: Span) -> Self {
        Self {
            module_id,
            name,
            span,
        }
    }

    fn as_key(&self) -> DataStructureKey {
        DataStructureKey {
            name: self.name.clone(),
            module: self.module_id,
        }
    }
}

impl From<(&ast::Definition, ids::FileId)> for TypeDefn {
    fn from((def, module): (&ast::Definition, ids::FileId)) -> Self {
        Self::new(module, def.name().ident().into(), def.name_span())
    }
}

impl TopLvlTypes {
    pub fn new<'a>(ast: &'a ast::Module, module_id: ids::FileId) -> Self {
        let typedefns = ast
            .definitions
            .iter()
            .map(|d| Into::into((d, module_id)))
            .collect();

        Self {
            typedefns,
            module_id,
        }
    }
}

/// A global context that will be filled to contain all the to level
/// declarations of all modules
pub struct TopLvlTypesCtx {
    pub type_arena: TypeArena,
    pub data_structs: DStructArena,
    pub errors: Vec<SemanticError>,
    pub rich_typedefn_info: hashbrown::HashMap<DStructId, TypeDefn>,
}

impl TopLvlTypesCtx {
    fn empty() -> Self {
        Self {
            type_arena: TypeArena::default(),
            data_structs: DStructArena::new_unchecked(),
            rich_typedefn_info: hashbrown::HashMap::new(),
            errors: vec![],
        }
    }

    fn insert_module(&mut self, mod_types: &TopLvlTypes) {
        for tp in &mod_types.typedefns {
            let key = tp.as_key();
            match self.data_structs.intern_checked(&key) {
                calamars_core::InternedId::New(id) => {
                    self.rich_typedefn_info.insert(id, tp.clone());
                    let ty = types::Type::Structure(id);
                    self.type_arena.intern(&ty);
                    id
                }
                calamars_core::InternedId::Old(id) => {
                    let old_span = self.rich_typedefn_info.get(&id).unwrap().span;
                    self.errors.push(SemanticError::Redeclaration {
                        original_span: old_span,
                        redec_span: tp.span,
                    });
                    id
                }
            };
        }
    }

    pub fn from_modules(modules: impl IntoIterator<Item = TopLvlTypes>) -> Self {
        let mut s = Self::empty();
        for mod_types in modules {
            s.insert_module(&mod_types);
        }
        s
    }
}
