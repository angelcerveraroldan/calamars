//! Goal:
//!
//! Parse all the headers of all functions. We need to make a
//! context that will later be used for type checking the function
//! bodies.
//!
//! We will also need to be able to export functions, so this context
//! could be kept alive for later.

use calamars_core::{data_structs, ids, types};

use crate::{
    semantic::{SemLogger, error::SemanticError, lower::lower_type},
    syntax::{ast, span::Span},
};

/// A module level source declaration. This exist even if the body is
/// not defined of ill-defined
pub type DeclKeyId = usize;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct DeclKey {
    module: ids::FileId,
    // TODO: Maybe this should not be a raw string...
    // It would be nice for this to impl Copy since we need to clone it a little bit
    name: String,
}

impl DeclKey {
    fn new(module: ids::FileId, name: impl Into<String>) -> Self {
        Self {
            module,
            name: name.into(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct DeclSignature {
    pub dtyp: ids::TypeId,
    pub dkey: DeclKey,
    pub name_span: Span,
    pub type_span: Span,
}

fn declsign_from_ast_info<'a>(
    declaration_type: &ast::Type,
    name: &ast::Ident,
    types: &'a mut types::TypeArena,
    data_structs: &data_structs::DStructArena,
    module: ids::FileId,
) -> SemLogger<DeclSignature> {
    let name_span = name.span();
    let type_span = declaration_type.span();
    lower_type(declaration_type, types, data_structs, module).map(|dtyp| DeclSignature {
        dkey: DeclKey::new(module, name.ident()),
        dtyp,
        name_span,
        type_span,
    })
}

/// Lower every single declaration without checking for uniqueness
fn lower_all_declarations(
    module: ids::FileId,
    module_declarations: &[ast::Declaration],
    types: &mut types::TypeArena,
    data_structs: &data_structs::DStructArena,
) -> SemLogger<Vec<DeclSignature>> {
    let vsem = module_declarations
        .iter()
        .filter_map(|dec| match dec {
            ast::Declaration::TypeSignature { name, dtype, .. }
            | ast::Declaration::TypeAndBinding { name, dtype, .. } => {
                declsign_from_ast_info(dtype, name, types, data_structs, module).into()
            }
            ast::Declaration::Binding { .. } => None,
        })
        .collect();

    SemLogger::flatten(vsem)
}

/// All the top-level declarations in a module
pub struct ModuleDeclCtx {
    module: ids::FileId,
    // TODO: Something like this needs to be implemented in our arena
    //       module. It needs a large refactor, so we'll add it when
    //       that takes place. These id's are a 'placeholder' for when
    //       we lower the bodies
    declaration_keys: hashbrown::HashMap<DeclKey, DeclKeyId>,
    declarations: Vec<DeclSignature>,
}

impl ModuleDeclCtx {
    pub fn lower_module<'a>(
        module: ids::FileId,
        module_declarations: &[ast::Declaration],
        types: &'a mut types::TypeArena,
        data_structs: &data_structs::DStructArena,
    ) -> SemLogger<Self> {
        // Lower all declarations
        let logger = lower_all_declarations(module, module_declarations, types, data_structs);
        // De-duplicate declarations and log errors
        logger.map_werrs(|declarations| {
            let mut kmap = hashbrown::HashMap::<DeclKey, DeclKeyId>::new();
            let mut decs: Vec<DeclSignature> = vec![];
            let mut logs: Vec<SemanticError> = vec![];

            for decl_signature in declarations.into_iter() {
                let dkey = decl_signature.dkey.clone();
                if let Some(old_key) = kmap.get(&dkey) {
                    let og_declaration: &DeclSignature = decs
                        .get(*old_key)
                        .expect("This key was already inserted, so it must exist");
                    logs.push(SemanticError::Redeclaration {
                        original_span: og_declaration.name_span,
                        redec_span: decl_signature.name_span,
                    })
                } else {
                    decs.push(decl_signature);
                    let index = decs.len() - 1;
                    kmap.insert(dkey, index);
                }
            }

            SemLogger::new(
                Self {
                    module,
                    declaration_keys: kmap,
                    declarations: decs,
                },
                logs,
            )
        })
    }

    pub fn declaration_from_name(&self, name: impl Into<String>) -> Option<&DeclSignature> {
        let name: String = name.into();
        let key = DeclKey::new(self.module, name);
        let key_id = self.declaration_keys.get(&key)?;

        let dec = self.declarations.get(*key_id);
        debug_assert!(
            dec.is_some(),
            "declaration ids and their actual declarations need to be aligned"
        );
        let dec = dec.unwrap();
        debug_assert_eq!(
            dec.dkey, key,
            "declaration ids and their actual declarations need to be aligned"
        );
        dec.into()
    }

    pub fn declaration_id_from_name(&self, name: impl Into<String>) -> Option<DeclKeyId> {
        let key = DeclKey::new(self.module, name);
        self.declaration_keys.get(&key).copied()
    }

    pub fn declaration(&self, id: DeclKeyId) -> Option<&DeclSignature> {
        self.declarations.get(id)
    }
}
