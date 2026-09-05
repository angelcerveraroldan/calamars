use crate::{
    StringArena,
    data_structs::{DStructArena, StructDefArena},
    ids,
    memory::MemoryLayoutArena,
    types::TypeArena,
};

pub struct TypeInterner<'a> {
    pub types: &'a mut TypeArena,
    pub struct_defs: &'a StructDefArena,
}

pub struct TypeDb<'a> {
    pub types: &'a TypeArena,
    pub struct_defs: &'a StructDefArena,
}

impl<'a> TypeDb<'a> {
    pub fn get_type_unchecked(&self, type_id: ids::TypeId) -> &crate::types::Type {
        self.types.get_unchecked(type_id)
    }

    pub fn get_typeid_unchecked(&self, ty: &crate::types::Type) -> &ids::TypeId {
        self.types.resolve_unchecked(ty)
    }
}

/// Context and information that needs to be filled and
/// shared between many stages of the compiler
pub struct GlobalContext {
    pub types: TypeArena,
    pub data_structs: DStructArena,
    pub struct_defs: StructDefArena,
    pub strings: StringArena,
    pub memlay: MemoryLayoutArena,
    pub struct_mem: hashbrown::HashMap<ids::DStructId, ids::MemLayoutId>,
}

impl GlobalContext {
    pub fn type_ctx(&mut self) -> TypeInterner<'_> {
        TypeInterner {
            types: &mut self.types,
            struct_defs: &self.struct_defs,
        }
    }

    pub fn type_db(&self) -> TypeDb<'_> {
        TypeDb {
            types: &self.types,
            struct_defs: &self.struct_defs,
        }
    }
}
